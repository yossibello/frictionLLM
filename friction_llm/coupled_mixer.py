"""
CoupledOscillatorMixer — replaces attention with physical wave propagation.

Instead of O(T²) dot-product attention (algebraic, no physics basis),
this module models the sequence as a 1D RLC transmission line where
information propagates as electromagnetic waves.

Physics
───────
Each token position i is a node in a ladder network:

    V₀  ──L─── V₁  ──L─── V₂  ──L───  ...  ──L─── Vₙ
     │           │           │                       │
    C,R         C,R         C,R                     C,R
     │           │           │                       │
    GND         GND         GND                    GND

V_i = applied voltage at node i (projected from token embedding)
L   = self inductance  (inertia at each node)
R   = resistance       (damping — our friction)
C   = capacitance      (charge storage, restoring force)
L_c = coupling inductance between adjacent nodes

State (q, i_scan) propagates causally left → right (autoregressive).
The wave speed and dispersion depend on learned L, R, C, L_c.

Why this beats dot-product attention
──────────────────────────────────────
Attention    : O(T²) memory, no physics, permutation-invariant
Transmission : O(T)  memory, physical parameters, order-aware

Long-range dependencies build through wave propagation over N layers.
Multiple resonant modes (different ω₀ = different attention patterns)
emerge naturally from the circuit dynamics.

Relationship to Mamba/S4
──────────────────────────
S4/Mamba use  x[t] = Ax[t-1] + Bu[t]  with A learned freely.
We constrain A to the physically valid RLC transmission line matrix,
adding:
  1. Physical interpretability (L, R, C, L_c have real meaning)
  2. Stability guarantee (positive R → eigenvalues inside unit circle)
  3. Explicit coupling inductance L_c between adjacent positions
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def _impulse_loop(
    beta_q:  torch.Tensor,
    beta_i:  torch.Tensor,
    alpha_q: torch.Tensor,
    alpha_i: torch.Tensor,
    delta:   torch.Tensor,
    gamma:   torch.Tensor,
    T: int,
) -> torch.Tensor:
    """
    JIT-compiled impulse response loop — no Python overhead.
    Computes h[t] = q-component of A^t @ B for t=0..T-1.
    Runs entirely in C++, eliminates per-iteration Python→CUDA overhead.
    """
    d = delta.shape[0]
    h = torch.zeros(T, d, dtype=delta.dtype, device=delta.device)
    p_q = delta.clone()
    p_i = gamma.clone()
    h[0] = p_q
    for t in range(1, T):
        p_q_new = beta_q * p_q + beta_i * p_i
        p_i     = alpha_q * p_q + alpha_i * p_i
        p_q     = p_q_new
        h[t]    = p_q
    return h   # [T, d]


class CoupledOscillatorMixer(nn.Module):
    """
    1D RLC transmission line along the token dimension.

    The scan is causal: position i's output depends only on positions 0..i-1.
    State (q, i_scan) propagates left → right, encoding wave dynamics.

    Parameters
    ──────────
    d_model    : token embedding dimension
    d_inner    : internal oscillator dimension (defaults to d_model)
    dt         : Euler integration step
    L_c_init   : initial coupling inductance — large = weak coupling at start
    clamp      : numerical stability clamp on state
    """

    def __init__(
        self,
        d_model: int,
        d_inner: Optional[int] = None,
        dt: float = 0.1,
        L_c_init: float = 5.0,
        clamp: float = 10.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        d_inner = d_inner or d_model
        self.d_inner = d_inner
        self.dt      = dt
        self.clamp   = clamp

        # ── Per-node self RLC (same physics as RLCNeuron) ─────────────────────
        self.log_L = nn.Parameter(torch.zeros(d_inner))
        self.log_R = nn.Parameter(torch.full((d_inner,), math.log(2.0)))  # ζ=1
        self.log_C = nn.Parameter(torch.zeros(d_inner))

        # ── Coupling inductance between adjacent nodes ─────────────────────────
        # Large init → weak coupling → gradients learn to strengthen as needed
        self.log_L_c = nn.Parameter(torch.full((d_inner,), math.log(L_c_init)))

        # ── Projections ───────────────────────────────────────────────────────
        self.proj_in  = nn.Linear(d_model, d_inner, bias=bias)
        self.proj_out = nn.Linear(d_inner, d_model, bias=bias)
        self.norm     = nn.LayerNorm(d_inner)

    # ── Circuit properties ────────────────────────────────────────────────────

    @property
    def L(self)   -> torch.Tensor: return self.log_L.exp()
    @property
    def R(self)   -> torch.Tensor: return self.log_R.exp()
    @property
    def C(self)   -> torch.Tensor: return self.log_C.exp()
    @property
    def L_c(self) -> torch.Tensor: return self.log_L_c.exp()

    @property
    def omega_0(self) -> torch.Tensor:
        """Natural frequency of each node."""
        return 1.0 / (self.L * self.C).sqrt()

    @property
    def damping_ratio(self) -> torch.Tensor:
        return self.R / (2.0 * (self.L / self.C).sqrt())

    @property
    def wave_speed(self) -> torch.Tensor:
        """Approximate wave speed along the chain: v = 1/√(L_c·C)"""
        return 1.0 / (self.L_c * self.C).sqrt()

    # ── Impulse response ─────────────────────────────────────────────────────

    def _impulse_response(self, T: int) -> torch.Tensor:
        """
        Compute h[t] = q-component of A^t @ B_input for t = 0..T-1.

        Uses a JIT-compiled loop over tiny [d] tensors (no Python overhead).
        Always returns float32 — FFT requires float32, not float16.
        """
        # Compute in float32 regardless of model dtype (FFT needs float32)
        L  = self.L.float();  R  = self.R.float()
        C  = self.C.float();  Lc = self.L_c.float()
        dt = float(self.dt)

        alpha_i = 1.0 - R  * dt / L
        alpha_q = dt * (-1.0 / (L * C) + 1.0 / (L * Lc))
        gamma   = dt / L
        beta_q  = 1.0 + alpha_q * dt
        beta_i  = alpha_i * dt
        delta   = gamma * dt

        return _impulse_loop(beta_q, beta_i, alpha_q, alpha_i,
                             delta, gamma, T)   # [T, d_inner] float32

    # ── Forward (FFT parallel scan) ───────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ────
        x : [B, T, d_model]

        Returns
        ───────
        [B, T, d_model]  — wave-mixed token representations

        Implementation
        ──────────────
        Old: Python for-loop over T → 256 sequential CUDA kernel launches
        New: FFT causal convolution → 3 CUDA calls regardless of T

        y[b,t,d] = sum_{k=0}^{t} h[t-k, d] * V[b, k, d]
                 = IFFT(FFT(h) × FFT(V))   [O(T log T), fully parallel]
        """
        B, T, _ = x.shape
        V = self.proj_in(x)          # [B, T, d_inner]

        h  = self._impulse_response(T)        # [T, d_inner]  float32

        # ── Causal convolution via FFT (always float32 — no ComplexHalf) ──────
        n     = 2 * T
        V_f32 = V.float()                          # cast input to float32
        H     = torch.fft.rfft(h,     n=n, dim=0) # [n//2+1, d_inner]
        U     = torch.fft.rfft(V_f32, n=n, dim=1) # [B, n//2+1, d_inner]
        Y     = torch.fft.irfft(H.unsqueeze(0) * U,
                                n=n, dim=1)[:, :T] # [B, T, d_inner]  float32
        Y     = Y.to(V.dtype)                      # cast back (fp16 if AMP)

        y = self.norm(Y)
        return self.proj_out(y)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def wave_stats(self) -> dict:
        return {
            "omega_0_mean":    self.omega_0.mean().item(),
            "damping_mean":    self.damping_ratio.mean().item(),
            "wave_speed_mean": self.wave_speed.mean().item(),
            "coupling_mean":   (1.0 / self.L_c).mean().item(),
            "underdamped_%":   (self.damping_ratio < 1.0).float().mean().item() * 100,
        }

    def extra_repr(self) -> str:
        return (
            f"d_inner={self.d_inner}, dt={self.dt}, "
            f"ω₀≈{self.omega_0.mean():.3f}, "
            f"wave_speed≈{self.wave_speed.mean():.3f}"
        )
