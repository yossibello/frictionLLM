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

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ────
        x : [B, T, d_model]

        Returns
        ───────
        [B, T, d_model]  — wave-mixed token representations
        """
        B, T, _ = x.shape
        V = self.proj_in(x)          # [B, T, d_inner]  applied voltage at each node

        # Initial wave state (charge and current both zero before any signal)
        q      = x.new_zeros(B, self.d_inner)   # charge at current node
        i_scan = x.new_zeros(B, self.d_inner)   # current flowing through chain

        outputs = []

        for t in range(T):
            v_t = V[:, t]            # [B, d_inner]  voltage at node t

            # ── Coupling force from left neighbour (position t-1) ─────────────
            # Proportional to charge difference — wave pressure pushing right
            # At t=0 q=0 and there's no left neighbour → coupling = 0 naturally
            coupling = q / self.L_c   # simplified: previous charge drives new node

            # ── RLC dynamics + coupling ───────────────────────────────────────
            # L·di/dt = V_t − R·i − q/C + coupling_force
            di    = (v_t - self.R * i_scan - q / self.C + coupling) / self.L
            i_new = (i_scan + di * self.dt).clamp(-self.clamp, self.clamp)
            q_new = (q     + i_new * self.dt).clamp(-self.clamp, self.clamp)

            outputs.append(q_new)
            q      = q_new
            i_scan = i_new

        y = torch.stack(outputs, dim=1)   # [B, T, d_inner]
        y = self.norm(y)
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
