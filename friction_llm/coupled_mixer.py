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
We constrain A to the physically valid RLC oscillator, adding:
  1. Physical interpretability (L, R, C, L_c have real meaning)
  2. Stability guarantee: the kernel is the exact analytic Green's function
     of a damped oscillator, h(t) ∝ e^{−γt}·(…) with γ = R/(2L) > 0, so it
     decays for every t and every parameter value — unconditionally stable.
  3. Coupling inductance L_c adds restoring stiffness to each node
     (ω² = (1/C + 1/L_c)/L), tuning the resonant frequency of the chain.

Note: this is a per-channel diagonal SSM (each channel is an independent
2-pole resonator convolved along the sequence), not a literal spatial
ladder — the "transmission line" is the mental model, the implementation is
an LTI causal convolution, same family as S4D.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoupledOscillatorMixer(nn.Module):
    """
    Multi-pole RLC resonator bank along the token dimension, with a
    content-dependent (selective) output gate.

    Each channel is no longer a single 2-pole oscillator but a BANK of
    `n_poles` damped resonators at diverse natural frequencies, whose impulse
    responses are mixed by learnable weights:

        h_d(t) = Σ_p  w[d,p] · h_{d,p}(t)        (h_{d,p} = analytic RLC kernel)

    This is the S4D upgrade: a single 2-pole filter per channel can only
    express one resonant mode, which is far weaker than attention; a bank of
    poles spanning a frequency range lets one channel capture both fast/local
    and slow/global structure.

    Selective gate (Mamba-style)
    ────────────────────────────
    The convolution itself is LTI (kernel is the same for every input).  We
    add a content-dependent multiplicative gate on the SSM output:

        y = proj_out( norm(h * V) ⊙ SiLU(proj_gate(x)) )

    This recovers most of the input-dependent "selectivity" that makes Mamba
    beat plain S4, while keeping the O(T log T) FFT convolution (true
    input-dependent state matrices would force a sequential scan).

    Parameters
    ──────────
    d_model    : token embedding dimension
    d_inner    : internal oscillator dimension (defaults to d_model)
    n_poles    : resonators per channel (frequencies spread at init)
    dt         : kernel sampling step
    L_c_init   : initial coupling inductance (adds restoring stiffness)
    clamp      : unused (kernel is unconditionally stable; kept for config compat)
    """

    def __init__(
        self,
        d_model: int,
        d_inner: Optional[int] = None,
        n_poles: int = 8,
        dt: float = 0.1,
        L_c_init: float = 5.0,
        clamp: float = 10.0,
        bias: bool = True,
        param_mode: str = "rlc",
    ) -> None:
        super().__init__()
        assert param_mode in ("rlc", "free"), param_mode
        d_inner = d_inner or d_model
        self.d_inner    = d_inner
        self.n_poles    = n_poles
        self.dt         = dt
        self.clamp      = clamp
        self.param_mode = param_mode

        # Diverse natural frequencies at init (shared by both modes)
        omega = torch.logspace(math.log10(0.5), math.log10(10.0), n_poles)  # [P]

        if param_mode == "rlc":
            # ── Physics parameterization: poles derived from L, R, C, L_c ──────
            #   ω₀ = 1/√(LC); fix L=1 ⇒ C = 1/ω₀² ⇒ log_C = −2·log ω₀
            log_C = (-2.0 * omega.log()).unsqueeze(0).repeat(d_inner, 1)    # [d,P]
            self.log_L   = nn.Parameter(torch.zeros(d_inner, n_poles))
            self.log_R   = nn.Parameter(torch.full((d_inner, n_poles), math.log(2.0)))
            self.log_C   = nn.Parameter(log_C)
            self.log_L_c = nn.Parameter(torch.full((d_inner, n_poles), math.log(L_c_init)))
        else:
            # ── Free S4D-style poles: s = −softplus(raw_gamma) ± i·omega_im ─────
            # Same structure as RLC mode but poles are unconstrained learnables
            # (this is the ablation baseline: "does the physics do work?").
            # Init to match the RLC init: γ≈1, ω_d spread like the RLC kernel.
            omega_d = (omega**2 + 0.2 - 1.0).clamp(min=0.04).sqrt()         # [P]
            raw_gamma_init = math.log(math.expm1(1.0))                      # softplus⁻¹(1)
            self.raw_gamma = nn.Parameter(torch.full((d_inner, n_poles), raw_gamma_init))
            self.omega_im  = nn.Parameter(omega_d.unsqueeze(0).repeat(d_inner, 1))

        # ── Per-channel mixing weights over the pole bank ─────────────────────
        # Init 1/n_poles ⇒ kernel starts as the average of the bank (O(1) scale).
        self.pole_mix = nn.Parameter(torch.full((d_inner, n_poles), 1.0 / n_poles))

        # ── Projections ───────────────────────────────────────────────────────
        self.proj_in   = nn.Linear(d_model, d_inner, bias=bias)
        self.proj_gate = nn.Linear(d_model, d_inner, bias=bias)   # selective gate
        self.proj_out  = nn.Linear(d_inner, d_model, bias=bias)
        self.norm      = nn.LayerNorm(d_inner)

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

    # ── Poles (shared by FFT and recurrent paths) ─────────────────────────────

    def _poles(self):
        """
        Continuous-time poles (s1, s2) and output scale, all [d_inner, n_poles]
        complex64. The ONLY thing that differs between the RLC and free-pole
        modes — everything downstream (kernel, recurrence, gating) is identical.

          rlc  : s± = −γ ± √(γ²−ω²),  γ=R/2L,  ω²=(1/C+1/L_c)/L   (physics)
          free : s± = −softplus(raw_γ) ± i·ω_im                   (S4D ablation)
        """
        if self.param_mode == "free":
            gamma = F.softplus(self.raw_gamma).float()             # [d,P] >0
            omega = self.omega_im.float()                          # [d,P]
            g  = (-gamma).to(torch.complex64)
            iw = (1j * omega).to(torch.complex64)
            s1, s2 = g + iw, g - iw
            scale  = torch.full_like(gamma, float(self.dt)).to(torch.complex64)
        else:
            L  = self.L.float();  R  = self.R.float()
            C  = self.C.float();  Lc = self.L_c.float()
            omega2 = (1.0 / C + 1.0 / Lc) / L                      # [d,P] >0
            gamma  = R / (2.0 * L)                                  # [d,P] >0
            disc   = (gamma * gamma - omega2).to(torch.complex64)
            sq     = torch.sqrt(disc)
            g      = (-gamma).to(torch.complex64)
            s1, s2 = g + sq, g - sq
            scale  = (float(self.dt) / L).to(torch.complex64)
        return s1, s2, scale

    # ── Impulse response ─────────────────────────────────────────────────────

    def _impulse_response(self, T: int) -> torch.Tensor:
        """
        Exact analytic impulse response of the damped oscillator, sampled at
        t = n·dt for n = 0..T-1.  Returns [T, d_inner] float32.

        h(t) = scale·(e^{s₊t} − e^{s₋t})/(s₊ − s₋),  poles from _poles().
        Unconditionally stable (Re[s] < 0 in both modes ⇒ e^{Re[s]·t} decays).
        Always float32 — FFT requires it.

        Continuous-time per-channel ODE
        ────────────────────────────────
            q̈ + 2γ q̇ + ω² q = (1/L)·δ(t)

            ω² = (1/C + 1/L_c) / L   effective stiffness — the coupling
                                      inductance L_c ADDS restoring stiffness
                                      to the node (passive ⇒ always > 0)
            γ  = R / (2L)            decay rate (R>0, L>0 ⇒ always > 0)

        Green's function (valid for under-, over-, and critically-damped):
            h(t) = (1/L)·(e^{s₊t} − e^{s₋t}) / (s₊ − s₋),   s± = −γ ± √(γ²−ω²)

        Using complex √ makes one expression cover all damping regimes
        (complex conjugate roots → underdamped sine, real roots → overdamped).

        Why this replaced the old symplectic-Euler scan
        ────────────────────────────────────────────────
        The previous loop discretised the ODE explicitly, which is only
        CONDITIONALLY stable: it diverged to ±inf once ω·dt > 2 or once the
        (mis-signed) coupling term flipped the spring negative — both reachable
        by the unconstrained log-space parameters during training, producing
        NaN loss.  Because γ > 0 here, e^{−γt} decays for every t, so this
        kernel is UNCONDITIONALLY stable for any L, R, C, L_c the optimiser
        chooses.  No clamp needed.  (Same diagonal-SSM kernel family as S4D.)

        With a pole bank, L,R,C,L_c are [d, P]; the kernel is computed for every
        (channel, pole), then mixed down to one kernel per channel by `pole_mix`.

        With a pole bank the kernel is computed per (channel, pole) then mixed
        down to one kernel per channel by `pole_mix`.
        """
        s1, s2, scale = self._poles()                          # [d,P] complex
        dt = float(self.dt)

        n = torch.arange(T, device=s1.device, dtype=torch.float32)
        t = (n * dt).to(torch.complex64).view(T, 1, 1)         # [T,1,1]

        e1  = torch.exp(s1.unsqueeze(0) * t)                   # [T,d,P]
        e2  = torch.exp(s2.unsqueeze(0) * t)                   # [T,d,P]
        den = (s1 - s2).unsqueeze(0)                           # [1,d,P]

        # Critical-damping limit (s1≈s2): (e^{s1 t}−e^{s2 t})/(s1−s2) → t·e^{s1 t}
        eps   = 1e-5
        near  = den.abs() < eps
        h_gen = (e1 - e2) / torch.where(near, den + eps, den)
        h_crit = t * e1
        h = torch.where(near, h_crit, h_gen)                     # [T,d,P] complex

        h = (h * scale.unsqueeze(0)).real                        # apply output scale
        h = (h * self.pole_mix.unsqueeze(0)).sum(dim=-1)         # mix bank → [T, d_inner]
        return h.to(torch.float32)

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
        V    = self.proj_in(x)               # [B, T, d_inner]  content
        gate = F.silu(self.proj_gate(x))     # [B, T, d_inner]  selective gate

        h  = self._impulse_response(T)        # [T, d_inner]  float32

        # ── Causal convolution via FFT (always float32 — no ComplexHalf) ──────
        n     = 2 * T
        V_f32 = V.float()                          # cast input to float32
        H     = torch.fft.rfft(h,     n=n, dim=0) # [n//2+1, d_inner]
        U     = torch.fft.rfft(V_f32, n=n, dim=1) # [B, n//2+1, d_inner]
        Y     = torch.fft.irfft(H.unsqueeze(0) * U,
                                n=n, dim=1)[:, :T] # [B, T, d_inner]  float32
        Y     = Y.to(V.dtype)                      # cast back (fp16 if AMP)

        # ── Selective gating: SSM output modulated by content-dependent gate ──
        y = self.norm(Y) * gate
        return self.proj_out(y)

    # ── Recurrent inference (O(1) per token) ──────────────────────────────────

    def gen_kernel(self) -> dict:
        """
        Precompute the recurrence constants ONCE per generation (not per token).
        Returns λ₁, λ₂ (discrete poles), the safe (s₁−s₂) denominator, and the
        output scale — all [d_inner, n_poles] complex64. Mode-agnostic (uses
        the shared _poles()).
        """
        s1, s2, scale = self._poles()
        dt = float(self.dt)
        den      = s1 - s2
        safe_den = torch.where(den.abs() < 1e-5, den + 1e-5, den)
        return {
            "lam1":  torch.exp(s1 * dt),
            "lam2":  torch.exp(s2 * dt),
            "den":   safe_den,
            "scale": scale,
            "dt":    dt,
        }

    def _emit(self, alpha, beta, gate, kernel):
        """Read output y from SSM state: (dt/L)·Re[(α−β)/(s₁−s₂)] mixed by pole_mix."""
        Y_c    = (alpha - beta) / kernel["den"].unsqueeze(0)
        Y_real = (Y_c * kernel["scale"].unsqueeze(0)).real      # [B, d_inner, n_poles]
        Y      = (Y_real * self.pole_mix.unsqueeze(0)).sum(dim=-1)  # [B, d_inner]
        y      = self.norm(Y.to(gate.dtype)) * gate
        return self.proj_out(y)

    def step(self, x: torch.Tensor, state: Optional[tuple] = None,
             kernel: Optional[dict] = None) -> tuple:
        """
        Single-token recurrent forward — O(1) per token regardless of context.

        Mathematically identical to the FFT forward (diagonal linear-RNN / S4D
        recurrent form):
            alpha[t] = λ₁·alpha[t-1] + V[t]
            beta[t]  = λ₂·beta[t-1]  + V[t]
            y[t]     = (dt/L)·Re[(alpha−beta)/(s₁−s₂)]  mixed by pole_mix

        Pass a `kernel` from gen_kernel() to avoid recomputing the poles each step.

        Args
        ────
        x      : [B, d_model]   single token embedding
        state  : (alpha, beta) each [B, d_inner, n_poles] complex64, or None
        kernel : dict from gen_kernel(), or None to compute on the fly

        Returns
        ───────
        y, (new_alpha, new_beta)
        """
        if kernel is None:
            kernel = self.gen_kernel()

        V    = self.proj_in(x)            # [B, d_inner]
        gate = F.silu(self.proj_gate(x))  # [B, d_inner]
        V_c  = V.to(torch.complex64).unsqueeze(-1)  # [B, d_inner, 1]

        B = x.shape[0]
        if state is None:
            shape = (B, self.d_inner, self.n_poles)
            alpha = torch.zeros(shape, dtype=torch.complex64, device=x.device)
            beta  = torch.zeros(shape, dtype=torch.complex64, device=x.device)
        else:
            alpha, beta = state

        new_alpha = kernel["lam1"].unsqueeze(0) * alpha + V_c
        new_beta  = kernel["lam2"].unsqueeze(0) * beta  + V_c
        return self._emit(new_alpha, new_beta, gate, kernel), (new_alpha, new_beta)

    def final_state(self, x: torch.Tensor, kernel: Optional[dict] = None) -> tuple:
        """
        Extract the SSM state (alpha, beta) at the LAST position of a sequence,
        computed in parallel (no Python loop) so the prompt can be ingested fast.

            alpha[T-1] = Σ_{k=0}^{T-1} λ₁^(T-1-k)·V[k]

        Since |λ| < 1 (stable poles) the powers decay, no overflow.

        Args
        ────
        x : [B, T, d_model]
        Returns (alpha, beta) each [B, d_inner, n_poles] complex64.
        """
        if kernel is None:
            kernel = self.gen_kernel()
        B, T, _ = x.shape
        V   = self.proj_in(x).to(torch.complex64)              # [B, T, d_inner]
        exps = torch.arange(T - 1, -1, -1, device=x.device,
                            dtype=torch.float32).to(torch.complex64)  # [T]
        # λ^exp : [T, d_inner, n_poles]
        pw1 = kernel["lam1"].unsqueeze(0) ** exps.view(T, 1, 1)
        pw2 = kernel["lam2"].unsqueeze(0) ** exps.view(T, 1, 1)
        Vc  = V.unsqueeze(-1)                                   # [B, T, d_inner, 1]
        alpha = (Vc * pw1.unsqueeze(0)).sum(dim=1)             # [B, d_inner, n_poles]
        beta  = (Vc * pw2.unsqueeze(0)).sum(dim=1)
        return alpha, beta

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def wave_stats(self) -> dict:
        # Derive frequency/damping from the actual poles so this works in both
        # modes. s = −γ ± iω_d  ⇒  ω₀ = |s| = √(γ²+ω_d²),  ζ = γ/ω₀.
        s1, _, _ = self._poles()
        gamma = (-s1.real)                                   # [d,P] decay rate
        omega_d = s1.imag.abs()                              # [d,P] damped freq
        omega_0 = (gamma * gamma + omega_d * omega_d).sqrt() # natural freq
        zeta    = gamma / omega_0.clamp(min=1e-6)
        spread  = omega_0.std(dim=-1).mean().item() if self.n_poles > 1 else 0.0
        return {
            "omega_0_mean":    omega_0.mean().item(),
            "omega_0_spread":  spread,
            "damping_mean":    zeta.mean().item(),
            "wave_speed_mean": 0.0,
            "coupling_mean":   0.0,
            "underdamped_%":   (zeta < 1.0).float().mean().item() * 100,
        }

    def extra_repr(self) -> str:
        return (f"d_inner={self.d_inner}, n_poles={self.n_poles}, "
                f"dt={self.dt}, param_mode={self.param_mode}")
