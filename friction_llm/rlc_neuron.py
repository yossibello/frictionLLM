"""
RLC Circuit Neuron — full electromechanical physics mapped to neural computation.

Physics equations (series RLC / damped oscillator)
────────────────────────────────────────────────────
    L·q̈  +  R·q̇  +  q/C  =  V(t)

    L  inductance  ↔  mass       — inertia, resists change in current
    R  resistance  ↔  damping    — dissipates energy (our friction)
    C  capacitance ↔  1/spring   — stores charge, creates restoring force
    V(t)           ↔  applied force  — input signal from layer below
    q              ↔  position   — charge (neuron's accumulated state)
    i = dq/dt      ↔  velocity   — current (rate of change)

Rewritten as first-order system (state: q, i):
    dq/dt  =  i
    di/dt  =  ( V  −  R·i  −  q/C )  /  L

Integration: symplectic (semi-implicit) Euler — stable for oscillatory systems:
    i[t+1]  =  i[t]  +  di · dt        (update current first)
    q[t+1]  =  q[t]  +  i[t+1] · dt   (update charge with NEW current)

Three dynamical regimes — emerge naturally from learned L, R, C:
    ζ > 1  overdamped   : slow stable crawl, no oscillation
    ζ = 1  critical     : fastest response without overshoot  (init target)
    ζ < 1  underdamped  : rings at ω₀ = 1/√(LC), amplifies resonant inputs

Stacking across layers (depth = time)
──────────────────────────────────────
The circuit state (q, i) from layer L is passed as the initial state to layer
L+1.  Charge accumulated deep in the circuit seeds the next stage — like a
ladder network of RLC filters, each tuned to a different frequency band.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .friction_gate import FrictionGate


# ─────────────────────────────────────────────────────────────────────────────

class RLCNeuron(nn.Module):
    """
    Per-neuron RLC circuit with learnable L, R, C.

    All three parameters are stored as log values → always positive.
    Initialized to critical damping (ζ = 1) by default, meaning the circuit
    reaches equilibrium as fast as possible without oscillating.  The network
    is free to learn underdamped (resonant) or overdamped neurons.

    State: (q, i) — charge and current — shape [..., size].
    Passed from one transformer block to the next (depth stacking).
    """

    def __init__(
        self,
        size: int,
        L_init: float = 1.0,
        R_init: float = 2.0,   # 2·√(L/C) = 2.0 → ζ = 1 when L=C=1
        C_init: float = 1.0,
        dt: float = 0.1,
        clamp: float = 10.0,
    ) -> None:
        super().__init__()
        self.size  = size
        self.dt    = dt
        self.clamp = clamp

        # log parameterisation keeps L, R, C strictly positive
        self.log_L = nn.Parameter(torch.full((size,), math.log(L_init)))
        self.log_R = nn.Parameter(torch.full((size,), math.log(R_init)))
        self.log_C = nn.Parameter(torch.full((size,), math.log(C_init)))

    # ── Circuit parameters ────────────────────────────────────────────────────

    @property
    def L(self) -> torch.Tensor:
        return self.log_L.exp()

    @property
    def R(self) -> torch.Tensor:
        return self.log_R.exp()

    @property
    def C(self) -> torch.Tensor:
        return self.log_C.exp()

    @property
    def omega_0(self) -> torch.Tensor:
        """Natural resonant frequency  ω₀ = 1 / √(LC)"""
        return 1.0 / (self.L * self.C).sqrt()

    @property
    def damping_ratio(self) -> torch.Tensor:
        """
        ζ = R / (2·√(L/C))
          < 1 : underdamped  (resonant — rings at ω₀)
          = 1 : critically damped (fastest stable response)
          > 1 : overdamped (slow, no oscillation)
        """
        return self.R / (2.0 * (self.L / self.C).sqrt())

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        V: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args
        ────
        V     : [..., size]  applied voltage (input force from W_gate projection)
        state : (q, i) or None — circuit state from previous layer

        Returns
        ───────
        q_new     : [..., size]  new charge (position)
        new_state : (q_new, i_new)
        """
        if state is None:
            q = torch.zeros_like(V)
            i = torch.zeros_like(V)
        else:
            q, i = state

        # di/dt = ( V − R·i − q/C ) / L
        di = (V - self.R * i - q / self.C) / self.L

        # Symplectic Euler: update i first, then q with new i
        i_new = i + di * self.dt
        q_new = q + i_new * self.dt

        # Clamp to prevent numerical blowup in early training
        i_new = i_new.clamp(-self.clamp, self.clamp)
        q_new = q_new.clamp(-self.clamp, self.clamp)

        return q_new, (q_new, i_new)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def circuit_stats(self) -> dict:
        return {
            "omega_0_mean":  self.omega_0.mean().item(),
            "omega_0_std":   self.omega_0.std().item(),
            "zeta_mean":     self.damping_ratio.mean().item(),
            "zeta_std":      self.damping_ratio.std().item(),
            "underdamped_%": (self.damping_ratio < 1.0).float().mean().item() * 100,
            "overdamped_%":  (self.damping_ratio > 1.0).float().mean().item() * 100,
        }

    def extra_repr(self) -> str:
        ω = self.omega_0.mean().item()
        ζ = self.damping_ratio.mean().item()
        return f"size={self.size}, ω₀≈{ω:.3f}, ζ≈{ζ:.3f}, dt={self.dt}"


# ─────────────────────────────────────────────────────────────────────────────

class RLCFrictionBlock(nn.Module):
    """
    Full circuit neuron: RLC dynamics + friction gate + GLU projection.

    Signal path
    ───────────
        gate_signal = W_gate(x)          [d_model → d_ff]  "applied voltage"
        value       = W_up(x)            [d_model → d_ff]  content carrier

        q, (q,i) = RLC(gate_signal, prev_state)     circuit dynamics
        gated, momentum = FrictionGate(q, momentum) static/kinetic filter
                                                     on accumulated charge

        output = W_out( gated ⊙ value )  [d_ff → d_model]

    Why friction on charge (q), not on raw voltage (V)?
    ────────────────────────────────────────────────────
    A capacitor charges up over layers.  The friction gate fires only when
    accumulated charge exceeds μ_s — like a defibrillator capacitor that
    builds until it breaks through the threshold, then discharges with kinetic
    drag μ_k.  Single-shot voltage (V) hitting the gate would degrade to a
    plain threshold activation; charge (q) carries the circuit history.

    Stacking
    ────────
    Block returns (q_new, i_new) as new_state.  The next block receives this
    as its initial state — charge and current carry over, creating a ladder
    network of coupled RLC filters across depth.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        mu_s_init: float = 0.5,
        mu_k_ratio_init: float = 0.6,
        sharpness: float = 3.0,
        momentum_alpha: float = 0.9,
        momentum_beta: float = 0.3,
        use_momentum: bool = True,
        bias: bool = True,
        L_init: float = 1.0,
        R_init: float = 2.0,
        C_init: float = 1.0,
        dt: float = 0.1,
        clamp: float = 10.0,
    ) -> None:
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.W_up   = nn.Linear(d_model, d_ff, bias=bias)
        self.W_out  = nn.Linear(d_ff,   d_model, bias=bias)
        self.drop   = nn.Dropout(dropout)

        self.rlc = RLCNeuron(
            size=d_ff,
            L_init=L_init, R_init=R_init, C_init=C_init,
            dt=dt, clamp=clamp,
        )
        self.friction = FrictionGate(
            size=d_ff,
            mu_s_init=mu_s_init,
            mu_k_ratio_init=mu_k_ratio_init,
            sharpness=sharpness,
            momentum_alpha=momentum_alpha,
            momentum_beta=momentum_beta,
            use_momentum=use_momentum,
        )

    def forward(
        self,
        x: torch.Tensor,
        rlc_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        momentum: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor,
               Tuple[torch.Tensor, torch.Tensor],
               torch.Tensor]:
        """
        Args
        ────
        x         : [B, T, d_model]
        rlc_state : (q, i) each [B, T, d_ff], or None
        momentum  : [B, T, d_ff] or None

        Returns
        ───────
        output    : [B, T, d_model]
        new_state : (q_new, i_new)  — passed to next block
        new_mom   : [B, T, d_ff]   — passed to next block
        """
        V = self.W_gate(x)                                    # applied voltage
        q, new_state = self.rlc(V, rlc_state)                # circuit dynamics
        gated, new_mom = self.friction(q, momentum)           # friction on charge
        out = self.drop(self.W_out(gated * self.W_up(x)))     # GLU output
        return out, new_state, new_mom

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor) -> dict:
        """Per-block physics stats for monitoring during training."""
        V = self.W_gate(x)
        q, _ = self.rlc(V)
        sparsity = self.friction.measure_sparsity(q)
        stats = self.rlc.circuit_stats()
        stats["sparsity"] = sparsity
        stats["mu_s_mean"] = self.friction.mu_s.mean().item()
        stats["mu_k_mean"] = self.friction.mu_k.mean().item()
        return stats
