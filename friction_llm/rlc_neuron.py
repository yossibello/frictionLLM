"""
RLC Circuit Neuron — full electromechanical physics mapped to neural computation.

Physics equations (series RLC / damped oscillator)
────────────────────────────────────────────────────
    L·q̈  +  R·q̇  +  q/C  =  V(t)

    L  inductance  ↔  mass       — inertia, resists change in current
    R  resistance  ↔  damping    — dissipates energy (our friction)
    C  capacitance ↔  1/spring   — stores charge, creates restoring force
    V(t)           ↔  applied force  — input signal
    q              ↔  position   — accumulated charge
    i = dq/dt      ↔  velocity   — current

Filter types from one RLC circuit
───────────────────────────────────
V splits across three components: V = V_L + V_R + V_C

    V_C = q          → LOW-PASS   voltage across capacitor (what we output by default)
    V_R = R · i      → BAND-PASS  voltage across resistor
    V_L = V−V_R−V_C  → HIGH-PASS  voltage across inductor
    V_C + V_L        → NOTCH      everything except the resonant band

filter_mode controls which output the neuron uses:
    "lowpass"   — original behaviour, backward-compatible
    "highpass"  — only fast-changing signal passes
    "bandpass"  — only signal near ω₀ passes
    "notch"     — ω₀ band is suppressed
    "learnable" — per-neuron learned mix of V_C, V_R, V_L
                  starts as ~95% low-pass; network can evolve any filter shape
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .friction_gate import FrictionGate


FILTER_MODES = ("lowpass", "highpass", "bandpass", "notch", "learnable")


# ─────────────────────────────────────────────────────────────────────────────

class RLCNeuron(nn.Module):
    """
    Per-neuron RLC circuit with learnable L, R, C and configurable filter type.

    filter_mode
    ───────────
    "lowpass"   → output q  (voltage across C — original behaviour)
    "highpass"  → output V_L = V − V_R − V_C
    "bandpass"  → output V_R = R · i
    "notch"     → output V_C + V_L
    "learnable" → output w_C·V_C + w_R·V_R + w_L·V_L
                  where w are learned per-neuron softmax weights
                  init: 95% low-pass → network evolves to whatever it needs
    """

    def __init__(
        self,
        size: int,
        L_init: float = 1.0,
        R_init: float = 2.0,
        C_init: float = 1.0,
        dt: float = 0.1,
        clamp: float = 10.0,
        filter_mode: str = "lowpass",
    ) -> None:
        super().__init__()
        assert filter_mode in FILTER_MODES, \
            f"filter_mode must be one of {FILTER_MODES}, got {filter_mode!r}"

        self.size        = size
        self.dt          = dt
        self.clamp       = clamp
        self.filter_mode = filter_mode

        self.log_L = nn.Parameter(torch.full((size,), math.log(L_init)))
        self.log_R = nn.Parameter(torch.full((size,), math.log(R_init)))
        self.log_C = nn.Parameter(torch.full((size,), math.log(C_init)))

        if filter_mode == "learnable":
            # mix[0] → V_C (low-pass weight)
            # mix[1] → V_R (band-pass weight)
            # mix[2] → V_L (high-pass weight)
            # Init: softmax([3,0,0]) ≈ [0.95, 0.025, 0.025] → nearly pure low-pass
            # Network is free to move weights during training
            self.mix = nn.Parameter(torch.zeros(3, size))
            nn.init.constant_(self.mix[0], 3.0)

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
        """ζ = R / (2·√(L/C))   <1=underdamped, 1=critical, >1=overdamped"""
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
        V     : [..., size]   applied voltage
        state : (q, i) or None

        Returns
        ───────
        output    : [..., size]   filtered signal (type depends on filter_mode)
        new_state : (q_new, i_new)
        """
        if state is None:
            q = torch.zeros_like(V)
            i = torch.zeros_like(V)
        else:
            q, i = state

        # Symplectic Euler integration
        di    = (V - self.R * i - q / self.C) / self.L
        i_new = (i + di * self.dt).clamp(-self.clamp, self.clamp)
        q_new = (q + i_new * self.dt).clamp(-self.clamp, self.clamp)

        output = self._filter_output(V, q_new, i_new)
        return output, (q_new, i_new)

    def _filter_output(
        self, V: torch.Tensor, q: torch.Tensor, i: torch.Tensor
    ) -> torch.Tensor:
        """Select / mix filter outputs based on filter_mode."""
        if self.filter_mode == "lowpass":
            return q                        # voltage across C (default)

        v_C = q                             # low-pass
        v_R = self.R * i                    # band-pass
        v_L = (V - v_R - q / self.C)       # high-pass

        if self.filter_mode == "highpass":
            return v_L.clamp(-self.clamp, self.clamp)
        if self.filter_mode == "bandpass":
            return v_R.clamp(-self.clamp, self.clamp)
        if self.filter_mode == "notch":
            return (v_C + v_L).clamp(-self.clamp, self.clamp)

        # "learnable"
        w = torch.softmax(self.mix, dim=0)          # [3, size]
        out = w[0] * v_C + w[1] * v_R + w[2] * v_L
        return out.clamp(-self.clamp, self.clamp)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def circuit_stats(self) -> dict:
        stats = {
            "omega_0_mean":  self.omega_0.mean().item(),
            "omega_0_std":   self.omega_0.std().item(),
            "zeta_mean":     self.damping_ratio.mean().item(),
            "zeta_std":      self.damping_ratio.std().item(),
            "underdamped_%": (self.damping_ratio < 1.0).float().mean().item() * 100,
            "overdamped_%":  (self.damping_ratio > 1.0).float().mean().item() * 100,
            "filter_mode":   self.filter_mode,
        }
        if self.filter_mode == "learnable":
            w = torch.softmax(self.mix, dim=0).mean(dim=1)  # [3]
            stats["mix_lowpass_%"]  = w[0].item() * 100
            stats["mix_bandpass_%"] = w[1].item() * 100
            stats["mix_highpass_%"] = w[2].item() * 100
        return stats

    @torch.no_grad()
    def filter_weights(self) -> Dict[str, float]:
        """
        Human-readable filter composition.
        For fixed modes returns 100% of that type.
        For 'learnable' returns per-neuron average weights.
        """
        if self.filter_mode != "learnable":
            return {self.filter_mode: 100.0}
        w = torch.softmax(self.mix, dim=0).mean(dim=1)
        return {
            "lowpass":  round(w[0].item() * 100, 1),
            "bandpass": round(w[1].item() * 100, 1),
            "highpass": round(w[2].item() * 100, 1),
        }

    def extra_repr(self) -> str:
        ω = self.omega_0.mean().item()
        ζ = self.damping_ratio.mean().item()
        return f"size={self.size}, ω₀≈{ω:.3f}, ζ≈{ζ:.3f}, dt={self.dt}, filter={self.filter_mode}"


# ─────────────────────────────────────────────────────────────────────────────

class RLCFrictionBlock(nn.Module):
    """
    Full circuit neuron: RLC dynamics + friction gate + GLU projection.

    filter_mode is passed through to RLCNeuron — see RLCNeuron docstring.
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
        filter_mode: str = "lowpass",
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
            filter_mode=filter_mode,
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
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        V              = self.W_gate(x)
        filtered, new_state = self.rlc(V, rlc_state)
        gated, new_mom      = self.friction(filtered, momentum)
        out                 = self.drop(self.W_out(gated * self.W_up(x)))
        return out, new_state, new_mom

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor) -> dict:
        V = self.W_gate(x)
        filtered, _ = self.rlc(V)
        sparsity    = self.friction.measure_sparsity(filtered)
        stats       = self.rlc.circuit_stats()
        stats["sparsity"]  = sparsity
        stats["mu_s_mean"] = self.friction.mu_s.mean().item()
        stats["mu_k_mean"] = self.friction.mu_k.mean().item()
        return stats
