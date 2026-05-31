"""
Core friction mechanism: FrictionGate and FGLUBlock.

Physics model
─────────────
  |Z_i| ≤ μ_s  →  Y_i = 0            (stuck — static friction holds)
  |Z_i| >  μ_s  →  Y_i = Z_i − sign(Z_i)·μ_k  (moving — kinetic drag)

The "slip jolt" (μ_s > μ_k) creates a discontinuity — the neuron jumps
the moment it breaks free, then costs only μ_k to stay moving.

Training: smooth sigmoid surrogate (fully differentiable, sharpness annealed)
Inference: hard threshold (true physics, maximum sparsity → CPU wins big)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrictionGate(nn.Module):
    """
    Learnable per-neuron static + kinetic friction gate.

    Parameters
    ──────────
    size             : number of neurons (d_ff)
    mu_s_init        : initial static threshold
    mu_k_ratio_init  : initial ratio mu_k / mu_s  ∈ (0, 1)  → ensures μ_k < μ_s
    sharpness        : surrogate gradient steepness (set externally by curriculum)
    momentum_alpha   : EMA decay for cross-layer momentum
    momentum_beta    : how much momentum lowers effective μ_s  (0 = none, 1 = full)
    use_momentum     : enable cross-layer momentum
    """

    def __init__(
        self,
        size: int,
        mu_s_init: float = 0.5,
        mu_k_ratio_init: float = 0.6,
        sharpness: float = 3.0,
        momentum_alpha: float = 0.9,
        momentum_beta: float = 0.3,
        use_momentum: bool = True,
    ) -> None:
        super().__init__()
        self.size = size
        self.sharpness = sharpness          # mutable — updated by SharpnessCurriculum
        self.use_momentum = use_momentum
        self.momentum_alpha = momentum_alpha
        self.momentum_beta = momentum_beta

        # ── μ_s = softplus(raw_μ_s)  →  always positive ──────────────────────
        raw_s = math.log(math.expm1(mu_s_init))  # inv-softplus
        self.raw_mu_s = nn.Parameter(torch.full((size,), raw_s))

        # ── μ_k = μ_s · sigmoid(raw_ratio)  →  μ_k ∈ (0, μ_s) always ────────
        raw_r = math.log(mu_k_ratio_init / (1.0 - mu_k_ratio_init))  # logit
        self.raw_ratio = nn.Parameter(torch.full((size,), raw_r))

    # ── Derived friction coefficients ────────────────────────────────────────

    @property
    def mu_s(self) -> torch.Tensor:
        return F.softplus(self.raw_mu_s)

    @property
    def mu_k(self) -> torch.Tensor:
        return self.mu_s * torch.sigmoid(self.raw_ratio)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        z: torch.Tensor,
        momentum: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ────
        z        : [..., size]   pre-activation signal ("applied force")
        momentum : [..., size]   cross-layer momentum from previous block, or None

        Returns
        ───────
        output       : [..., size]   friction-gated signal
        new_momentum : [..., size]   updated momentum (detached, no grad)
        """
        mu_s = self.mu_s   # [size]
        mu_k = self.mu_k   # [size]

        # Neurons "already in motion" have a lower effective threshold
        if self.use_momentum and momentum is not None:
            reduction = (self.momentum_beta * momentum).clamp(max=0.95)
            mu_s = (mu_s * (1.0 - reduction)).clamp(min=1e-6)

        output = self._smooth_forward(z, mu_s, mu_k) if self.training \
            else self._hard_forward(z, mu_s, mu_k)

        # Momentum update — no gradient flows through here
        with torch.no_grad():
            fired = (z.abs() > mu_s).float()
            if momentum is not None and self.use_momentum:
                new_momentum = self.momentum_alpha * momentum + (1.0 - self.momentum_alpha) * fired
            else:
                new_momentum = fired

        return output, new_momentum

    # ── Internal modes ───────────────────────────────────────────────────────

    def _smooth_forward(
        self, z: torch.Tensor, mu_s: torch.Tensor, mu_k: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable surrogate: sigmoid ramp stands in for the step function."""
        gate = torch.sigmoid(self.sharpness * (z.abs() - mu_s))
        kinetic = z - z.sign() * mu_k
        return gate * kinetic

    def _hard_forward(
        self, z: torch.Tensor, mu_s: torch.Tensor, mu_k: torch.Tensor
    ) -> torch.Tensor:
        """True friction at inference — every zero is a real zero (skippable on CPU)."""
        mask = z.abs() > mu_s
        return torch.where(mask, z - z.sign() * mu_k, torch.zeros_like(z))

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def measure_sparsity(self, z: torch.Tensor) -> float:
        """Fraction of neurons zeroed. 0 = dense, 1 = fully sparse."""
        return (z.abs() <= self.mu_s).float().mean().item()

    def extra_repr(self) -> str:
        s = self.mu_s.mean().item()
        k = self.mu_k.mean().item()
        return f"size={self.size}, μ_s≈{s:.3f}, μ_k≈{k:.3f}, sharpness={self.sharpness}"


# ─────────────────────────────────────────────────────────────────────────────

class FGLUBlock(nn.Module):
    """
    Friction-Gated Linear Unit — drop-in replacement for the transformer FFN.

    Architecture (GLU-style)
    ────────────────────────
        gate_signal = W_gate(x)             [d_model → d_ff]  "applied force"
        value       = W_up(x)              [d_model → d_ff]  content
        gated       = FrictionGate(gate_signal)   sparse activation
        output      = W_out(gated ⊙ value)  [d_ff → d_model]

    Why GLU-style?
    ──────────────
    The gate controls *which* information passes; the value carries *what* it says.
    Sparsity in `gated` propagates into W_out — on CPU at inference, those rows
    of W_out can be skipped entirely (see inference.py sparse path).

    Weight count: 3 × d_model × d_ff  (vs 2 × for plain FFN).
    Set d_ff ≈ 2.67 × d_model to match parameter count with standard FFN.
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
    ) -> None:
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.W_up   = nn.Linear(d_model, d_ff, bias=bias)
        self.W_out  = nn.Linear(d_ff,   d_model, bias=bias)
        self.drop   = nn.Dropout(dropout)

        self.friction = FrictionGate(
            d_ff,
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
        momentum: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ────
        x        : [B, T, d_model]
        momentum : [B, T, d_ff] or None

        Returns
        ───────
        output       : [B, T, d_model]
        new_momentum : [B, T, d_ff]
        """
        gate_signal = self.W_gate(x)                          # applied force
        value       = self.W_up(x)                            # content
        gated, new_momentum = self.friction(gate_signal, momentum)
        out = self.drop(self.W_out(gated * value))
        return out, new_momentum

    @torch.no_grad()
    def sparsity(self, x: torch.Tensor) -> float:
        """Gate sparsity for diagnostics (call in eval mode)."""
        return self.friction.measure_sparsity(self.W_gate(x))
