"""
CoulombAttention — attention driven by electric charge forces.

Physics
───────
Coulomb's Law:  F = k × q₁ × q₂ / r²

  q₁, q₂  : charges  (+ or −, learned from token embeddings)
  r        : distance (positional distance |i−j| + 1)
  k        : coupling constant (learned per head)

  Same sign  (+ × + or − × −) → REPULSION  → low attention weight
  Opp. sign  (+ × −)           → ATTRACTION → high attention weight

Why this is different from dot-product attention
──────────────────────────────────────────────────
Dot-product: score = Q·K  measures SIMILARITY   (like attracts like)
Coulomb:     score = k·q·q̃/r²  measures COMPLEMENTARITY (opposites attract)

In language, meaning often comes from contrast: subject↔verb, question↔answer,
cause↔effect. Coulomb forces naturally model this.  Dot-product attention
cannot represent repulsion — it only adds, never subtracts based on type.

Three forces (multi-head interpretation)
─────────────────────────────────────────
Different heads learn different coupling constants k and charge scales:
  Head with large k, small r-range → strong local force (syntax, local structure)
  Head with small k, large r-range → weak long-range force (topic, global meaning)
  Head with negative charges       → mutual repulsion (diverse information routing)

This naturally gives the multi-scale behaviour of real electromagnetic fields.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FrictionConfig


class CoulombAttention(nn.Module):
    """
    Multi-head Coulomb force attention.

    Each head computes attention scores as:
        score_h(i,j) = k_h × q_h(i) × q̃_h(j) / (|i−j| + 1)²

    where q_h(i) and q̃_h(j) are scalar charges derived from token embeddings.

    Parameters
    ──────────
    d_model  : token embedding dimension
    n_heads  : number of attention heads (each with its own charge pair)
    dropout  : attention dropout
    bias     : use bias in projections
    r_min    : minimum distance (default 1, avoids singularity at r=0)
    r_power  : distance exponent (default 2 = Coulomb; 1 = linear decay)
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        dropout:  float = 0.1,
        bias:     bool  = True,
        r_min:    float = 1.0,
        r_power:  float = 2.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.r_min   = r_min
        self.r_power = r_power

        # ── Charge projections (scalar per head per token) ────────────────────
        # q_charge: "source" charge of each token (what it emits)
        # k_charge: "receiver" charge of each token (how it responds)
        # These are scalars (→ n_heads dims), not vectors like standard Q/K
        self.q_charge = nn.Linear(d_model, n_heads, bias=bias)
        self.k_charge = nn.Linear(d_model, n_heads, bias=bias)

        # ── Value projection (standard — carries the content) ────────────────
        self.v_proj  = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj= nn.Linear(d_model, d_model, bias=bias)

        # ── Per-head coupling constant k (learned, always positive) ──────────
        # log_k so k = exp(log_k) > 0 always
        # Initialised to log(1/√d_head) — similar scale to standard attention
        init_k = math.log(1.0 / math.sqrt(self.d_head))
        self.log_k = nn.Parameter(torch.full((n_heads,), init_k))

        self.drop_a = nn.Dropout(dropout)
        self.drop_r = nn.Dropout(dropout)

    # ── Distance matrix (cached inside forward for current T) ────────────────

    def _distance_matrix(self, T: int, device: torch.device) -> torch.Tensor:
        """
        r[i,j] = |i - j| + r_min   (avoids 1/0 singularity)
        Returns [T, T] distance matrix.
        """
        pos = torch.arange(T, device=device, dtype=torch.float32)
        r   = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() + self.r_min
        return r

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ────
        x : [B, T, d_model]

        Returns
        ───────
        [B, T, d_model]

        Score computation
        ─────────────────
        q_h[b,i] = q_charge(x[b,i])[h]    scalar charge of token i for head h
        k̃_h[b,j] = k_charge(x[b,j])[h]    scalar charge of token j for head h
        score[b,h,i,j] = k_h × q_h[i] × k̃_h[j] / r[i,j]^r_power
        """
        B, T, _ = x.shape

        # ── Scalar charges ────────────────────────────────────────────────────
        q = self.q_charge(x)        # [B, T, n_heads]  source charge
        k = self.k_charge(x)        # [B, T, n_heads]  receiver charge

        q = q.permute(0, 2, 1)      # [B, n_heads, T]
        k = k.permute(0, 2, 1)      # [B, n_heads, T]

        # ── Coulomb interaction matrix ────────────────────────────────────────
        # charge_product[b,h,i,j] = q[b,h,i] × k[b,h,j]
        # + × + or − × − → positive → after 1/r² → still positive → repulsion
        # + × − or − × + → negative → after 1/r² → negative → before softmax → low weight
        # (note: negative pre-softmax score → low but non-zero attention weight)
        charge_product = q.unsqueeze(-1) * k.unsqueeze(-2)   # [B, H, T, T]

        # ── Distance decay ────────────────────────────────────────────────────
        r     = self._distance_matrix(T, x.device)           # [T, T]
        r_pow = r.pow(self.r_power)                           # [T, T]

        # ── Coulomb score ─────────────────────────────────────────────────────
        k_const = self.log_k.exp()                            # [H]
        score   = (k_const.view(1, -1, 1, 1) *
                   charge_product /
                   r_pow.unsqueeze(0).unsqueeze(0))           # [B, H, T, T]

        # ── Causal mask ───────────────────────────────────────────────────────
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        score = score.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        # ── Attention weights ─────────────────────────────────────────────────
        attn = F.softmax(score, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)   # rows that were all -inf → 0
        attn = self.drop_a(attn)                  # [B, H, T, T]

        # ── Value projection and head merge ───────────────────────────────────
        # V is split into heads the standard way
        v    = self.v_proj(x)                                  # [B, T, d_model]
        v_h  = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,d_head]

        out  = (attn @ v_h)                                    # [B, H, T, d_head]
        out  = out.transpose(1, 2).contiguous().view(B, T, -1) # [B, T, d_model]
        return self.drop_r(self.out_proj(out))

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def charge_stats(self, x: torch.Tensor) -> dict:
        """
        Per-head charge statistics for a sample input.
        Shows how polarised each head is (fraction of + vs − charges).
        """
        q  = self.q_charge(x).mean(dim=(0, 1))  # [H]
        k_const = self.log_k.exp()
        return {
            f"head_{h}": {
                "mean_charge": q[h].item(),
                "coupling_k":  k_const[h].item(),
                "polarity":    "+" if q[h] > 0 else "-",
            }
            for h in range(self.n_heads)
        }

    def extra_repr(self) -> str:
        k = self.log_k.exp()
        return (f"n_heads={self.n_heads}, d_head={self.d_head}, "
                f"r_power={self.r_power}, k_mean={k.mean():.3f}")
