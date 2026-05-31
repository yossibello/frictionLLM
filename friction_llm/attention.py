"""
Causal self-attention.

Phase 1: standard multi-head attention (flash attention if torch >= 2.0).
Phase 2: friction-sparse attention (toggled via config.use_friction_attention).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FrictionConfig
from .friction_gate import FrictionGate


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Uses torch.nn.functional.scaled_dot_product_attention (flash attention)
    when available (PyTorch >= 2.0), otherwise falls back to manual O(T²).

    Phase 2 friction-sparse attention
    ──────────────────────────────────
    When config.use_friction_attention=True, a FrictionGate is applied to the
    raw attention logits before softmax. Logits below μ_s_attn are hard-zeroed;
    the surviving logits lose μ_k_attn of energy. This creates sparse attention
    patterns where low-relevance tokens are completely ignored.
    """

    def __init__(self, config: FrictionConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, \
            "d_model must be divisible by n_heads"

        self.n_heads = config.n_heads
        self.d_head  = config.d_model // config.n_heads
        self.d_model = config.d_model

        self.qkv    = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj   = nn.Linear(config.d_model, config.d_model,     bias=config.bias)
        self.drop_a = nn.Dropout(config.dropout)
        self.drop_r = nn.Dropout(config.dropout)

        # Phase 2: optional friction-sparse attention gate
        self.use_friction_attn = config.use_friction_attention
        if self.use_friction_attn:
            # One gate per head, gate operates on attention logits
            self.attn_friction = FrictionGate(
                size=config.n_heads,          # one threshold per head
                mu_s_init=config.attn_mu_s_init,
                mu_k_ratio_init=config.attn_mu_k_ratio_init,
                sharpness=config.sharpness_init,
                use_momentum=False,           # no cross-layer state in attention
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ────
        x : [B, T, d_model]

        Returns
        ───────
        [B, T, d_model]
        """
        B, T, C = x.shape

        # Project to Q, K, V
        q, k, v = self.qkv(x).split(C, dim=-1)
        # Reshape to [B, heads, T, d_head]
        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        if self.use_friction_attn:
            out = self._friction_attention(q, k, v, B, T)
        elif hasattr(F, "scaled_dot_product_attention"):
            # Flash attention — O(T) memory, fastest path
            drop_p = self.drop_a.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p, is_causal=True)
        else:
            out = self._manual_attention(q, k, v, T)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop_r(self.proj(out))

    def _manual_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, T: int
    ) -> torch.Tensor:
        scale = self.d_head ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop_a(att)
        return att @ v

    def _friction_attention(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        B: int, T: int,
    ) -> torch.Tensor:
        """
        Phase 2: friction-gated sparse attention.

        Logits that don't have enough "force" to overcome μ_s_attn are zeroed —
        the model physically refuses to attend to weak relationships.
        """
        scale = self.d_head ** -0.5
        logits = (q @ k.transpose(-2, -1)) * scale    # [B, heads, T, T]

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1
        )
        logits = logits.masked_fill(causal_mask, float("-inf"))

        # Friction gate on logits (per head, averaged across positions)
        # We reshape so friction operates per-head across the T×T logit space
        B_, H, T1, T2 = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, H)   # [B·T·T, H]
        gated_flat, _ = self.attn_friction(logits_flat)            # sparse logits
        logits = gated_flat.reshape(B_, T1, T2, H).permute(0, 3, 1, 2)

        # Replace fully-zeroed rows with -inf so softmax doesn't give uniform
        all_zero_rows = (logits == 0).all(dim=-1, keepdim=True)
        logits = logits.masked_fill(all_zero_rows, float("-inf"))

        att = F.softmax(logits, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)   # heads that attended nowhere → 0
        att = self.drop_a(att)
        return att @ v
