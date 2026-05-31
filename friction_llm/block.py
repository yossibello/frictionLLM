"""
FrictionTransformerBlock — one layer of the friction LM.

Wiring
──────
    x  →  LayerNorm  →  CausalSelfAttention  →  residual
       →  LayerNorm  →  FGLUBlock(momentum_in)  →  residual
    momentum_in  (from previous block)  →  momentum_out  (to next block)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import FrictionConfig
from .attention import CausalSelfAttention
from .friction_gate import FGLUBlock


class FrictionTransformerBlock(nn.Module):
    def __init__(self, config: FrictionConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.fglu = FGLUBlock(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            mu_s_init=config.mu_s_init,
            mu_k_ratio_init=config.mu_k_ratio_init,
            sharpness=config.sharpness_init,
            momentum_alpha=config.momentum_alpha,
            momentum_beta=config.momentum_beta,
            use_momentum=config.use_momentum,
            bias=config.bias,
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
        momentum : [B, T, d_ff] or None   — from the previous block

        Returns
        ───────
        x            : [B, T, d_model]   updated hidden state
        new_momentum : [B, T, d_ff]      momentum for the next block
        """
        x = x + self.attn(self.ln1(x))
        fglu_out, new_momentum = self.fglu(self.ln2(x), momentum)
        x = x + fglu_out
        return x, new_momentum
