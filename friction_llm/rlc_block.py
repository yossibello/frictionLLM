"""
RLCTransformerBlock — one layer of the RLC circuit language model.

Threads three state tensors through the block stack:
    rlc_state : (q, i)   — charge and current  [B, T, d_ff]
    momentum  :          — friction momentum    [B, T, d_ff]

Both carry over from layer L to layer L+1, creating depth-as-time dynamics.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import FrictionConfig
from .attention import CausalSelfAttention
from .rlc_neuron import RLCFrictionBlock


class RLCTransformerBlock(nn.Module):
    def __init__(self, config: FrictionConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.rlc_block = RLCFrictionBlock(
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
            L_init=config.rlc_L_init,
            R_init=config.rlc_R_init,
            C_init=config.rlc_C_init,
            dt=config.rlc_dt,
            clamp=config.rlc_clamp,
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
        rlc_state : (q, i) from previous block, or None
        momentum  : friction momentum from previous block, or None

        Returns
        ───────
        x         : [B, T, d_model]  updated hidden state
        new_state : (q_new, i_new)   circuit state for next block
        new_mom   : [B, T, d_ff]     friction momentum for next block
        """
        x = x + self.attn(self.ln1(x))
        rlc_out, new_state, new_mom = self.rlc_block(self.ln2(x), rlc_state, momentum)
        x = x + rlc_out
        return x, new_state, new_mom
