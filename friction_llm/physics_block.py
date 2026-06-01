"""
PhysicsBlock — fully physics-native transformer layer.

Replaces BOTH components of a standard transformer block:
  Attention   → CoupledOscillatorMixer  (wave propagation along sequence)
  FFN (GELU)  → RLCFrictionBlock        (circuit filter per position)

Standard transformer block:
    x → LayerNorm → Attention   → residual
    x → LayerNorm → FFN(GELU)   → residual

PhysicsBlock:
    x → LayerNorm → CoupledOscillatorMixer → residual   ← wave propagation
    x → LayerNorm → RLCFrictionBlock       → residual   ← circuit filter
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import FrictionConfig
from .coupled_mixer import CoupledOscillatorMixer
from .rlc_neuron import RLCFrictionBlock


class PhysicsBlock(nn.Module):
    """
    One layer of the fully physics-native language model.

    mixer  : CoupledOscillatorMixer — wave propagation along token dimension
    filter : RLCFrictionBlock       — per-position circuit filter with friction gate

    State threading
    ───────────────
    The RLC friction state (q, i) and momentum carry over from block L to L+1,
    just like in RLCFrictionLM.  The mixer state (q, i_scan) does NOT carry
    across blocks — it restarts at zero for each block's scan (different scan
    over the same sequence from a fresh wave state).
    """

    def __init__(self, config: FrictionConfig) -> None:
        super().__init__()
        self.ln_mix  = nn.LayerNorm(config.d_model)
        self.mixer   = CoupledOscillatorMixer(
            d_model  = config.d_model,
            d_inner  = config.d_model,      # same dim as model
            n_poles  = config.mixer_n_poles,
            dt       = config.rlc_dt,
            L_c_init = config.mixer_L_c_init,
            clamp    = config.rlc_clamp,
            bias     = config.bias,
        )
        self.ln_filt = nn.LayerNorm(config.d_model)
        self.filter  = RLCFrictionBlock(
            d_model         = config.d_model,
            d_ff            = config.d_ff,
            dropout         = config.dropout,
            mu_s_init       = config.mu_s_init,
            mu_k_ratio_init = config.mu_k_ratio_init,
            sharpness       = config.sharpness_init,
            momentum_alpha  = config.momentum_alpha,
            momentum_beta   = config.momentum_beta,
            use_momentum    = config.use_momentum,
            bias            = config.bias,
            L_init          = config.rlc_L_init,
            R_init          = config.rlc_R_init,
            C_init          = config.rlc_C_init,
            dt              = config.rlc_dt,
            clamp           = config.rlc_clamp,
            filter_mode     = config.rlc_filter_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        rlc_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        momentum:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor,
               Tuple[torch.Tensor, torch.Tensor],
               torch.Tensor]:
        """
        Args
        ────
        x         : [B, T, d_model]
        rlc_state : (q, i) from previous block's RLC filter, or None
        momentum  : friction momentum from previous block, or None

        Returns
        ───────
        x            : [B, T, d_model]
        new_rlc_state: (q, i) for next block
        new_momentum : for next block
        """
        # Wave propagation (replaces attention)
        x = x + self.mixer(self.ln_mix(x))

        # Circuit filter (replaces FFN)
        filt_out, new_rlc_state, new_momentum = self.filter(
            self.ln_filt(x), rlc_state, momentum
        )
        x = x + filt_out

        return x, new_rlc_state, new_momentum
