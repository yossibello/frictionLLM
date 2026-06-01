"""
CoulombLM — transformer with Coulomb force attention + RLC friction FFN.

Architecture
────────────
  Standard transformer:
    Attention (dot-product, similarity)  + GELU FFN

  RLCFrictionLM:
    Attention (dot-product, similarity)  + RLC friction FFN   ← prev experiment

  CoulombLM:
    Attention (Coulomb force, complementarity) + RLC friction FFN  ← this

The Coulomb attention replaces dot-product similarity with electric force:
  score(i,j) = k × q_i × q̃_j / (|i−j|+1)²

Opposite-charged tokens attract, same-charged repel.
Distance decays as 1/r² — locality is physically enforced.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FrictionConfig
from .coulomb_attention import CoulombAttention
from .rlc_neuron import RLCFrictionBlock
from .curriculum import SharpnessCurriculum


class CoulombBlock(nn.Module):
    """
    One layer: Coulomb attention + RLC friction FFN.
    """
    def __init__(self, config: FrictionConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = CoulombAttention(
            d_model = config.d_model,
            n_heads = config.n_heads,
            dropout = config.dropout,
            bias    = config.bias,
            r_power = config.coulomb_r_power,
        )
        self.ln2  = nn.LayerNorm(config.d_model)
        self.fglu = RLCFrictionBlock(
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
        rlc_state: Optional[Tuple] = None,
        momentum: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        fglu_out, new_state, new_mom = self.fglu(self.ln2(x), rlc_state, momentum)
        x = x + fglu_out
        return x, new_state, new_mom


class CoulombLM(nn.Module):
    """
    Full CoulombLM language model.

    Comparison:
      BaselineLM    : dot-product attention  + GELU FFN
      RLCFrictionLM : dot-product attention  + RLC friction FFN
      CoulombLM     : Coulomb attention      + RLC friction FFN  ← this
      PhysicsLM     : wave propagation       + RLC friction FFN
    """

    def __init__(self, config: FrictionConfig) -> None:
        super().__init__()
        self.config  = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop    = nn.Dropout(config.dropout)
        self.blocks  = nn.ModuleList(
            [CoulombBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f    = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        scale = (2 * self.config.n_layers) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=0.02 * scale)
            nn.init.normal_(block.fglu.W_out.weight,    std=0.02 * scale)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.config.max_seq_len
        pos    = torch.arange(T, device=idx.device)
        x      = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        rlc_state = None
        momentum  = None
        for block in self.blocks:
            x, rlc_state, momentum = block(x, rlc_state, momentum)
        x      = self.ln_f(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_c  = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx

    @torch.no_grad()
    def charge_report(self, idx: torch.Tensor) -> None:
        """Show per-layer, per-head charge statistics."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x   = self.tok_emb(idx) + self.pos_emb(pos)
        print("\n── Coulomb Charge Report ─────────────────────────────────────")
        for i, block in enumerate(self.blocks):
            stats = block.attn.charge_stats(block.ln1(x))
            k_vals = [f"k={v['coupling_k']:.3f}" for v in stats.values()]
            print(f"  layer_{i}: " + "  ".join(k_vals))
            x, _, _ = block(x)
        print()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
