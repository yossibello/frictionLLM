"""
BaselineLM — standard GPT-2 style transformer for fair comparison.

Identical to RLCFrictionLM in every way EXCEPT the FFN:
  RLCFrictionLM : W_gate → RLC circuit → FrictionGate → W_out  (physics)
  BaselineLM    : W_fc1  → GELU        → W_fc2               (standard)

Same parameter count, same attention, same depth.
Train both on identical data/steps → direct architecture comparison.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FrictionConfig
from .attention import CausalSelfAttention


class GELUFFN(nn.Module):
    """Standard transformer FFN: two linear layers with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2  = nn.Linear(d_ff, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class BaselineBlock(nn.Module):
    def __init__(self, config: FrictionConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.ffn  = GELUFFN(config.d_model, config.d_ff, config.dropout, config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BaselineLM(nn.Module):
    """
    Standard GPT-2 style language model — the control experiment.

    Use this to establish whether RLCFrictionLM actually improves on
    the baseline architecture at the same parameter count and compute budget.
    """

    def __init__(self, config: FrictionConfig) -> None:
        super().__init__()
        self.config  = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop    = nn.Dropout(config.dropout)
        self.blocks  = nn.ModuleList(
            [BaselineBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f    = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        scale = (2 * self.config.n_layers) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 * scale)
            nn.init.normal_(block.ffn.fc2.weight,   std=0.02 * scale)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
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

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
