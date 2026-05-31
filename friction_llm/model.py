"""
FrictionLM — full autoregressive language model.

Stack: token emb + positional emb → N × FrictionTransformerBlock → LayerNorm → LM head.
Weight tying: lm_head shares weights with token embedding (standard practice).
Momentum tensor is threaded through all blocks during a single forward pass.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FrictionConfig
from .block import FrictionTransformerBlock
from .friction_gate import FrictionGate


class FrictionLM(nn.Module):
    def __init__(self, config: FrictionConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop    = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [FrictionTransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f    = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying — halves embedding parameters, stabilises training
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-2-style initialisation: small normal, residual paths scaled by depth."""
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        # Scale residual projections by 1/√(2·n_layers) — GPT-2 paper
        scale = (2 * self.config.n_layers) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 * scale)
            nn.init.normal_(block.fglu.W_out.weight, std=0.02 * scale)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args
        ────
        idx     : [B, T]  token indices
        targets : [B, T]  next-token targets (None at inference)

        Returns
        ───────
        logits : [B, T, vocab_size]
        loss   : scalar cross-entropy loss, or None
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        device = idx.device
        pos = torch.arange(T, device=device, dtype=torch.long)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        # Thread momentum through all blocks (None → first block initialises it)
        momentum: Optional[torch.Tensor] = None
        for block in self.blocks:
            x, momentum = block(x, momentum)

        x = self.ln_f(x)
        logits = self.lm_head(x)   # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ── Inference helpers ────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args
        ────
        idx            : [B, T]  seed token indices
        max_new_tokens : number of tokens to generate
        temperature    : sampling temperature (1.0 = unscaled)
        top_k          : if set, restrict sampling to top-k logits

        Returns
        ───────
        [B, T + max_new_tokens]
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def measure_sparsity(self, idx: torch.Tensor) -> dict:
        """
        Run a forward pass and return per-layer gate sparsity.
        Call in eval mode for hard-threshold (true) sparsity numbers.
        """
        self.eval()
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        stats = {}
        momentum = None
        for i, block in enumerate(self.blocks):
            gate_signal = block.fglu.W_gate(block.ln2(x))
            sparsity = block.fglu.friction.measure_sparsity(gate_signal)
            mu_s_mean = block.fglu.friction.mu_s.mean().item()
            mu_k_mean = block.fglu.friction.mu_k.mean().item()
            stats[f"layer_{i}"] = {
                "sparsity": sparsity,
                "mu_s": mu_s_mean,
                "mu_k": mu_k_mean,
            }
            x, momentum = block(x, momentum)

        overall = sum(v["sparsity"] for v in stats.values()) / len(stats)
        stats["overall"] = overall
        return stats

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"params={self.param_count()/1e6:.1f}M, "
            f"layers={self.config.n_layers}, "
            f"d_model={self.config.d_model}"
        )
