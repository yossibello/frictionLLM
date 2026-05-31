"""
RLCFrictionLM — language model where every FFN is a full RLC circuit neuron.

What's new vs FrictionLM
────────────────────────
  FrictionLM   : FFN replaced by FGLUBlock  (friction gate — R only)
  RLCFrictionLM: FFN replaced by RLCFrictionBlock (L + R + C + friction gate)

Each neuron now carries:
  L  inductance  — inertia, resists rapid signal changes
  R  resistance  — our friction gate (static/kinetic threshold)
  C  capacitance — accumulates charge across layers

The circuit state (q=charge, i=current) propagates through all N blocks.
This turns depth into a physical time axis: charge builds through layers
like a wave moving through a ladder network of coupled RLC filters.

What emerges from training
──────────────────────────
  Different layers learn different natural frequencies ω₀ = 1/√(LC):
    Early layers → high ω₀ → high-pass → local / syntactic patterns
    Late layers  → low ω₀  → low-pass  → global / semantic patterns
  This is hierarchical feature extraction derived from physics — not just
  an empirical observation but a direct consequence of the circuit model.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FrictionConfig
from .rlc_block import RLCTransformerBlock
from .rlc_neuron import RLCNeuron
from .friction_gate import FrictionGate
from .curriculum import SharpnessCurriculum


class RLCFrictionLM(nn.Module):
    def __init__(self, config: FrictionConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop    = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [RLCTransformerBlock(config) for _ in range(config.n_layers)]
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
            nn.init.normal_(block.attn.proj.weight,         std=0.02 * scale)
            nn.init.normal_(block.rlc_block.W_out.weight,   std=0.02 * scale)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args
        ────
        idx     : [B, T]  token indices
        targets : [B, T]  next-token targets, or None at inference

        Returns
        ───────
        logits : [B, T, vocab_size]
        loss   : cross-entropy scalar, or None
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len

        device = idx.device
        pos = torch.arange(T, device=device, dtype=torch.long)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        # Circuit state and friction momentum — both None at first block
        rlc_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        momentum:  Optional[torch.Tensor] = None

        for block in self.blocks:
            x, rlc_state, momentum = block(x, rlc_state, momentum)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ── Generation ───────────────────────────────────────────────────────────

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
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def circuit_report(self, idx: torch.Tensor) -> dict:
        """
        Run a forward pass and return per-layer circuit physics stats.

        Reports for each layer:
          ω₀   — natural frequency (should diverge after training)
          ζ    — damping ratio (1=critical, <1=resonant, >1=overdamped)
          sparsity — fraction of neurons zeroed by friction gate
          μ_s, μ_k — learned friction thresholds
        """
        self.eval()
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        report = {}
        rlc_state = None
        momentum  = None

        for i, block in enumerate(self.blocks):
            stats = block.rlc_block.diagnostics(block.ln2(x))
            report[f"layer_{i}"] = stats
            x, rlc_state, momentum = block(x, rlc_state, momentum)

        # Overall sparsity
        report["overall_sparsity"] = sum(
            v["sparsity"] for v in report.values() if isinstance(v, dict)
        ) / self.config.n_layers

        return report

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"params={self.param_count()/1e6:.1f}M  "
            f"layers={self.config.n_layers}  "
            f"d_model={self.config.d_model}  "
            f"RLC(L={self.config.rlc_L_init}, R={self.config.rlc_R_init}, C={self.config.rlc_C_init})"
        )

    def print_circuit_report(self, idx: torch.Tensor) -> None:
        """Pretty-print the per-layer circuit stats."""
        report = self.circuit_report(idx)

        print("\n── RLC Circuit Report ─────────────────────────────────────────────")
        print(f"{'Layer':<10} {'ω₀ mean':>10} {'ζ mean':>10} "
              f"{'Underdamp%':>12} {'Sparsity':>10} {'μ_s':>8} {'μ_k':>8}")
        print("─" * 72)

        for key, val in report.items():
            if not isinstance(val, dict):
                continue
            print(
                f"{key:<10} "
                f"{val['omega_0_mean']:>10.3f} "
                f"{val['zeta_mean']:>10.3f} "
                f"{val['underdamped_%']:>11.1f}% "
                f"{val['sparsity']:>9.1%} "
                f"{val['mu_s_mean']:>8.4f} "
                f"{val['mu_k_mean']:>8.4f}"
            )

        print("─" * 72)
        print(f"{'OVERALL':<10} {'':>10} {'':>10} {'':>12} "
              f"{report['overall_sparsity']:>9.1%}")
        print()
