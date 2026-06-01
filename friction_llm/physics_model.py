"""
PhysicsLM — fully physics-native language model.

Zero algebraic attention. Every component is a physical system:

  Token path:
    embeddings → N × PhysicsBlock → LayerNorm → LM head

  Each PhysicsBlock:
    CoupledOscillatorMixer  (replaces attention: wave along sequence)
    RLCFrictionBlock        (replaces FFN: circuit filter per position)

Architecture comparison
───────────────────────
  Standard transformer:  Attention (O(T²), algebraic) + GELU FFN
  RLCFrictionLM:         Attention (O(T²), algebraic) + RLC friction FFN
  PhysicsLM:             Wave mixer (O(T), physical)  + RLC friction FFN  ← this

The wave mixer replaces attention with a 1D RLC transmission line scan.
Long-range dependencies are captured through wave propagation across layers,
not through direct all-to-all connections.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FrictionConfig
from .physics_block import PhysicsBlock
from .curriculum import SharpnessCurriculum
from .rlc_neuron import RLCNeuron
from .friction_gate import FrictionGate


class PhysicsLM(nn.Module):
    def __init__(self, config: FrictionConfig) -> None:
        super().__init__()
        self.config  = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop    = nn.Dropout(config.dropout)

        self.blocks  = nn.ModuleList(
            [PhysicsBlock(config) for _ in range(config.n_layers)]
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
            nn.init.normal_(block.mixer.proj_out.weight,  std=0.02 * scale)
            nn.init.normal_(block.filter.W_out.weight,    std=0.02 * scale)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.config.max_seq_len

        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        rlc_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        momentum:  Optional[torch.Tensor] = None

        for block in self.blocks:
            x, rlc_state, momentum = block(x, rlc_state, momentum)

        x      = self.ln_f(x)
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
            idx_c  = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def wave_report(self, idx: torch.Tensor) -> dict:
        """
        Per-layer wave physics: ω₀, damping, wave speed, coupling strength, sparsity.
        """
        self.eval()
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        report = {}
        rlc_state = None
        momentum  = None

        for i, block in enumerate(self.blocks):
            mixer_stats  = block.mixer.wave_stats()
            filter_stats = block.filter.diagnostics(block.ln_filt(x))
            filter_stats["filter_weights"] = block.filter.rlc.filter_weights()
            report[f"layer_{i}"] = {**mixer_stats, **filter_stats}
            x, rlc_state, momentum = block(x, rlc_state, momentum)

        report["overall_sparsity"] = sum(
            v["sparsity"] for v in report.values() if isinstance(v, dict)
        ) / self.config.n_layers
        return report

    def print_wave_report(self, idx: torch.Tensor) -> None:
        report = self.wave_report(idx)
        print("\n── PhysicsLM Wave Report ──────────────────────────────────────────")
        print(f"{'Layer':<10} {'ω₀':>8} {'ζ':>8} {'wave_v':>8} "
              f"{'coupling':>10} {'sparse':>8}")
        print("─" * 60)
        for key, val in report.items():
            if not isinstance(val, dict):
                continue
            print(
                f"{key:<10} "
                f"{val['omega_0_mean']:>8.3f} "
                f"{val['damping_mean']:>8.3f} "
                f"{val['wave_speed_mean']:>8.3f} "
                f"{val['coupling_mean']:>10.4f} "
                f"{val['sparsity']:>7.1%}"
            )
        print("─" * 60)
        print(f"{'OVERALL':<10} {'':>8} {'':>8} {'':>8} "
              f"{'':>10} {report['overall_sparsity']:>7.1%}\n")

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"params={self.param_count()/1e6:.1f}M  "
            f"layers={self.config.n_layers}  "
            f"d_model={self.config.d_model}  "
            f"[NO ATTENTION — wave propagation only]"
        )
