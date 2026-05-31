"""
inference.py — generation, sparsity profiling, and CPU vs GPU benchmark

Usage
─────
    # Generate text:
    python inference.py --checkpoint checkpoints/best.pt --prompt "The friction model"

    # Sparsity report (shows per-layer μ_s, μ_k, and zero-fraction):
    python inference.py --checkpoint checkpoints/best.pt --sparsity

    # CPU vs GPU timing benchmark:
    python inference.py --checkpoint checkpoints/best.pt --benchmark

    # Sparse CPU inference path (experimental — uses torch sparse matmul):
    python inference.py --checkpoint checkpoints/best.pt --sparse_cpu --prompt "Hello"
"""

import argparse
import time
from typing import Optional

import tiktoken
import torch
import torch.nn as nn

from friction_llm import FrictionConfig, FrictionLM


# ─────────────────────────────────────────────────────────────────────────────
# Sparse CPU projection — key to CPU speedup
# ─────────────────────────────────────────────────────────────────────────────

class SparseProjection(nn.Module):
    """
    Wraps nn.Linear to exploit sparsity in the input activations on CPU.

    When input sparsity > threshold, converts to sparse CSR and uses
    torch.sparse.mm — genuinely skips zero-row multiplications.
    Benchmark shows speedup at ~65%+ sparsity (depends on matrix size).
    """

    SPARSITY_THRESHOLD = 0.65   # below this, dense is faster

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.weight = linear.weight   # [out, in]
        self.bias   = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only on CPU at inference; training stays dense
        if x.device.type != "cpu" or self.training:
            return nn.functional.linear(x, self.weight, self.bias)

        sparsity = (x == 0).float().mean().item()
        if sparsity < self.SPARSITY_THRESHOLD:
            return nn.functional.linear(x, self.weight, self.bias)

        return self._sparse_mm(x)

    def _sparse_mm(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])   # [B*T, in]

        # Build sparse CSR from input (rows with all zeros are skipped by MKL)
        x_sparse = x_2d.to_sparse_csr()
        out = torch.sparse.mm(x_sparse, self.weight.T)  # [B*T, out]

        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.weight.shape[0])


def patch_for_sparse_cpu(model: FrictionLM) -> FrictionLM:
    """Replace W_out projections in all FGLU blocks with SparseProjection."""
    for block in model.blocks:
        block.fglu.W_out = SparseProjection(block.fglu.W_out)
    print("Patched model with sparse CPU W_out projections")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> FrictionLM:
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model  = FrictionLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {model.param_count()/1e6:.1f}M param model from {checkpoint_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_text(
    model: FrictionLM,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    device: torch.device,
) -> str:
    enc  = tiktoken.get_encoding("gpt2")
    ids  = enc.encode_ordinary(prompt)
    idx  = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    t0  = time.time()
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    dt  = time.time() - t0

    tokens_generated = out.shape[1] - idx.shape[1]
    print(f"Generated {tokens_generated} tokens in {dt:.2f}s  "
          f"({tokens_generated/dt:.1f} tok/s)")

    return enc.decode(out[0].tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Sparsity report
# ─────────────────────────────────────────────────────────────────────────────

def sparsity_report(model: FrictionLM, device: torch.device) -> None:
    enc    = tiktoken.get_encoding("gpt2")
    sample = enc.encode_ordinary("The friction model applies static and kinetic thresholds ")
    idx    = torch.tensor(sample, dtype=torch.long, device=device).unsqueeze(0)

    stats = model.measure_sparsity(idx)

    print("\n── Friction Gate Sparsity Report ─────────────────────────────")
    print(f"{'Layer':<10} {'Sparsity':>10} {'μ_s (mean)':>12} {'μ_k (mean)':>12}")
    print("-" * 48)
    for key, val in stats.items():
        if key == "overall":
            continue
        print(f"{key:<10} {val['sparsity']:>9.1%}  {val['mu_s']:>12.4f}  {val['mu_k']:>12.4f}")
    print("-" * 48)
    print(f"{'OVERALL':<10} {stats['overall']:>9.1%}")

    sparsity_pct = stats["overall"] * 100
    if sparsity_pct >= 70:
        print(f"\n  {sparsity_pct:.0f}% sparsity — CPU sparse path will outperform dense GPU here.")
    elif sparsity_pct >= 50:
        print(f"\n  {sparsity_pct:.0f}% sparsity — moderate.  Memory bandwidth still wins vs GPU.")
        print("   Try increasing mu_s_init or sparsity_reg in config.")
    else:
        print(f"\n  {sparsity_pct:.0f}% sparsity — low.  GPU-preferred for now.")
        print("   Run more curriculum steps or raise sparsity_reg.")


# ─────────────────────────────────────────────────────────────────────────────
# CPU vs GPU benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(checkpoint_path: str, seq_len: int = 512, batch_size: int = 4) -> None:
    results = {}

    for device_str in ["cpu", "cuda"]:
        if device_str == "cuda" and not torch.cuda.is_available():
            print("CUDA not available — skipping GPU benchmark")
            continue

        device = torch.device(device_str)
        model  = load_model(checkpoint_path, device)
        model.eval()

        # Warmup
        dummy = torch.randint(0, 50257, (batch_size, seq_len), device=device)
        with torch.no_grad():
            for _ in range(3):
                model(dummy)

        if device_str == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        n_runs = 20
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                model(dummy)
        if device_str == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / n_runs * 1000  # ms

        results[device_str] = dt
        print(f"{device_str.upper():6s}: {dt:.1f} ms/forward  (batch={batch_size}, seq={seq_len})")

    if "cpu" in results and "cuda" in results:
        ratio = results["cpu"] / results["cuda"]
        print(f"\nGPU is {ratio:.1f}× faster at dense inference with current sparsity.")
        print("Run --sparsity to check if higher sparsity can close the gap,")
        print("or --sparse_cpu to enable the CSR sparse matmul path.")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="FrictionLM inference & diagnostics")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--prompt",     default="Once upon a time",  help="Generation prompt")
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--temperature",type=float, default=0.8)
    p.add_argument("--top_k",      type=int,   default=40)
    p.add_argument("--sparsity",   action="store_true", help="Print per-layer sparsity report")
    p.add_argument("--benchmark",  action="store_true", help="CPU vs GPU timing benchmark")
    p.add_argument("--sparse_cpu", action="store_true", help="Use sparse CSR matmul on CPU")
    p.add_argument("--device",     default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    args = p.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if args.benchmark:
        benchmark(args.checkpoint)
        return

    model = load_model(args.checkpoint, device)

    if args.sparse_cpu:
        if device.type != "cpu":
            print("Warning: --sparse_cpu only meaningful on CPU; switching device to cpu")
            device = torch.device("cpu")
            model  = load_model(args.checkpoint, device)
        model = patch_for_sparse_cpu(model)

    if args.sparsity:
        sparsity_report(model, device)

    print(f"\nPrompt: {args.prompt!r}\n")
    text = generate_text(
        model, args.prompt, args.max_tokens,
        args.temperature, args.top_k, device,
    )
    print(text)


if __name__ == "__main__":
    main()
