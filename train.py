"""
train.py — FrictionLM / RLCFrictionLM training script

Usage
─────
    # Prepare data first (tokenises a raw .txt file once):
    python train.py --prepare --data_path data/my_corpus.txt

    # Train friction model (R only):
    python train.py --model friction --config small --batch_size 32 --max_steps 20000

    # Train full RLC model (L + R + C):
    python train.py --model rlc --config small --batch_size 32 --max_steps 20000

    # Resume from checkpoint:
    python train.py --resume checkpoints/step_10000.pt

The script tokenises with tiktoken (GPT-2 vocab) and saves binary .bin shards
so the training loop never touches raw text.
"""

import argparse
import math
import os
import time

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from friction_llm import FrictionConfig, FrictionLM, RLCFrictionLM, PhysicsLM, SharpnessCurriculum


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(data_path: str, out_dir: str = "data") -> None:
    """Tokenise a raw .txt file → train.bin + val.bin (90/10 split)."""
    os.makedirs(out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    print(f"Reading {data_path} ...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = enc.encode_ordinary(text)
    tokens = np.array(tokens, dtype=np.uint16)

    split = int(0.9 * len(tokens))
    train_tokens = tokens[:split]
    val_tokens   = tokens[split:]

    train_path = os.path.join(out_dir, "train.bin")
    val_path   = os.path.join(out_dir, "val.bin")
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    print(f"train: {len(train_tokens):,} tokens  →  {train_path}")
    print(f"val:   {len(val_tokens):,} tokens  →  {val_path}")


class TokenDataset:
    """Memory-mapped dataset from a pre-tokenised .bin file."""

    def __init__(self, bin_path: str, seq_len: int) -> None:
        self.data    = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def get_batch(self, batch_size: int, device: torch.device) -> tuple:
        ix  = torch.randint(len(self), (batch_size,))
        x   = torch.stack([torch.from_numpy(self.data[i : i + self.seq_len].astype(np.int64)) for i in ix])
        y   = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.seq_len].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup → cosine decay → floor
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, max_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / warmup
    if step > max_steps:
        return lr_min
    progress = (step - warmup) / (max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    model: FrictionLM,
    optimizer: torch.optim.Optimizer,
    curriculum: SharpnessCurriculum,
    step: int,
    val_loss: float,
    config: FrictionConfig,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "curriculum": curriculum.state_dict(),
            "step":       step,
            "val_loss":   val_loss,
            "config":     config,
        },
        path,
    )
    print(f"  ✓ saved {path}")


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: FrictionLM,
    dataset: TokenDataset,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = dataset.get_batch(batch_size, device)
        with autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print(f"CPU: {os.cpu_count()} cores — sparsity will pay off here")

    use_amp = args.use_amp and device.type == "cuda"

    # ── Config ───────────────────────────────────────────────────────────────
    if args.resume:
        ckpt   = load_checkpoint(args.resume, device)
        config = ckpt["config"]
        print(f"Resuming from step {ckpt['step']}")
    else:
        preset = {"tiny":   FrictionConfig.tiny,
                  "small":  FrictionConfig.small,
                  "medium": FrictionConfig.medium,
                  "large":  FrictionConfig.large}.get(args.config)
        config = preset() if preset else FrictionConfig()
        config.use_amp = use_amp

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = TokenDataset(os.path.join(args.data_dir, "train.bin"), config.max_seq_len)
    val_ds   = TokenDataset(os.path.join(args.data_dir, "val.bin"),   config.max_seq_len)
    print(f"Train tokens: {len(train_ds):,}   Val tokens: {len(val_ds):,}")

    # ── Model ────────────────────────────────────────────────────────────────
    model_type = getattr(args, "model_type", "friction")
    if model_type == "rlc":
        config.use_rlc = True
        model = RLCFrictionLM(config).to(device)
        print(f"Model: RLCFrictionLM — {model.param_count()/1e6:.1f} M parameters")
    elif model_type == "physics":
        config.use_rlc = True
        config.use_coupled_mixer = True
        model = PhysicsLM(config).to(device)
        print(f"Model: PhysicsLM (no attention) — {model.param_count()/1e6:.1f} M parameters")
    else:
        model = FrictionLM(config).to(device)
        print(f"Model: FrictionLM — {model.param_count()/1e6:.1f} M parameters")

    if args.compile and hasattr(torch, "compile"):
        print("torch.compile enabled")
        model = torch.compile(model)

    # ── Optimiser ────────────────────────────────────────────────────────────
    # Separate physics params (mu_s, ratio, log_L/R/C) from weight-decayed params
    physics_params, other_params = [], []
    for name, p in model.named_parameters():
        if any(k in name for k in ("raw_mu", "raw_ratio", "log_L", "log_R", "log_C", "pole_mix")):
            physics_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params,  "weight_decay": config.weight_decay},
            {"params": physics_params, "weight_decay": 0.0},  # don't decay physical params
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
    )

    scaler     = GradScaler("cuda", enabled=use_amp)
    curriculum = SharpnessCurriculum(model, config)

    start_step = 0
    if args.resume:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        curriculum.load_state_dict(ckpt["curriculum"])
        start_step = ckpt["step"]

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(start_step, args.max_steps + 1):

        # LR schedule
        lr = get_lr(
            step, warmup=args.lr_warmup, max_steps=args.max_steps,
            lr_max=config.learning_rate, lr_min=config.learning_rate / 10,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        model.train()
        x, y = train_ds.get_batch(args.batch_size, device)

        with autocast("cuda", enabled=use_amp):
            logits, loss = model(x, y)

            # Optional sparsity regularisation (L1 on gate signals).
            # Block layouts differ per model:
            #   FrictionLM   : block.fglu.W_gate      , norm = block.ln2
            #   RLCFrictionLM: block.rlc_block.W_gate , norm = block.ln2
            #   PhysicsLM    : block.filter.W_gate    , norm = block.ln_filt
            if config.sparsity_reg > 0:
                def _gate_l1(block):
                    gate = getattr(block, "fglu", None) or getattr(block, "rlc_block", None) \
                        or getattr(block, "filter", None)
                    norm = getattr(block, "ln2", None) or getattr(block, "ln_filt", None)
                    return gate.W_gate(norm(x)).abs().mean()
                sparse_loss = sum(_gate_l1(block) for block in model.blocks)
                loss = loss + config.sparsity_reg * sparse_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        sharpness = curriculum.step()

        # ── Logging ──────────────────────────────────────────────────────────
        if step % config.log_every == 0:
            dt = (time.time() - t0) / max(step - start_step, 1)
            print(
                f"step {step:6d} | loss {loss.item():.4f} | "
                f"lr {lr:.2e} | sharpness {sharpness:.1f} | {dt*1000:.1f} ms/step"
            )

        # ── Evaluation + sparsity report ─────────────────────────────────────
        if step % config.eval_every == 0 and step > 0:
            val_loss = evaluate(model, val_ds, args.batch_size, config.eval_iters, device, use_amp)

            # Measure sparsity / circuit stats
            sample_x, _ = val_ds.get_batch(4, device)
            model.eval()
            if hasattr(model, "circuit_report"):
                stats = model.circuit_report(sample_x)
                overall_sparsity = stats["overall_sparsity"]
            else:
                stats = model.measure_sparsity(sample_x)
                overall_sparsity = stats["overall"]
            model.train()
            print(
                f"  val_loss={val_loss:.4f} | "
                f"sparsity={overall_sparsity:.1%} | "
                f"{'NEW BEST' if val_loss < best_val_loss else ''}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    os.path.join(config.checkpoint_dir, "best.pt"),
                    model, optimizer, curriculum, step, val_loss, config,
                )

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if step % config.save_every == 0 and step > 0:
            save_checkpoint(
                os.path.join(config.checkpoint_dir, f"step_{step:07d}.pt"),
                model, optimizer, curriculum, step, float("nan"), config,
            )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train FrictionLM")
    p.add_argument("--prepare",    action="store_true",       help="Tokenise data only")
    p.add_argument("--data_path",  default="data/input.txt",  help="Raw text file for --prepare")
    p.add_argument("--data_dir",   default="data",            help="Dir with train.bin / val.bin")
    p.add_argument("--config",     default="small",           choices=["tiny","small","medium","large","custom"])
    p.add_argument("--resume",     default=None,              help="Path to checkpoint to resume")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps",  type=int, default=20000)
    p.add_argument("--lr_warmup",  type=int, default=500)
    p.add_argument("--use_amp",    action="store_true", default=True)
    p.add_argument("--compile",    action="store_true", default=False)
    p.add_argument("--model",      default="friction",  choices=["friction", "rlc", "physics"],
                   dest="model_type", help="friction=R only  |  rlc=full L+R+C circuit")
    args = p.parse_args()

    if args.prepare:
        prepare_data(args.data_path, args.data_dir)
    else:
        train(args)


if __name__ == "__main__":
    main()
