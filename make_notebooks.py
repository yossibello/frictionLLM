"""
make_notebooks.py — generate JupyterHub and Kaggle notebooks from source.

Reads the actual friction_llm/*.py files so the notebooks always stay in sync.
Run:  python make_notebooks.py

Focus: PhysicsLM (selective multi-pole SSM, no attention) head-to-head against a
standard GPT-2-style transformer baseline — same params budget, same data, same steps.
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Notebook helpers
# ─────────────────────────────────────────────────────────────────────────────

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}

def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }

def notebook(cells, accelerator="GPU"):
    meta = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    }
    if accelerator:
        meta["accelerator"] = accelerator
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}

def write_nb(nb, path):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  wrote {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cell content
# ─────────────────────────────────────────────────────────────────────────────

TITLE_MD = """\
# PhysicsLM vs GPT-2 — Head-to-Head
### Selective multi-pole SSM (no attention) vs a standard transformer

**PhysicsLM** replaces attention with a `CoupledOscillatorMixer`: each channel is a
bank of damped RLC resonators (a diagonal state-space model, S4D family) convolved
causally along the sequence — **O(T·logT)**, not O(T²) — with a Mamba-style
content gate for selectivity. The FFN is an RLC circuit neuron with a friction gate
(sparse at inference).

**GPT-2 baseline** is the control: same depth, same width, dot-product attention +
GELU FFN. Identical data, steps, and optimiser. PhysicsLM must beat this on
validation loss to justify the architecture.

**What to watch during training:**
- `val` — the number that matters. Lower wins.
- `bank ω₀` — spread of resonant frequencies *within* each channel's pole bank
  (starts ~3.3 thanks to the spread init; the old single-pole model started at 0).
- `underdamped` — layers/poles that went resonant (ζ < 1).
- `sparse` — fraction of FFN neurons silent (CPU-inference advantage; target 70%+).
"""

SETUP_CELL = """\
import os, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed seed — both models see batches sampled from the same distribution
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cpu")
)
n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
print(f"Device : {device}")
if device.type == "cuda":
    for i in range(n_gpus):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB")
    print(f"GPUs   : {n_gpus}  ({'DataParallel' if n_gpus > 1 else 'single GPU'})")
"""

CLONE_CELL = """\
!pip install tiktoken datasets --quiet

import os, subprocess, sys, shutil

# Always use absolute path — prevents recursive cloning on re-runs
REPO = "/kaggle/working/frictionLLM"

# ── One-time cleanup: remove any nested frictionLLM/frictionLLM recursion ──
nested = os.path.join(REPO, "frictionLLM")
if os.path.exists(nested):
    print(f"Cleaning up recursive clone at {nested} ...")
    shutil.rmtree(nested)
    print("Done.")

# ── Clone or pull ──────────────────────────────────────────────────────────
if not os.path.exists(REPO):
    subprocess.run(["git", "clone",
                    "https://github.com/yossibello/frictionLLM.git", REPO],
                   check=True)
    print("Cloned →", REPO)
else:
    subprocess.run(["git", "-C", REPO, "pull"], check=True)
    print("Updated →", REPO)

sys.path.insert(0, REPO)
os.chdir(REPO)
print("Working dir:", os.getcwd())
"""

CKPT_FINDER_CELL = """\
import subprocess, os
result = subprocess.run(
    ['find', '/kaggle/working', '-name', '*.pt', '-not', '-path', '*/.git/*'],
    capture_output=True, text=True
)
files = sorted(result.stdout.strip().split('\\n'))
if files and files[0]:
    print('Found checkpoints:')
    for f in files:
        print(f'  {f}  ({os.path.getsize(f)/1e6:.0f} MB)')
else:
    print('No checkpoints found yet.')
"""

IMPORT_CELL = """\
from friction_llm import FrictionConfig, PhysicsLM, BaselineLM, SharpnessCurriculum

cfg_test = FrictionConfig.tiny()
cfg_test.use_rlc = True
cfg_test.use_coupled_mixer = True
m_phy  = PhysicsLM(cfg_test)
m_base = BaselineLM(cfg_test)
print("Import OK")
print(f"  PhysicsLM  (tiny): {m_phy.param_count()/1e6:.2f}M params  (no attention)")
print(f"  BaselineLM (tiny): {m_base.param_count()/1e6:.2f}M params  (GPT-2 style)")
del m_phy, m_base, cfg_test
"""

DATA_CELL = """\
import os
os.makedirs("data", exist_ok=True)

try:
    from datasets import load_dataset
    print("Loading WikiText-103 (~103M tokens)...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    with open("data/input.txt", "w", encoding="utf-8") as f:
        for split in ["train", "validation", "test"]:
            for row in ds[split]:
                text = row["text"].strip()
                if text:
                    f.write(text + "\\n")
    print(f"Saved: {os.path.getsize('data/input.txt')/1e6:.0f} MB")
except Exception as e:
    import urllib.request
    print(f"Falling back to TinyShakespeare ({e})")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "data/input.txt"
    )
    print(f"Downloaded: {os.path.getsize('data/input.txt')/1e6:.1f} MB")
"""

TOKENIZE_CELL = """\
import tiktoken, numpy as np

enc = tiktoken.get_encoding("gpt2")
with open("data/input.txt", encoding="utf-8") as f:
    text = f.read()

tokens = np.array(enc.encode_ordinary(text), dtype=np.uint16)
split  = int(0.9 * len(tokens))
tokens[:split].tofile("data/train.bin")
tokens[split:].tofile("data/val.bin")
print(f"Train : {split:,} tokens")
print(f"Val   : {len(tokens)-split:,} tokens")
"""

DATALOADER_CELL = """\
class TokenDataset:
    def __init__(self, path, seq_len):
        self.data    = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def get_batch(self, batch_size, device):
        ix = torch.randint(len(self), (batch_size,))
        x  = torch.stack([torch.from_numpy(
                self.data[i : i+self.seq_len].astype(np.int64)) for i in ix])
        y  = torch.stack([torch.from_numpy(
                self.data[i+1 : i+1+self.seq_len].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

SEQ_LEN  = 256
train_ds = TokenDataset("data/train.bin", SEQ_LEN)
val_ds   = TokenDataset("data/val.bin",   SEQ_LEN)
print(f"Seq len : {SEQ_LEN}")
print(f"Train   : {len(train_ds):,} positions")
print(f"Val     : {len(val_ds):,} positions")
"""

TRAIN_FUNC_CELL = """\
from friction_llm import FrictionConfig, PhysicsLM, BaselineLM, SharpnessCurriculum

def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def circuit_snapshot(base):
    \"\"\"Per-layer ω₀, ζ, underdamped %, within-bank ω₀ spread, sparsity.\"\"\"
    rows = []
    for block in base.blocks:
        # PhysicsBlock → mixer (CoupledOscillatorMixer) + filter (RLCFrictionBlock)
        # BaselineBlock → neither
        mixer     = getattr(block, "mixer", None)
        rlc_block = getattr(block, "filter", None) or getattr(block, "rlc_block", None)
        if mixer is not None and hasattr(mixer, "wave_stats"):
            ws   = mixer.wave_stats()
            fric = getattr(rlc_block, "friction", None) if rlc_block else None
            rows.append({
                "omega_0":           ws["omega_0_mean"],
                "omega_bank_spread": ws.get("omega_0_spread", 0.0),
                "zeta":              ws["damping_mean"],
                "underdamped_pct":   ws["underdamped_%"],
                "mu_s":              fric.mu_s.mean().item() if fric else 0.0,
            })
        else:
            rows.append({"omega_0": 1.0, "omega_bank_spread": 0.0,
                         "zeta": 1.0, "underdamped_pct": 0.0, "mu_s": 0.0})
    return rows

def train_model(model, train_ds, val_ds,
                max_steps=10000, batch_size=8,
                lr=3e-4, log_every=100,
                ckpt_every=1000, ckpt_dir="checkpoints"):

    # Always save outside the repo so checkpoints survive git operations
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join("/kaggle/working", ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints → {ckpt_dir}")
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    effective_batch = batch_size * max(n_gpus, 1)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel: {n_gpus} GPUs  effective batch={effective_batch}")

    base = unwrap(model)

    # Physics params (incl. pole_mix) are NOT weight-decayed
    physics, other = [], []
    for n, p in base.named_parameters():
        if any(k in n for k in ("raw_mu","raw_ratio","log_L","log_R","log_C","pole_mix")):
            physics.append(p)
        else:
            other.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": other,   "weight_decay": 0.1},
         {"params": physics, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95)
    )

    curriculum = SharpnessCurriculum(base, base.config)

    history = {
        "step": [], "loss": [], "val_loss": [], "sparsity": [],
        "omega_spread": [], "omega_bank_spread": [],
        "underdamped_layers": [], "sharpness": [], "layers": [],
    }
    best_val_loss = float("inf")
    lr_min = lr / 10
    t0     = time.time()

    for step in range(max_steps + 1):
        # Cosine LR with warmup
        warmup = min(500, max_steps // 10)
        if step < warmup:
            cur_lr = lr * step / max(warmup, 1)
        else:
            progress = (step - warmup) / (max_steps - warmup)
            cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        x, y = train_ds.get_batch(effective_batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)
        if loss.dim() > 0:
            loss = loss.mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        sharpness = curriculum.step()

        # ── Log ──────────────────────────────────────────────────────────────
        if step % log_every == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = val_ds.get_batch(effective_batch, device)
                _, vloss = model(xv, yv)
                if vloss.dim() > 0:
                    vloss = vloss.mean()

            sample, _ = val_ds.get_batch(4, device)
            if hasattr(base, "circuit_report"):
                report   = base.circuit_report(sample)
                sparsity = report["overall_sparsity"]
                snap     = circuit_snapshot(base)
            else:
                sparsity = 0.0   # baseline has no friction gate
                snap     = [{"omega_0": 1.0, "zeta": 1.0,
                             "omega_bank_spread": 0.0, "underdamped_pct": 0.0}
                            for _ in range(base.config.n_layers)]
            model.train()

            omegas      = [r["omega_0"] for r in snap]
            zetas       = [r["zeta"]    for r in snap]
            omega_spread = max(omegas) - min(omegas)
            bank_spread  = float(np.mean([r.get("omega_bank_spread", 0.0) for r in snap]))
            underdamped  = sum(1 for z in zetas if z < 1.0)

            dt = (time.time() - t0) / max(step, 1)
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["val_loss"].append(vloss.item())
            history["sparsity"].append(sparsity)
            history["omega_spread"].append(omega_spread)
            history["omega_bank_spread"].append(bank_spread)
            history["underdamped_layers"].append(underdamped)
            history["sharpness"].append(sharpness)
            history["layers"].append(snap)

            is_best = vloss.item() < best_val_loss
            if is_best:
                best_val_loss = vloss.item()
                torch.save({
                    "model":      base.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "curriculum": curriculum.state_dict(),
                    "step":       step,
                    "val_loss":   best_val_loss,
                    "history":    history,
                    "config":     base.config,
                }, os.path.join(ckpt_dir, "best.pt"))

            print(
                f"step {step:5d} | loss {loss.item():.4f} | val {vloss.item():.4f} "
                f"{'★' if is_best else ' '}"
                f"| sparse {sparsity:.1%} | bank ω₀ {bank_spread:.2f} "
                f"| underdamped {underdamped}/{len(snap)} "
                f"| sharp {sharpness:.1f} | {dt*1000:.0f}ms/step"
            )

        # ── Checkpoint (weights only, ~470 MB) ─────────────────────────────────
        if step % ckpt_every == 0 and step > 0:
            path = f"{ckpt_dir}/step_{step:06d}.pt"
            torch.save({"model": base.state_dict(),
                        "config": base.config, "step": step}, path)
            print(f"  → saved {path}  (model only)")

    return history, base
"""

BASELINE_TRAIN_CELL = """\
# ── GPT-2 baseline (control) ─────────────────────────────────────────────────
cfg_b = FrictionConfig.medium()
cfg_b.max_seq_len = SEQ_LEN
baseline = BaselineLM(cfg_b).to(device)
print(f"BaselineLM (GPT-2 style): {baseline.param_count()/1e6:.1f}M params")
print(f"  {cfg_b.n_layers} layers · d_model {cfg_b.d_model} · {cfg_b.n_heads} heads · GELU FFN")

history_base, baseline = train_model(
    baseline, train_ds, val_ds,
    max_steps=10000, batch_size=8, lr=3e-4,
    log_every=100, ckpt_every=1000, ckpt_dir="checkpoints/baseline",
)
"""

PHYSICS_TRAIN_CELL = """\
# ── PhysicsLM (selective multi-pole SSM, no attention) ───────────────────────
cfg_p = FrictionConfig.medium()
cfg_p.max_seq_len       = SEQ_LEN
cfg_p.use_rlc           = True
cfg_p.use_coupled_mixer = True
cfg_p.mu_s_init         = 0.05        # charge scale is small → low friction threshold
cfg_p.rlc_dt            = 0.3         # integration / kernel sampling step
cfg_p.rlc_filter_mode   = 'learnable' # each FFN neuron learns its own LP/BP/HP mix
cfg_p.mixer_L_c_init    = 5.0         # coupling inductance (adds restoring stiffness)
cfg_p.mixer_n_poles     = 8           # resonators per channel (S4D-style bank)

physics_model = PhysicsLM(cfg_p).to(device)
print(f"PhysicsLM      : {physics_model.param_count()/1e6:.1f}M params  (no attention, O(T·logT))")
print(f"  {cfg_p.n_layers} layers · d_model {cfg_p.d_model} · {cfg_p.mixer_n_poles} poles/channel + selective gate")
print(f"  vs baseline {baseline.param_count()/1e6:.1f}M  (PhysicsLM's gate branch adds the delta)")

history_phy, physics_model = train_model(
    physics_model, train_ds, val_ds,
    max_steps=10000, batch_size=8, lr=3e-4,
    log_every=100, ckpt_every=1000,
    ckpt_dir='/kaggle/working/checkpoints/physics',
)
"""

COMPARE_CELL = """\
import matplotlib.pyplot as plt

runs = [
    ('GPT-2 baseline', history_base, 'gray'),
    ('PhysicsLM (no attn)', history_phy, 'darkorange'),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
fig.suptitle('PhysicsLM vs GPT-2  —  same params budget, same data, same steps',
             fontweight='bold')

ax = axes[0]
for label, hist, c in runs:
    ax.plot(hist['step'], hist['val_loss'], label=label, color=c, lw=2)
ax.set(xlabel='Step', ylabel='Val Loss', title='Validation Loss  ← lower wins')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
for label, hist, c in runs:
    ax.plot(hist['step'], hist['loss'], label=label, color=c, alpha=0.7, lw=1.5)
ax.set(xlabel='Step', ylabel='Train Loss', title='Training Loss')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(history_phy['step'], [s*100 for s in history_phy['sparsity']],
        color='darkorange', lw=2, label='PhysicsLM FFN')
ax.axhline(0, color='gray', lw=2, label='GPT-2 (dense, 0%)')
ax.axhline(70, color='green', lw=1, ls='--', label='CPU target')
ax.set(xlabel='Step', ylabel='Sparsity %', title='FFN Gate Sparsity', ylim=[-5, 105])
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('physics_vs_gpt2.png', dpi=150, bbox_inches='tight')
plt.show()

print('\\n── Final / best validation loss ─────────────────')
for label, hist, _ in runs:
    final = hist['val_loss'][-1]
    best  = min(hist['val_loss'])
    print(f'  {label:<22} final {final:.4f}   best {best:.4f}   ppl {math.exp(best):.1f}')
winner = min(runs, key=lambda r: min(r[1]['val_loss']))[0]
print(f'\\n  WINNER (best val): {winner}')
print(f'  PhysicsLM FFN sparsity: {history_phy[\"sparsity\"][-1]:.1%}')
"""

WAVE_REPORT_CELL = """\
import tiktoken
enc = tiktoken.get_encoding("gpt2")
sample = torch.tensor(
    enc.encode_ordinary("The relationship between"),
    dtype=torch.long, device=device
).unsqueeze(0)

physics_model.eval()
physics_model.print_wave_report(sample)
"""

FILTER_WEIGHTS_CELL = """\
import matplotlib.pyplot as plt
import numpy as np

physics_model.eval()
sample, _ = val_ds.get_batch(4, device)
report = physics_model.circuit_report(sample)

n_layers = physics_model.config.n_layers
lp, bp, hp = [], [], []
for i in range(n_layers):
    fw = report[f'layer_{i}'].get('filter_weights', {})
    lp.append(fw.get('lowpass',  0))
    bp.append(fw.get('bandpass', 0))
    hp.append(fw.get('highpass', 0))

x = np.arange(n_layers)
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(x, lp, label='Low-pass  (slow/global)', color='#4878cf')
ax.bar(x, bp, bottom=lp, label='Band-pass (resonant)', color='#6acc65')
ax.bar(x, hp, bottom=[l+b for l,b in zip(lp,bp)],
       label='High-pass (fast/local)', color='#d65f5f')
ax.set(xlabel='Layer', ylabel='Filter weight %', ylim=[0,100],
       title='Learned FFN filter type per layer  (emerged from training)',
       xticks=x, xticklabels=[f'L{i}' for i in range(n_layers)])
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('filter_weights.png', dpi=150, bbox_inches='tight')
plt.show()
"""

PHYSICS_PLOT_CELL = """\
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

h = history_phy
steps = h["step"]
n_layers = len(h["layers"][0]) if h["layers"] else 0

fig = plt.figure(figsize=(18, 8))
fig.suptitle("PhysicsLM — physics emerging during training", fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax = fig.add_subplot(gs[0, :2])
ax.plot(steps, h["loss"], label="train", alpha=0.7, lw=1.5)
ax.plot(steps, h["val_loss"], label="val", lw=2)
ax.set(xlabel="Step", ylabel="Loss", title="Loss"); ax.legend(); ax.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(steps, [s*100 for s in h["sparsity"]], color="steelblue", lw=2)
ax2.axhline(70, ls="--", color="gray", alpha=0.5, label="CPU target")
ax2.set(xlabel="Step", ylabel="Sparsity %", title="FFN Gate Sparsity")
ax2.legend(); ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(steps, h["omega_bank_spread"], color="darkorange", lw=2)
ax3.set(xlabel="Step", ylabel="mean std(ω₀) within bank",
        title="Within-channel pole-frequency spread  (diversity of resonators)")
ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(steps, h["underdamped_layers"], color="crimson", lw=2, drawstyle="steps-post")
ax4.set(xlabel="Step", ylabel="Layers with ζ < 1",
        title=f"Resonant layers (of {n_layers})")
ax4.set_yticks(range(n_layers + 1)); ax4.grid(alpha=0.3)

plt.savefig("physics_training.png", dpi=150, bbox_inches="tight")
plt.show()
"""

GENERATE_CELL = """\
import tiktoken
enc = tiktoken.get_encoding("gpt2")

prompts = ["The theory of", "In the year 1900", "Scientists discovered that"]
physics_model.eval()
for prompt in prompts:
    ids = enc.encode_ordinary(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = physics_model.generate(idx, max_new_tokens=120, temperature=0.8, top_k=40)
    print(f"Prompt: {prompt!r}")
    print(enc.decode(out[0].tolist()))
    print("-" * 60)
"""

RESUME_MD = """\
## Resuming PhysicsLM from checkpoint

Checkpoints are saved to `/kaggle/working/checkpoints/physics/` (absolute path).

```python
import glob, torch
from friction_llm import PhysicsLM

ckpts = sorted(glob.glob("/kaggle/working/checkpoints/physics/step_*.pt"))
print("Available:", ckpts)

ckpt  = torch.load(ckpts[-1], map_location=device)
cfg   = ckpt["config"]
physics_model = PhysicsLM(cfg).to(device)
physics_model.load_state_dict(ckpt["model"])
print(f"Resumed from step {ckpt['step']}")

# NOTE: the mixer architecture changed (analytic kernel + multi-pole bank +
# selective gate). Checkpoints from the OLD single-pole PhysicsLM are NOT
# compatible — start a fresh run for the upgraded model.

history_phy, physics_model = train_model(
    physics_model, train_ds, val_ds,
    max_steps=10000, batch_size=8, lr=3e-4,
    ckpt_dir="/kaggle/working/checkpoints/physics",
)
```
"""


# ─────────────────────────────────────────────────────────────────────────────
# Build Kaggle notebook — PhysicsLM vs GPT-2
# ─────────────────────────────────────────────────────────────────────────────

def build_kaggle():
    cells = []
    cells.append(md(TITLE_MD))

    cells.append(md("## 1 · Clone repo & install"))
    cells.append(code(CLONE_CELL))

    cells.append(md("## 2 · GPU setup"))
    cells.append(code(SETUP_CELL))

    cells.append(md("## Find existing checkpoints (run if resuming)"))
    cells.append(code(CKPT_FINDER_CELL))

    cells.append(md("## 3 · Verify imports"))
    cells.append(code(IMPORT_CELL))

    cells.append(md("## 4 · Data — WikiText-103"))
    cells.append(code(DATA_CELL))
    cells.append(code(TOKENIZE_CELL))
    cells.append(code(DATALOADER_CELL))

    cells.append(md("## 5 · Training function\n\nShared loop for both models. Logs val loss, "
                    "within-bank ω₀ spread, resonant layers, and FFN sparsity every `log_every` steps."))
    cells.append(code(TRAIN_FUNC_CELL))

    cells.append(md("## 6 · Train — GPT-2 baseline (control)\n\n"
                    "Standard transformer: same depth/width, dot-product attention + GELU FFN. "
                    "PhysicsLM must beat this val loss to justify the architecture."))
    cells.append(code(BASELINE_TRAIN_CELL))

    cells.append(md("## 7 · Train — PhysicsLM (no attention)\n\n"
                    "Selective multi-pole RLC SSM replaces attention (O(T·logT)); "
                    "RLC friction circuit replaces the FFN (sparse at inference)."))
    cells.append(code(PHYSICS_TRAIN_CELL))

    cells.append(md("## 8 · Head-to-head: PhysicsLM vs GPT-2\n\nThe definitive test — who wins?"))
    cells.append(code(COMPARE_CELL))

    cells.append(md("## 9 · PhysicsLM training dynamics"))
    cells.append(code(PHYSICS_PLOT_CELL))

    cells.append(md("## 10 · Wave / circuit report (per layer)"))
    cells.append(code(WAVE_REPORT_CELL))

    cells.append(md("## 11 · Learned FFN filter type per layer\n\n"
                    "LP=low-pass (slow/global), BP=band-pass (resonant), HP=high-pass (fast/local)."))
    cells.append(code(FILTER_WEIGHTS_CELL))

    cells.append(md("## 12 · Text generation"))
    cells.append(code(GENERATE_CELL))

    cells.append(md(RESUME_MD))
    return notebook(cells, accelerator="GPU")


# ─────────────────────────────────────────────────────────────────────────────
# Build JupyterHub notebook (local, single machine)
# ─────────────────────────────────────────────────────────────────────────────

def build_jupyterhub():
    cells = []
    cells.append(md(TITLE_MD))
    cells.append(md("## Setup\n\nRun from repo root. Requires: `torch tiktoken datasets`"))
    cells.append(code("import sys; sys.path.insert(0, '.')\n" + SETUP_CELL))
    cells.append(code(IMPORT_CELL))
    cells.append(md("## Data"))
    cells.append(code(DATA_CELL))
    cells.append(code(TOKENIZE_CELL))
    cells.append(code(DATALOADER_CELL))
    cells.append(md("## Training function"))
    cells.append(code(TRAIN_FUNC_CELL))
    cells.append(md("## Train GPT-2 baseline"))
    cells.append(code(BASELINE_TRAIN_CELL))
    cells.append(md("## Train PhysicsLM"))
    cells.append(code(PHYSICS_TRAIN_CELL))
    cells.append(md("## Compare"))
    cells.append(code(COMPARE_CELL))
    cells.append(md("## PhysicsLM dynamics"))
    cells.append(code(PHYSICS_PLOT_CELL))
    cells.append(md("## Wave report"))
    cells.append(code(WAVE_REPORT_CELL))
    cells.append(md("## Generate"))
    cells.append(code(GENERATE_CELL))
    cells.append(md(RESUME_MD))
    return notebook(cells, accelerator=None)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    print("Building notebooks...")
    write_nb(build_kaggle(),
             os.path.join(ROOT, "notebooks", "friction_llm_kaggle.ipynb"))
    write_nb(build_jupyterhub(),
             os.path.join(ROOT, "notebooks", "friction_llm_jupyterhub.ipynb"))
    print("Done.")
