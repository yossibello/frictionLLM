"""
make_notebooks.py — generate JupyterHub and Kaggle notebooks from source.

Reads the actual friction_llm/*.py files so the notebooks always stay in sync.
Run:  python make_notebooks.py
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
# RLCFrictionLM — Deep Training Run
### Full L+R+C circuit neurons: watching physics emerge across layers

Each neuron is a damped harmonic oscillator with **learned** inductance (L),
resistance (R), and capacitance (C). Starting from identical critical damping,
the network freely discovers which layers should resonate and which should stabilise.

**What to watch during training:**
- `ω₀ spread` — natural frequencies diverging across layers (starts at 0, grows)
- `underdamped layers` — how many layers went resonant (ζ < 1)
- `sparsity` — fraction of neurons silent (target: 70%+ for CPU advantage)
"""

SETUP_CELL = """\
import os, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed seed — both models see batches sampled from same distribution
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

IMPORT_CELL = """\
from friction_llm import (
    FrictionConfig, RLCFrictionLM, BaselineLM,
    SharpnessCurriculum, RLCNeuron
)
cfg_test = FrictionConfig.tiny()
cfg_test.use_rlc = True
m_rlc  = RLCFrictionLM(cfg_test)
m_base = BaselineLM(cfg_test)
print(f"Import OK")
print(f"  RLCFrictionLM (tiny): {m_rlc.param_count()/1e6:.2f}M params")
print(f"  BaselineLM    (tiny): {m_base.param_count()/1e6:.2f}M params")
print(f"  Param difference    : {(m_rlc.param_count()-m_base.param_count())/1e3:.1f}K  (L,R,C overhead)")
del m_rlc, m_base, cfg_test
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
from friction_llm import FrictionConfig, RLCFrictionLM, BaselineLM, SharpnessCurriculum

def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def circuit_snapshot(base):
    \"\"\"Per-layer ω₀, ζ, underdamped %, sparsity — for live monitoring.\"\"\"
    rows = []
    for block in base.blocks:
        rlc  = block.rlc_block.rlc
        fric = block.rlc_block.friction
        rows.append({
            "omega_0": rlc.omega_0.mean().item(),
            "zeta":    rlc.damping_ratio.mean().item(),
            "underdamped_pct": (rlc.damping_ratio < 1.0).float().mean().item() * 100,
            "mu_s":    fric.mu_s.mean().item(),
        })
    return rows

def train_rlc(model, train_ds, val_ds,
              max_steps=10000, batch_size=16,
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

    physics, other = [], []
    for n, p in base.named_parameters():
        if any(k in n for k in ("raw_mu","raw_ratio","log_L","log_R","log_C")):
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
        "omega_spread": [], "zeta_spread": [],
        "underdamped_layers": [], "sharpness": [],
        "layers": [],
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

            # Sparsity + circuit snapshot (use base model directly, no DataParallel)
            sample, _ = val_ds.get_batch(4, device)
            report  = base.circuit_report(sample)
            sparsity = report["overall_sparsity"]
            snap    = circuit_snapshot(base)
            model.train()

            omegas = [r["omega_0"] for r in snap]
            zetas  = [r["zeta"]    for r in snap]
            omega_spread = max(omegas) - min(omegas)
            zeta_spread  = max(zetas)  - min(zetas)
            underdamped  = sum(1 for z in zetas if z < 1.0)

            dt = (time.time() - t0) / max(step, 1)
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["val_loss"].append(vloss.item())
            history["sparsity"].append(sparsity)
            history["omega_spread"].append(omega_spread)
            history["zeta_spread"].append(zeta_spread)
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
                f"| sparse {sparsity:.1%} | ω₀spread {omega_spread:.4f} "
                f"| underdamped {underdamped}/{len(snap)} "
                f"| sharp {sharpness:.1f} | {dt*1000:.0f}ms/step"
            )

        # ── Checkpoint ───────────────────────────────────────────────────────
        # Periodic: model weights only (~470 MB). best.pt keeps full state for resuming.
        if step % ckpt_every == 0 and step > 0:
            path = f"{ckpt_dir}/rlc_step_{step:06d}.pt"
            torch.save({
                "model":  base.state_dict(),   # weights only — no optimizer, no history
                "config": base.config,
                "step":   step,
            }, path)
            print(f"  → saved {path}  (model only, ~470 MB)")

    return history, base
"""

RLC_TRAIN_CELL = """\
# Medium config: 117M params, 12 layers — right size for WikiText-103
cfg = FrictionConfig.medium()
cfg.max_seq_len     = SEQ_LEN
cfg.use_rlc         = True
cfg.mu_s_init       = 0.05        # charge scale is ~dt²×V, much smaller than raw signal
cfg.rlc_dt          = 0.3         # larger step → more charge per layer
cfg.rlc_filter_mode = "learnable" # each neuron learns its own LP/BP/HP mix
                                   # change to "lowpass" to reproduce original behaviour

model = RLCFrictionLM(cfg).to(device)
print(f"RLCFrictionLM  : {model.param_count()/1e6:.1f}M params")
print(f"Layers         : {cfg.n_layers}  d_model: {cfg.d_model}")
print(f"Filter mode    : {cfg.rlc_filter_mode}")
print(f"μ_s={cfg.mu_s_init}  rlc_dt={cfg.rlc_dt}")
print()
print("Watch: ω₀ spread, underdamped layers, AND filter weights evolving per layer")

history, model = train_rlc(
    model, train_ds, val_ds,
    max_steps=10000,
    batch_size=8,
    lr=3e-4,
    log_every=100,
    ckpt_every=1000,
)
"""

CIRCUIT_REPORT_CELL = """\
import tiktoken
enc = tiktoken.get_encoding("gpt2")

sample_ids = torch.tensor(
    enc.encode_ordinary("The relationship between"),
    dtype=torch.long, device=device
).unsqueeze(0)

model.eval()
model.print_circuit_report(sample_ids)
"""

PHYSICS_PLOT_CELL = """\
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

steps = history["step"]
n_layers = len(history["layers"][0]) if history["layers"] else 0

fig = plt.figure(figsize=(18, 12))
fig.suptitle("RLCFrictionLM — Circuit Physics Emerging During Training",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Loss ─────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, :2])
ax.plot(steps, history["loss"],     label="train", alpha=0.7, linewidth=1.5)
ax.plot(steps, history["val_loss"], label="val",   linewidth=2)
ax.set(xlabel="Step", ylabel="Loss", title="Training & Validation Loss")
ax.legend(); ax.grid(alpha=0.3)

# ── Sparsity ─────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(steps, [s*100 for s in history["sparsity"]], color="steelblue", linewidth=2)
ax2.axhline(70, linestyle="--", color="gray", alpha=0.5, label="CPU target (70%)")
ax2.set(xlabel="Step", ylabel="Sparsity %", title="Gate Sparsity Over Training")
ax2.legend(); ax2.grid(alpha=0.3)

# ── ω₀ spread (key metric: are layers finding different frequencies?) ─────────
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(steps, history["omega_spread"], color="darkorange", linewidth=2)
ax3.set(xlabel="Step", ylabel="max(ω₀) − min(ω₀)",
        title="ω₀ Spread Across Layers  (0 = all same,  > 0 = differentiated)")
ax3.grid(alpha=0.3)

# ── Underdamped layer count ───────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(steps, history["underdamped_layers"], color="crimson", linewidth=2,
         drawstyle="steps-post")
ax4.set(xlabel="Step", ylabel="Layers with ζ < 1",
        title=f"Resonant Layers  (out of {n_layers})")
ax4.set_yticks(range(n_layers + 1)); ax4.grid(alpha=0.3)

# ── Per-layer ω₀ evolution (heatmap over training) ───────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
if history["layers"] and n_layers > 0:
    omega_matrix = np.array([[r["omega_0"] for r in snap]
                              for snap in history["layers"]]).T   # [n_layers, steps]
    im = ax5.imshow(omega_matrix, aspect="auto", cmap="RdYlGn",
                    extent=[steps[0], steps[-1], n_layers-0.5, -0.5])
    plt.colorbar(im, ax=ax5, label="ω₀")
    ax5.set(xlabel="Step", ylabel="Layer", yticks=range(n_layers),
            yticklabels=[f"L{i}" for i in range(n_layers)],
            title="ω₀ per Layer Over Training  (green=higher freq, red=lower freq)")

# ── Per-layer ζ at end of training ───────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
if history["layers"]:
    final_zeta = [r["zeta"] for r in history["layers"][-1]]
    colors = ["crimson" if z < 1.0 else "steelblue" for z in final_zeta]
    ax6.barh(range(n_layers), final_zeta, color=colors)
    ax6.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="ζ=1 (critical)")
    ax6.set(xlabel="Damping ratio ζ", yticks=range(n_layers),
            yticklabels=[f"L{i}" for i in range(n_layers)],
            title="Final ζ per Layer\n(red=resonant, blue=overdamped)")
    ax6.legend(fontsize=8); ax6.grid(alpha=0.3, axis="x")

plt.savefig("rlc_physics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: rlc_physics.png")
"""

SPARSITY_CELL = """\
sample, _ = val_ds.get_batch(8, device)
model.eval()
model.print_circuit_report(sample)
"""

GENERATE_CELL = """\
import tiktoken
enc = tiktoken.get_encoding("gpt2")

prompts = [
    "The theory of",
    "In the year 1900",
    "Scientists discovered that",
]

for prompt in prompts:
    ids = enc.encode_ordinary(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(idx, max_new_tokens=120, temperature=0.8, top_k=40)
    print(f"Prompt: {prompt!r}")
    print(enc.decode(out[0].tolist()))
    print("-" * 60)
"""

RESUME_MD = """\
## Resuming from checkpoint

Checkpoints are saved to `/kaggle/working/checkpoints/` (absolute path, always the same place).

```python
import glob, torch
# Find latest checkpoint
ckpts = sorted(glob.glob("/kaggle/working/checkpoints/rlc_step_*.pt"))
print("Available:", ckpts)

ckpt  = torch.load(ckpts[-1], map_location=device)  # load latest
cfg   = ckpt["config"]
model = RLCFrictionLM(cfg).to(device)
model.load_state_dict(ckpt["model"])
history = ckpt["history"]
print(f"Resumed from step {ckpt['step']}, last val loss {history['val_loss'][-1]:.4f}")

# Continue training from where it left off
history, model = train_rlc(
    model, train_ds, val_ds,
    max_steps=10000,   # will pick up from ckpt["step"] internally
    batch_size=8, lr=3e-4,
    ckpt_dir="/kaggle/working/checkpoints",
)
```
"""


# ─────────────────────────────────────────────────────────────────────────────
# Build Kaggle notebook — RLC deep training
# ─────────────────────────────────────────────────────────────────────────────

def build_kaggle():
    cells = []

    cells.append(md(TITLE_MD))

    cells.append(md("## 1 · Clone repo & install"))
    cells.append(code(CLONE_CELL))

    cells.append(md("## 2 · GPU setup"))
    cells.append(code(SETUP_CELL))

    cells.append(md("## Find existing checkpoints (run this if resuming)"))
    cells.append(code(
        "import subprocess\n"
        "result = subprocess.run(\n"
        "    ['find', '/kaggle/working', '-name', '*.pt', '-not', '-path', '*/.git/*'],\n"
        "    capture_output=True, text=True\n"
        ")\n"
        "files = sorted(result.stdout.strip().split('\\n'))\n"
        "if files and files[0]:\n"
        "    print('Found checkpoints:')\n"
        "    for f in files:\n"
        "        import os\n"
        "        size = os.path.getsize(f) / 1e6\n"
        "        print(f'  {f}  ({size:.0f} MB)')\n"
        "else:\n"
        "    print('No checkpoints found yet.')\n"
    ))

    cells.append(md("## 3 · Verify imports"))
    cells.append(code(IMPORT_CELL))

    cells.append(md("## 4 · Data — WikiText-103"))
    cells.append(code(DATA_CELL))
    cells.append(code(TOKENIZE_CELL))
    cells.append(code(DATALOADER_CELL))

    cells.append(md("## 5 · Training function\n\nLogs ω₀ spread and underdamped layer count every step — watch the physics emerge."))
    cells.append(code(TRAIN_FUNC_CELL))

    cells.append(md(
        "## 6 · Train — Baseline GPT-2 (control experiment)\n\n"
        "Standard transformer: same size, same data, same steps — **no physics**.\n"
        "This is the control. RLC must beat this to prove the architecture works."
    ))
    cells.append(code(
        "cfg_b = FrictionConfig.medium()\n"
        "cfg_b.max_seq_len = SEQ_LEN\n"
        "baseline = BaselineLM(cfg_b).to(device)\n"
        "print(f'BaselineLM: {baseline.param_count()/1e6:.1f}M params  (standard GELU FFN)')\n"
        "\n"
        "# Reuse train_rlc — baseline has no RLC state, circuit_report returns None\n"
        "# so sparsity defaults to 0 and circuit cols are skipped automatically\n"
        "history_base, baseline = train_rlc(\n"
        "    baseline, train_ds, val_ds,\n"
        "    max_steps=10000, batch_size=8, lr=3e-4,\n"
        "    log_every=100, ckpt_every=1000, ckpt_dir='checkpoints/baseline',\n"
        ")\n"
    ))

    cells.append(md(
        "## 7 · Train — RLCFrictionLM (117M params, 10 000 steps)\n\n"
        "Expected time on 2× T4: **~3 hours**.\n"
        "Checkpoints saved every 1 000 steps → session can be resumed.\n\n"
        "Key milestones to watch:\n"
        "- **step ~500**: warmup ends, LR hits peak\n"
        "- **step ~1000**: sharpness curriculum starts annealing\n"
        "- **step ~2000**: ω₀ spread starts growing — layers finding different frequencies\n"
        "- **step ~5000**: resonant vs overdamped pattern solidifies\n"
    ))
    cells.append(code(RLC_TRAIN_CELL))

    cells.append(md(
        "## 8 · Train — PhysicsLM (no attention — pure wave propagation)\n\n"
        "Replaces dot-product attention with a 1D RLC transmission line.\n"
        "Information propagates as waves along the token sequence — O(T) not O(T²)."
    ))
    cells.append(code(
        "from friction_llm import PhysicsLM\n"
        "\n"
        "cfg_p = FrictionConfig.medium()\n"
        "cfg_p.max_seq_len      = SEQ_LEN\n"
        "cfg_p.use_rlc          = True\n"
        "cfg_p.use_coupled_mixer= True\n"
        "cfg_p.mu_s_init        = 0.05\n"
        "cfg_p.rlc_dt           = 0.3\n"
        "cfg_p.rlc_filter_mode  = 'learnable'\n"
        "cfg_p.mixer_L_c_init   = 5.0\n"
        "\n"
        "physics_model = PhysicsLM(cfg_p).to(device)\n"
        "print(f'PhysicsLM      : {physics_model.param_count()/1e6:.1f}M params')\n"
        "print(f'No attention   : wave propagation only  (O(T) vs O(T²))')\n"
        "\n"
        "history_phy, physics_model = train_rlc(\n"
        "    physics_model, train_ds, val_ds,\n"
        "    max_steps=10000, batch_size=8, lr=3e-4,\n"
        "    log_every=100, ckpt_every=1000,\n"
        "    ckpt_dir='/kaggle/working/checkpoints/physics',\n"
        ")\n"
    ))

    cells.append(md("## 9 · Head-to-head: Baseline vs RLC vs PhysicsLM\n\nThe definitive test — same params, same data, same steps. Who wins?"))
    cells.append(code(
        "import matplotlib.pyplot as plt\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 4))\n"
        "fig.suptitle('3-way: Baseline (GELU) vs RLCFrictionLM vs PhysicsLM (no attention)',\n"
        "             fontweight='bold')\n"
        "\n"
        "COLORS = {'Baseline':'gray', 'RLC':'steelblue', 'Physics':'darkorange'}\n"
        "runs = [\n"
        "    ('Baseline', history_base, 'gray'),\n"
        "    ('RLC+Friction (attn)', history,     'steelblue'),\n"
        "    ('PhysicsLM (no attn)', history_phy, 'darkorange'),\n"
        "]\n"
        "\n"
        "ax = axes[0]\n"
        "for label, hist, c in runs:\n"
        "    ax.plot(hist['step'], hist['val_loss'], label=label, color=c, lw=2)\n"
        "ax.set(xlabel='Step', ylabel='Val Loss', title='Validation Loss  ← lower wins')\n"
        "ax.legend(fontsize=8); ax.grid(alpha=0.3)\n"
        "\n"
        "ax = axes[1]\n"
        "for label, hist, c in runs:\n"
        "    ax.plot(hist['step'], hist['loss'], color=c, alpha=0.6, lw=1.5, label=label)\n"
        "ax.set(xlabel='Step', ylabel='Train Loss', title='Training Loss')\n"
        "ax.legend(fontsize=8); ax.grid(alpha=0.3)\n"
        "\n"
        "ax = axes[2]\n"
        "ax.axhline(0, color='gray', lw=2, label='Baseline (0%)')\n"
        "ax.plot(history['step'],     [s*100 for s in history['sparsity']],\n"
        "        color='steelblue', lw=2, label='RLC')\n"
        "ax.plot(history_phy['step'], [s*100 for s in history_phy['sparsity']],\n"
        "        color='darkorange', lw=2, label='PhysicsLM')\n"
        "ax.axhline(70, color='green', lw=1, linestyle='--', label='CPU target')\n"
        "ax.set(xlabel='Step', ylabel='Sparsity %', title='Gate Sparsity', ylim=[-5,105])\n"
        "ax.legend(fontsize=8); ax.grid(alpha=0.3)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('3way_comparison.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "\n"
        "print('\\n── Final val loss ───────────────────────')\n"
        "results = {label: hist['val_loss'][-1] for label, hist, _ in runs}\n"
        "best = min(results, key=results.get)\n"
        "for label, val in sorted(results.items(), key=lambda x: x[1]):\n"
        "    marker = ' ← WINNER' if label == best else ''\n"
        "    print(f'  {label:<30} {val:.4f}{marker}')\n"
        "print(f'\\nRLC sparsity    : {history[\"sparsity\"][-1]:.1%}')\n"
        "print(f'Physics sparsity: {history_phy[\"sparsity\"][-1]:.1%}')\n"
    ))

    cells.append(md("## 9 · Circuit physics report"))
    cells.append(code(CIRCUIT_REPORT_CELL))

    cells.append(md(
        "## 10 · Filter weights per layer\n\n"
        "What filter type did each layer learn? "
        "LP=low-pass (slow/global), BP=band-pass (resonant/selective), HP=high-pass (fast/local)."
    ))
    cells.append(code(
        "model.eval()\n"
        "sample, _ = val_ds.get_batch(4, device)\n"
        "report = model.circuit_report(sample)\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "\n"
        "n_layers = model.config.n_layers\n"
        "lp, bp, hp = [], [], []\n"
        "for i in range(n_layers):\n"
        "    fw = report[f'layer_{i}']['filter_weights']\n"
        "    lp.append(fw.get('lowpass',  0))\n"
        "    bp.append(fw.get('bandpass', 0))\n"
        "    hp.append(fw.get('highpass', 0))\n"
        "\n"
        "x = np.arange(n_layers)\n"
        "fig, ax = plt.subplots(figsize=(12, 4))\n"
        "ax.bar(x, lp, label='Low-pass  (slow/global)',      color='#4878cf')\n"
        "ax.bar(x, bp, bottom=lp, label='Band-pass (resonant)',  color='#6acc65')\n"
        "ax.bar(x, hp, bottom=[l+b for l,b in zip(lp,bp)],\n"
        "       label='High-pass (fast/local)', color='#d65f5f')\n"
        "ax.set(xlabel='Layer', ylabel='Filter weight %', ylim=[0,100],\n"
        "       title='Learned Filter Type per Layer  (emerged from training)',\n"
        "       xticks=x, xticklabels=[f'L{i}' for i in range(n_layers)])\n"
        "ax.legend(); ax.grid(alpha=0.3, axis='y')\n"
        "plt.tight_layout()\n"
        "plt.savefig('filter_weights.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "\n"
        "print('\\nFilter composition per layer:')\n"
        "for i in range(n_layers):\n"
        "    fw = report[f'layer_{i}']['filter_weights']\n"
        "    dominant = max(fw, key=fw.get)\n"
        "    print(f'  Layer {i:2d}: LP={lp[i]:5.1f}%  BP={bp[i]:5.1f}%  '\n"
        "          f'HP={hp[i]:5.1f}%  → {dominant}')\n"
    ))

    cells.append(md("## 11 · Circuit evolution plots"))
    cells.append(code(PHYSICS_PLOT_CELL))

    cells.append(md("## 11 · Sparsity detail"))
    cells.append(code(SPARSITY_CELL))

    cells.append(md("## 12 · Text generation"))
    cells.append(code(GENERATE_CELL))

    cells.append(md(RESUME_MD))

    return notebook(cells, accelerator="GPU")


# ─────────────────────────────────────────────────────────────────────────────
# Build JupyterHub notebook
# ─────────────────────────────────────────────────────────────────────────────

def build_jupyterhub():
    cells = []
    cells.append(md(TITLE_MD))
    cells.append(md("## Setup\n\nRun from repo root. Requires: `torch tiktoken datasets`"))
    cells.append(code(
        "import sys; sys.path.insert(0, '.')\n" + SETUP_CELL
    ))
    cells.append(code(IMPORT_CELL))
    cells.append(md("## Data"))
    cells.append(code(DATA_CELL))
    cells.append(code(TOKENIZE_CELL))
    cells.append(code(DATALOADER_CELL))
    cells.append(md("## Training"))
    cells.append(code(TRAIN_FUNC_CELL))
    cells.append(md("## Train RLCFrictionLM"))
    cells.append(code(RLC_TRAIN_CELL))
    cells.append(md("## Circuit report"))
    cells.append(code(CIRCUIT_REPORT_CELL))
    cells.append(md("## Visualise"))
    cells.append(code(PHYSICS_PLOT_CELL))
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
