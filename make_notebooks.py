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

def writefile_cell(path, filepath):
    """Cell that writes a source file using %%writefile magic."""
    with open(path, "r") as f:
        body = f.read()
    return code(f"%%writefile {filepath}\n" + body)

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
# Source files to bundle
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_FILES = [
    ("friction_llm/config.py",       "friction_llm/config.py"),
    ("friction_llm/friction_gate.py","friction_llm/friction_gate.py"),
    ("friction_llm/attention.py",    "friction_llm/attention.py"),
    ("friction_llm/block.py",        "friction_llm/block.py"),
    ("friction_llm/model.py",        "friction_llm/model.py"),
    ("friction_llm/curriculum.py",   "friction_llm/curriculum.py"),
    ("friction_llm/rlc_neuron.py",   "friction_llm/rlc_neuron.py"),
    ("friction_llm/rlc_block.py",    "friction_llm/rlc_block.py"),
    ("friction_llm/rlc_model.py",    "friction_llm/rlc_model.py"),
    ("friction_llm/__init__.py",     "friction_llm/__init__.py"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared cell blocks
# ─────────────────────────────────────────────────────────────────────────────

TITLE_MD = """# FrictionLLM + RLCFrictionLM
### A neural architecture based on physical friction and RLC circuit mechanics

**Core idea:**
Current LLMs are pure resistors — every weight fires for every token.
This architecture maps real physics onto neurons:

| Physics | Component | Effect in the network |
|---|---|---|
| Static friction  | μ_s threshold  | Neuron stays **stuck at 0** unless signal exceeds threshold |
| Kinetic friction | μ_k drag       | Once fired, output = z − sign(z)·μ_k  (energy loss) |
| Inductance (L)   | Inertia        | Resists rapid changes in activation |
| Capacitance (C)  | Charge storage | Accumulates signal across **layers** before firing |
| Resonance        | ω₀ = 1/√(LC)  | Each neuron tunes to a natural frequency |

**What emerges:** sparsity (CPU-friendly inference), hierarchical feature tuning, and physically-principled dynamics.
"""

SETUP_COMMON = """\
import os, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cpu")
)
n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
print(f"Device: {device}")
if device.type == "cuda":
    for i in range(n_gpus):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB")
    print(f"Total GPUs: {n_gpus} — {'DataParallel enabled' if n_gpus > 1 else 'single GPU'}")
"""

DATA_CELL = """\
import os
os.makedirs("data", exist_ok=True)

# WikiText-103: 103M tokens — right-sized for a 33M param model.
# Falls back to TinyShakespeare if datasets not available.
try:
    from datasets import load_dataset
    print("Loading WikiText-103 (~103M tokens)...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", trust_remote_code=True)
    # Concatenate all splits into one text file
    with open("data/input.txt", "w", encoding="utf-8") as f:
        for split in ["train", "validation", "test"]:
            for row in ds[split]:
                text = row["text"].strip()
                if text:
                    f.write(text + "\\n")
    print(f"WikiText-103 saved: {os.path.getsize('data/input.txt')/1e6:.0f} MB")
except Exception as e:
    print(f"datasets not available ({e}), falling back to TinyShakespeare")
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "data/input.txt")
    print(f"TinyShakespeare: {os.path.getsize('data/input.txt')/1e6:.1f} MB — consider upgrading to WikiText-103")
"""

TOKENIZE_CELL = """\
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")
with open("data/input.txt") as f:
    text = f.read()

tokens = np.array(enc.encode_ordinary(text), dtype=np.uint16)
split  = int(0.9 * len(tokens))
tokens[:split].tofile("data/train.bin")
tokens[split:].tofile("data/val.bin")
print(f"Train: {split:,} tokens   Val: {len(tokens)-split:,} tokens")
"""

DATALOADER_CELL = """\
class TokenDataset:
    def __init__(self, path, seq_len):
        self.data    = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def get_batch(self, batch_size, device):
        ix = torch.randint(len(self.data) - self.seq_len - 1, (batch_size,))
        x  = torch.stack([torch.from_numpy(self.data[i:i+self.seq_len].astype(np.int64)) for i in ix])
        y  = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.seq_len].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

train_ds = TokenDataset("data/train.bin", seq_len=256)
val_ds   = TokenDataset("data/val.bin",   seq_len=256)
print(f"Batches available: {len(train_ds):,}")
"""

TRAIN_FUNC_CELL = """\
from friction_llm import FrictionConfig, FrictionLM, RLCFrictionLM, SharpnessCurriculum

def unwrap(model):
    \"\"\"Unwrap DataParallel to access base model methods.\"\"\"""
    return model.module if isinstance(model, nn.DataParallel) else model

def train_model(model, train_ds, val_ds, max_steps=600, batch_size=16,
                lr=3e-4, log_every=50, label="model"):
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Scale batch size across all GPUs
    effective_batch = batch_size * max(n_gpus, 1)

    # Wrap in DataParallel if multiple GPUs available
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"[{label}] DataParallel across {n_gpus} GPUs — effective batch {effective_batch}")

    base = unwrap(model)   # always use base model for config / diagnostics

    # Separate physics params (no weight decay)
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
    history = {"step": [], "loss": [], "val_loss": [], "sparsity": []}
    t0 = time.time()

    lr_min = lr / 10
    for step in range(max_steps + 1):
        # Linear warmup then cosine decay
        warmup = min(500, max_steps // 10)
        if step < warmup:
            cur_lr = lr * step / warmup
        else:
            progress = (step - warmup) / (max_steps - warmup)
            cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        x, y = train_ds.get_batch(effective_batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()   # DataParallel returns per-GPU losses

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        curriculum.step()

        if step % log_every == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = val_ds.get_batch(effective_batch, device)
                _, vloss = model(xv, yv)
                if isinstance(vloss, torch.Tensor) and vloss.dim() > 0:
                    vloss = vloss.mean()

            sample, _ = val_ds.get_batch(4, device)
            if hasattr(base, "circuit_report"):
                report   = base.circuit_report(sample)
                sparsity = report["overall_sparsity"]
            else:
                report   = base.measure_sparsity(sample)
                sparsity = report["overall"]
            model.train()

            dt = (time.time() - t0) / max(step, 1)
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["val_loss"].append(vloss.item())
            history["sparsity"].append(sparsity)
            print(f"[{label}] step {step:4d} | loss {loss.item():.4f} "
                  f"| val {vloss.item():.4f} | sparsity {sparsity:.1%} "
                  f"| {dt*1000:.0f}ms/step")

    # Always return the unwrapped base model for diagnostics
    return history, base
"""

FRICTION_TRAIN_CELL = """\
cfg_f = FrictionConfig.small()
cfg_f.max_seq_len = 256
model_f = FrictionLM(cfg_f).to(device)
print(f"FrictionLM: {model_f.param_count()/1e6:.1f}M params")

history_f, model_f = train_model(model_f, train_ds, val_ds,
                                  max_steps=5000, batch_size=16, label="FrictionLM")
"""

RLC_TRAIN_CELL = """\
cfg_r = FrictionConfig.small()
cfg_r.max_seq_len = 256
cfg_r.use_rlc     = True
# RLC charge q ≈ dt²×V ≈ 0.01×V — much smaller than raw gate signal.
# Lower μ_s so charge can actually break through the threshold.
cfg_r.mu_s_init   = 0.05   # was 0.5 — charge lives in 0.01–0.1 range
cfg_r.rlc_dt      = 0.3    # larger step → more charge per layer (was 0.1)
model_r = RLCFrictionLM(cfg_r).to(device)
print(f"RLCFrictionLM: {model_r.param_count()/1e6:.1f}M params")
print(f"μ_s={cfg_r.mu_s_init}  dt={cfg_r.rlc_dt}  (tuned for charge scale)")

history_r, model_r = train_model(model_r, train_ds, val_ds,
                                  max_steps=5000, batch_size=16, label="RLCFrictionLM")
"""

CIRCUIT_REPORT_CELL = """\
import tiktoken
enc = tiktoken.get_encoding("gpt2")
sample_text = "To be or not to be, that is the question"
sample_ids  = torch.tensor(enc.encode_ordinary(sample_text),
                           dtype=torch.long, device=device).unsqueeze(0)

model_r.eval()
model_r.print_circuit_report(sample_ids)
"""

PLOT_CELL = """\
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 10))
fig.suptitle("FrictionLLM vs RLCFrictionLM — Training Dynamics", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── Loss curves ───────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(history_f["step"], history_f["val_loss"],  label="FrictionLM val",  color="#e05a5a", linewidth=2)
ax1.plot(history_r["step"], history_r["val_loss"],  label="RLCFrictionLM val", color="#5a9ce0", linewidth=2)
ax1.plot(history_f["step"], history_f["loss"],  linestyle="--", alpha=0.4, color="#e05a5a")
ax1.plot(history_r["step"], history_r["loss"],  linestyle="--", alpha=0.4, color="#5a9ce0")
ax1.set_xlabel("Step"); ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Training & Validation Loss"); ax1.legend(); ax1.grid(alpha=0.3)

# ── Sparsity ──────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(history_f["step"], [s*100 for s in history_f["sparsity"]], color="#e05a5a", linewidth=2, label="FrictionLM")
ax2.plot(history_r["step"], [s*100 for s in history_r["sparsity"]], color="#5a9ce0", linewidth=2, label="RLCFrictionLM")
ax2.axhline(70, linestyle="--", color="gray", alpha=0.5, label="CPU-efficient threshold")
ax2.set_xlabel("Step"); ax2.set_ylabel("Sparsity (%)")
ax2.set_title("Gate Sparsity (higher = more neurons silent)"); ax2.legend(); ax2.grid(alpha=0.3)

# ── RLC natural frequencies per layer ────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
model_r.eval()
omega_per_layer, zeta_per_layer = [], []
for i, block in enumerate(model_r.blocks):
    rlc = block.rlc_block.rlc
    omega_per_layer.append(rlc.omega_0.detach().cpu().numpy())
    zeta_per_layer.append(rlc.damping_ratio.detach().cpu().numpy())

colors = plt.cm.viridis([i / len(omega_per_layer) for i in range(len(omega_per_layer))])
for i, (w, c) in enumerate(zip(omega_per_layer, colors)):
    ax3.hist(w, bins=30, alpha=0.6, color=c, label=f"Layer {i}")
ax3.set_xlabel("Natural Frequency ω₀"); ax3.set_ylabel("Count")
ax3.set_title("ω₀ Distribution per Layer\\n(divergence = different frequency bands)"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

# ── Damping ratio per layer ───────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
for i, (z, c) in enumerate(zip(zeta_per_layer, colors)):
    ax4.hist(z, bins=30, alpha=0.6, color=c, label=f"Layer {i}")
ax4.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Critical damping (ζ=1)")
ax4.set_xlabel("Damping Ratio ζ"); ax4.set_ylabel("Count")
ax4.set_title("Damping Ratio per Layer\\n(<1=resonant, 1=critical, >1=overdamped)"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

# ── μ_s per layer (friction thresholds) ──────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
mu_s_per_layer = []
for block in model_r.blocks:
    mu_s_per_layer.append(block.rlc_block.friction.mu_s.detach().cpu().numpy())
ax5.boxplot(mu_s_per_layer, labels=[f"L{i}" for i in range(len(mu_s_per_layer))])
ax5.set_xlabel("Layer"); ax5.set_ylabel("μ_s value")
ax5.set_title("Static Friction Thresholds μ_s\\n(higher = harder to activate)"); ax5.grid(alpha=0.3)

plt.savefig("friction_rlc_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to friction_rlc_analysis.png")
"""

GENERATE_CELL = """\
import tiktoken
enc = tiktoken.get_encoding("gpt2")

def generate(model, prompt, max_tokens=150, temperature=0.8, top_k=40):
    model.eval()
    ids = enc.encode_ordinary(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(idx, max_new_tokens=max_tokens,
                         temperature=temperature, top_k=top_k)
    return enc.decode(out[0].tolist())

prompt = "HAMLET:\\nTo be or not to be"

print("=" * 60)
print("FrictionLM output:")
print("=" * 60)
print(generate(model_f, prompt))

print()
print("=" * 60)
print("RLCFrictionLM output:")
print("=" * 60)
print(generate(model_r, prompt))
"""

SPARSITY_ANALYSIS_CELL = """\
# Per-layer sparsity breakdown for both models
model_f.eval(); model_r.eval()
sample, _ = val_ds.get_batch(8, device)

print("FrictionLM — per-layer sparsity:")
stats_f = model_f.measure_sparsity(sample)
for k, v in stats_f.items():
    if k == "overall":
        print(f"  OVERALL: {v:.1%}")
    else:
        print(f"  {k}: {v['sparsity']:.1%}  μ_s={v['mu_s']:.3f}  μ_k={v['mu_k']:.3f}")

print()
print("RLCFrictionLM — per-layer circuit stats:")
model_r.print_circuit_report(sample)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Build Kaggle notebook (fully self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def build_kaggle():
    cells = []

    cells.append(md(TITLE_MD))

    cells.append(md("## 1 · Install dependencies & clone repo"))
    cells.append(code(
        "!pip install tiktoken --quiet\n"
        "\n"
        "import os, subprocess\n"
        "if not os.path.exists('frictionLLM'):\n"
        "    subprocess.run(['git', 'clone', 'https://github.com/yossibello/frictionLLM.git'], check=True)\n"
        "    print('Cloned frictionLLM')\n"
        "else:\n"
        "    subprocess.run(['git', '-C', 'frictionLLM', 'pull'], check=True)\n"
        "    print('Updated frictionLLM')\n"
        "\n"
        "import sys\n"
        "sys.path.insert(0, 'frictionLLM')\n"
        "os.chdir('frictionLLM')\n"
        "print('Working dir:', os.getcwd())"
    ))

    cells.append(md("## 2 · GPU setup"))
    cells.append(code(SETUP_COMMON))

    cells.append(md("## 3 · Verify imports"))
    cells.append(code(
        "from friction_llm import (\n"
        "    FrictionConfig, FrictionLM, RLCFrictionLM,\n"
        "    SharpnessCurriculum, RLCNeuron\n"
        ")\n"
        "print('All imports OK')\n"
        "cfg_test = FrictionConfig.tiny()\n"
        "m_test = RLCFrictionLM(cfg_test)\n"
        "print(f'Tiny RLC model: {m_test.param_count()/1e6:.1f}M params')"
    ))

    cells.append(md("## 4 · Download & tokenise data\n\nUsing TinyShakespeare (~1 MB) — swap in any .txt corpus."))
    cells.append(code(DATA_CELL))
    cells.append(code(TOKENIZE_CELL))
    cells.append(code(DATALOADER_CELL))

    cells.append(md("## 5 · Training function"))
    cells.append(code(TRAIN_FUNC_CELL))

    cells.append(md(
        "## 6 · Train FrictionLM  (R only — static + kinetic friction)\n\n"
        "The surrogate sharpness anneals from 3 → 50 over training.\n"
        "Watch **sparsity** grow as the curriculum hardens the gate."
    ))
    cells.append(code(FRICTION_TRAIN_CELL))

    cells.append(md(
        "## 7 · Train RLCFrictionLM  (full L + R + C circuit)\n\n"
        "Same architecture but the FFN is now a full RLC circuit neuron.\n"
        "Each neuron has its own inductance L, resistance R, and capacitance C.\n"
        "Charge accumulates across layers — the circuit fires when charge > μ_s."
    ))
    cells.append(code(RLC_TRAIN_CELL))

    cells.append(md("## 8 · Circuit physics report\n\nPer-layer ω₀ (natural frequency) and ζ (damping ratio) after training."))
    cells.append(code(CIRCUIT_REPORT_CELL))

    cells.append(md("## 9 · Sparsity analysis"))
    cells.append(code(SPARSITY_ANALYSIS_CELL))

    cells.append(md("## 10 · Visualisations"))
    cells.append(code(PLOT_CELL))

    cells.append(md("## 11 · Text generation"))
    cells.append(code(GENERATE_CELL))

    cells.append(md(
        "## What to try next\n\n"
        "- **More steps**: `max_steps=20000` on an A6000 will get loss below 3.0\n"
        "- **Friction attention**: set `config.use_friction_attention=True` to gate attention logits\n"
        "- **Sparsity reg**: set `config.sparsity_reg=0.01` to push sparsity toward 80%+\n"
        "- **Large corpus**: swap TinyShakespeare for WikiText-103 or OpenWebText\n"
        "- **Ablation**: compare FrictionLM vs RLCFrictionLM at the same param count\n"
    ))

    return notebook(cells, accelerator="GPU")


# ─────────────────────────────────────────────────────────────────────────────
# Build JupyterHub notebook (assumes local repo, friction_llm installed)
# ─────────────────────────────────────────────────────────────────────────────

def build_jupyterhub():
    cells = []

    cells.append(md(TITLE_MD))

    cells.append(md("## Setup\n\nRun from the repo root after `pip install torch tiktoken numpy tqdm`."))
    cells.append(code(SETUP_COMMON))

    cells.append(md("## Verify package"))
    cells.append(code(
        "import sys\n"
        "sys.path.insert(0, '.')   # repo root\n"
        "\n"
        "from friction_llm import (\n"
        "    FrictionConfig, FrictionLM, RLCFrictionLM,\n"
        "    SharpnessCurriculum, RLCNeuron\n"
        ")\n"
        "print('friction_llm loaded OK')\n"
    ))

    cells.append(md("## Prepare data\n\nOnly needed once. Points at `data/input.txt`."))
    cells.append(code(DATA_CELL))
    cells.append(code(TOKENIZE_CELL))
    cells.append(code(DATALOADER_CELL))

    cells.append(md("## Training"))
    cells.append(code(TRAIN_FUNC_CELL))

    cells.append(md("### Train FrictionLM (R only)"))
    cells.append(code(FRICTION_TRAIN_CELL))

    cells.append(md("### Train RLCFrictionLM (full L + R + C)"))
    cells.append(code(RLC_TRAIN_CELL))

    cells.append(md("## Circuit physics report"))
    cells.append(code(CIRCUIT_REPORT_CELL))

    cells.append(md("## Sparsity analysis"))
    cells.append(code(SPARSITY_ANALYSIS_CELL))

    cells.append(md("## Visualisations"))
    cells.append(code(PLOT_CELL))

    cells.append(md("## Text generation"))
    cells.append(code(GENERATE_CELL))

    cells.append(md(
        "## Physics concepts demonstrated\n\n"
        "| Concept | Where it shows up | How to observe |\n"
        "|---|---|---|\n"
        "| Static friction | Sparsity % | Most neurons stay at 0 |\n"
        "| Kinetic drag | Output amplitude | Active neurons lose μ_k energy |\n"
        "| Jolt | μ_s − μ_k gap | Neurons jump, don't ramp |\n"
        "| Capacitance | Charge accumulation | q builds across layers |\n"
        "| Resonance | ω₀ divergence | Layers evolve different frequencies |\n"
        "| Damping modes | ζ histogram | Some neurons go underdamped (resonant) |\n"
    ))

    return notebook(cells, accelerator=None)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)

    kaggle_path = os.path.join(ROOT, "notebooks", "friction_llm_kaggle.ipynb")
    jupyterhub_path = os.path.join(ROOT, "notebooks", "friction_llm_jupyterhub.ipynb")

    print("Building notebooks...")
    write_nb(build_kaggle(),     kaggle_path)
    write_nb(build_jupyterhub(), jupyterhub_path)
    print("Done.")
