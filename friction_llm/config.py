from dataclasses import dataclass, field


@dataclass
class FrictionConfig:
    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab_size: int = 50257      # GPT-2 tiktoken vocab

    # ── Architecture ─────────────────────────────────────────────────────────
    max_seq_len: int = 1024
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048             # FFN hidden dim (4 × d_model recommended)
    dropout: float = 0.1
    bias: bool = True

    # ── Friction Gate ─────────────────────────────────────────────────────────
    # mu_s : static threshold  — force needed to START moving
    # mu_k : kinetic drag      — energy lost once moving  (always < mu_s)
    mu_s_init: float = 0.5
    mu_k_ratio_init: float = 0.6     # mu_k = mu_s * ratio,  ratio ∈ (0, 1)

    # Surrogate gradient sharpness (annealed from init → max during training)
    sharpness_init: float = 3.0
    sharpness_max: float = 50.0

    # ── Cross-Layer Momentum ─────────────────────────────────────────────────
    # Neurons "already in motion" in layer L have lower effective mu_s in L+1.
    use_momentum: bool = True
    momentum_alpha: float = 0.9      # exponential decay of momentum
    momentum_beta: float = 0.3       # how much momentum reduces mu_s (0–1)

    # ── RLC Circuit Neuron ───────────────────────────────────────────────────
    # Full RLC model: each neuron is an inductor (L) + resistor (R) + capacitor (C)
    # L : inductance  — inertia, resists changes in current
    # R : resistance  — our friction (dissipates energy)
    # C : capacitance — stores charge, creates restoring force
    # Natural frequency ω₀ = 1/√(LC), damping ratio ζ = R/(2√(L/C))
    use_rlc: bool = False               # switch from FGLU to RLC block
    rlc_dt: float = 0.1                 # Euler integration timestep
    rlc_L_init: float = 1.0             # initial inductance (per neuron)
    rlc_R_init: float = 2.0             # initial resistance — 2.0 = critical damping when L=C=1
    rlc_C_init: float = 1.0             # initial capacitance (per neuron)
    rlc_clamp: float = 10.0             # hard clamp on (q,i) to prevent blowup
    rlc_filter_mode: str = "lowpass"    # lowpass | highpass | bandpass | notch | learnable

    # ── CoulombLM — electric force attention ────────────────────────────────
    coulomb_r_power: float = 2.0        # distance exponent (2 = Coulomb, 1 = linear)

    # ── PhysicsLM — Coupled Oscillator Mixer (replaces attention) ────────────
    use_coupled_mixer: bool = False     # True = PhysicsLM (no attention)
    mixer_L_c_init: float = 5.0        # coupling inductance between adjacent tokens
                                        # larger = weaker coupling at init

    # ── Friction Attention (Phase 2 — off by default) ─────────────────────────
    use_friction_attention: bool = False
    attn_mu_s_init: float = 0.1
    attn_mu_k_ratio_init: float = 0.5

    # ── Training ─────────────────────────────────────────────────────────────
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    sparsity_reg: float = 0.0        # L1 penalty on gate activations

    # ── Curriculum: anneal sharpness smoothly so hard threshold emerges ───────
    curriculum_warmup_steps: int = 1000
    curriculum_anneal_steps: int = 10000

    # ── Hardware ─────────────────────────────────────────────────────────────
    use_amp: bool = True             # mixed precision (FP16 on CUDA)
    compile_model: bool = False      # torch.compile (PyTorch 2.x)

    # ── Checkpointing / Logging ───────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000
    log_every: int = 100
    eval_every: int = 500
    eval_iters: int = 50

    # ── Quick-start presets ───────────────────────────────────────────────────
    @classmethod
    def tiny(cls) -> "FrictionConfig":
        """~1 M params — smoke test only, verifies code runs in seconds."""
        return cls(d_model=128, n_heads=4, n_layers=2, d_ff=512,
                   log_every=1, eval_every=9999, save_every=9999)

    @classmethod
    def small(cls) -> "FrictionConfig":
        """~30 M params — fast experiments on a single GPU or strong CPU."""
        return cls(d_model=384, n_heads=6, n_layers=6, d_ff=1536)

    @classmethod
    def medium(cls) -> "FrictionConfig":
        """~117 M params — GPT-2 scale, fits on RTX A6000 (48 GB)."""
        return cls(d_model=768, n_heads=12, n_layers=12, d_ff=3072)

    @classmethod
    def large(cls) -> "FrictionConfig":
        """~350 M params — training-scale experiment."""
        return cls(d_model=1024, n_heads=16, n_layers=24, d_ff=4096)
