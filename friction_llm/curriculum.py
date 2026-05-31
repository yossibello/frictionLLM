"""
SharpnessCurriculum — anneals the surrogate gradient sharpness over training.

Why this matters
────────────────
At low sharpness the friction gate is smooth (≈ soft-shrinkage).
The network trains easily but doesn't produce real sparsity.
As sharpness increases, the sigmoid ramp steepens toward a step function —
the "slip jolt" physics emerge, neurons snap to fully on/off, and true
sparsity grows.  By the end of curriculum_anneal_steps, inference can run
the hard-threshold path and get maximum sparsity for free.

Schedule: flat warmup → cosine anneal from sharpness_init to sharpness_max.
"""

import math

import torch.nn as nn

from .friction_gate import FrictionGate


class SharpnessCurriculum:
    """
    Controls how sharply the surrogate gradient approximates the step function.

    Usage
    ─────
        curriculum = SharpnessCurriculum(model, config)
        for step, batch in enumerate(dataloader):
            ...
            loss.backward()
            optimizer.step()
            sharpness = curriculum.step()   # call once per optimizer step
    """

    def __init__(self, model: nn.Module, config) -> None:
        self.model          = model
        self.sharpness_init = config.sharpness_init
        self.sharpness_max  = config.sharpness_max
        self.warmup_steps   = config.curriculum_warmup_steps
        self.anneal_steps   = config.curriculum_anneal_steps
        self._step          = 0

    def step(self) -> float:
        """Advance one optimizer step. Returns current sharpness."""
        self._step += 1
        sharpness = self._compute_sharpness(self._step)
        self._apply(sharpness)
        return sharpness

    def _compute_sharpness(self, step: int) -> float:
        if step <= self.warmup_steps:
            return self.sharpness_init
        progress = min((step - self.warmup_steps) / max(self.anneal_steps, 1), 1.0)
        # Cosine anneal: slow start, fast middle, slow end
        cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.sharpness_init + (self.sharpness_max - self.sharpness_init) * cosine

    def _apply(self, sharpness: float) -> None:
        for module in self.model.modules():
            if isinstance(module, FrictionGate):
                module.sharpness = sharpness

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, state: dict) -> None:
        self._step = state["step"]
        self._apply(self._compute_sharpness(self._step))
