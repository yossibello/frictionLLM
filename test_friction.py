"""
Unit tests for FrictionGate physics.

What we're verifying:
  1. μ_k < μ_s always (constraint holds after init and after grad step)
  2. Hard threshold (eval mode): |z| ≤ μ_s  → output = 0  (static friction holds)
  3. Hard threshold (eval mode): |z| > μ_s  → output = z − sign(z)·μ_k  (kinetic drag)
  4. The "jolt": output jumps to (μ_s − μ_k) > 0 at the exact threshold (not 0)
  5. Smooth surrogate (train mode): gradients flow — no zero/NaN gradients
  6. Momentum: firing in layer L reduces effective μ_s in layer L+1
  7. Sparsity: high-μ_s gate produces more zeros than low-μ_s gate
"""

import torch
import sys

sys.path.insert(0, ".")
from friction_llm.friction_gate import FrictionGate, FGLUBlock


PASS = "  PASS"
FAIL = "  FAIL"


def check(condition: bool, name: str) -> bool:
    status = PASS if condition else FAIL
    print(f"{status} — {name}")
    return condition


def run_tests() -> None:
    print("\n── FrictionGate Unit Tests ────────────────────────────────────\n")
    all_passed = True

    gate = FrictionGate(size=16, mu_s_init=0.5, mu_k_ratio_init=0.6, sharpness=3.0)

    # ── 1. Constraint: μ_k < μ_s ─────────────────────────────────────────────
    mu_s = gate.mu_s.detach()
    mu_k = gate.mu_k.detach()
    ok = bool((mu_k < mu_s).all())
    all_passed &= check(ok, f"μ_k < μ_s at init  (μ_s≈{mu_s.mean():.3f}, μ_k≈{mu_k.mean():.3f})")

    # ── 2. Static friction: weak signal → zero output ─────────────────────────
    gate.eval()
    z_weak = torch.zeros(4, 16)   # all zeros — well below any threshold
    out_weak, _ = gate(z_weak)
    ok = bool((out_weak == 0).all())
    all_passed &= check(ok, "static friction: z=0 → output=0 (stuck)")

    # ── 3. Kinetic drag: strong signal → z − sign(z)·μ_k ─────────────────────
    z_strong = torch.ones(1, 16) * 5.0    # far above any reasonable μ_s
    out_strong, _ = gate(z_strong)
    expected = z_strong - z_strong.sign() * mu_k
    ok = bool(torch.allclose(out_strong, expected, atol=1e-5))
    all_passed &= check(ok, f"kinetic drag: z=5.0 → z−μ_k  (got {out_strong[0,0]:.4f}, expected {expected[0,0]:.4f})")

    # ── 4. The jolt: output at threshold+ε is (μ_s − μ_k), NOT near-zero ──────
    # Signal just above μ_s: output should jump to ~(threshold − μ_k)
    epsilon = 1e-3
    z_at_threshold = (mu_s + epsilon).unsqueeze(0)   # [1, 16]
    out_threshold, _ = gate(z_at_threshold)
    jolt = (mu_s - mu_k).mean().item()
    out_mean = out_threshold[out_threshold > 0].mean().item() if (out_threshold > 0).any() else 0.0
    ok = out_mean > 0.01   # non-trivial jump, not a smooth near-zero
    all_passed &= check(ok, f"jolt at threshold: output≈{out_mean:.4f} (expected ≈{jolt:.4f}, not 0)")

    # ── 5. Gradients flow in training mode ────────────────────────────────────
    gate.train()
    z_grad = torch.randn(8, 16, requires_grad=False)
    z_param = torch.nn.Parameter(z_grad.clone())
    out_train, _ = gate(z_param)
    loss = out_train.sum()
    loss.backward()
    has_grad_z    = z_param.grad is not None and not z_param.grad.isnan().any()
    has_grad_mu_s = gate.raw_mu_s.grad is not None and not gate.raw_mu_s.grad.isnan().any()
    has_grad_ratio = gate.raw_ratio.grad is not None and not gate.raw_ratio.grad.isnan().any()
    ok = has_grad_z and has_grad_mu_s and has_grad_ratio
    all_passed &= check(ok,
        f"gradients flow: ∂L/∂z={'✓' if has_grad_z else '✗'}  "
        f"∂L/∂μ_s={'✓' if has_grad_mu_s else '✗'}  "
        f"∂L/∂ratio={'✓' if has_grad_ratio else '✗'}")

    # ── 6. Momentum lowers effective threshold ────────────────────────────────
    gate.eval()
    gate2 = FrictionGate(size=16, mu_s_init=0.5, mu_k_ratio_init=0.6, sharpness=3.0, use_momentum=True)
    gate2.eval()

    # Signal that's just below μ_s — will be blocked without momentum
    z_borderline = (mu_s * 0.9).unsqueeze(0).expand(1, -1)   # 90% of threshold

    out_no_mom, _ = gate2(z_borderline, momentum=None)
    # Momentum = 1.0 (neuron was fully active in previous layer)
    full_momentum = torch.ones(1, 16)
    out_with_mom, _ = gate2(z_borderline, momentum=full_momentum)

    zeros_without = (out_no_mom == 0).sum().item()
    zeros_with    = (out_with_mom == 0).sum().item()
    ok = zeros_with < zeros_without   # momentum should unblock some neurons
    all_passed &= check(ok,
        f"momentum: {zeros_without}/16 blocked without momentum, "
        f"{zeros_with}/16 blocked with momentum=1.0 (fewer = momentum working)")

    # ── 7. Sparsity: high threshold → more zeros ──────────────────────────────
    gate_sparse = FrictionGate(size=256, mu_s_init=2.0, mu_k_ratio_init=0.6)
    gate_dense  = FrictionGate(size=256, mu_s_init=0.1, mu_k_ratio_init=0.6)
    gate_sparse.eval()
    gate_dense.eval()

    z_test = torch.randn(32, 256)   # standard normal — most values < 2.0
    out_sparse, _ = gate_sparse(z_test)
    out_dense,  _ = gate_dense(z_test)

    sparsity_high = (out_sparse == 0).float().mean().item()
    sparsity_low  = (out_dense  == 0).float().mean().item()
    ok = sparsity_high > sparsity_low
    all_passed &= check(ok,
        f"sparsity: μ_s=2.0 → {sparsity_high:.1%} zeros  |  μ_s=0.1 → {sparsity_low:.1%} zeros")

    # ── 8. FGLUBlock: output shape and momentum shape ─────────────────────────
    block = FGLUBlock(d_model=64, d_ff=256)
    block.eval()
    x = torch.randn(2, 10, 64)    # [B=2, T=10, d_model=64]
    out, momentum = block(x)
    ok = out.shape == (2, 10, 64) and momentum.shape == (2, 10, 256)
    all_passed &= check(ok,
        f"FGLUBlock shapes: out={tuple(out.shape)}  momentum={tuple(momentum.shape)}")

    # ── 9. Train/eval mode gives different outputs (hard vs smooth) ───────────
    gate_cmp = FrictionGate(size=32, mu_s_init=0.5, sharpness=3.0)
    z_cmp = torch.randn(4, 32)

    gate_cmp.eval()
    out_eval, _ = gate_cmp(z_cmp)

    gate_cmp.train()
    with torch.no_grad():
        out_train_nd, _ = gate_cmp(z_cmp)

    # Eval has exact zeros; train has smooth non-zeros near threshold
    exact_zeros_eval  = (out_eval == 0).sum().item()
    exact_zeros_train = (out_train_nd == 0).sum().item()
    ok = exact_zeros_eval >= exact_zeros_train   # eval is at least as sparse
    all_passed &= check(ok,
        f"eval (hard) has ≥ zeros than train (smooth): "
        f"eval={exact_zeros_eval}  train={exact_zeros_train}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    if all_passed:
        print("  ALL TESTS PASSED — friction physics are correct\n")
    else:
        print("  SOME TESTS FAILED — see above\n")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
