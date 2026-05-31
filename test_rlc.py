"""
Unit tests for the full RLC circuit neuron.

What we verify
──────────────
  1.  L, R, C > 0 always  (log parameterisation works)
  2.  Critical damping at init  ζ ≈ 1
  3.  Natural frequency formula  ω₀ = 1/√(LC)
  4.  Charge accumulates — q grows with applied voltage
  5.  Kinetic drag from friction gate on charge (not raw voltage)
  6.  Overdamped regime: no oscillation, q settles monotonically
  7.  Underdamped regime: q oscillates (sign changes across "time" steps)
  8.  State threads across blocks: q from block L seeds block L+1
  9.  Gradients flow through L, R, C parameters
  10. RLCFrictionLM forward pass: correct shapes, loss decreases
  11. Circuit report: per-layer ω₀ and ζ are reported correctly
"""

import math
import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from friction_llm.rlc_neuron import RLCNeuron, RLCFrictionBlock
from friction_llm.rlc_model  import RLCFrictionLM
from friction_llm.config     import FrictionConfig


PASS = "  PASS"
FAIL = "  FAIL"


def check(condition: bool, name: str) -> bool:
    print(f"{PASS if condition else FAIL} — {name}")
    return condition


def run_tests() -> None:
    print("\n── RLC Circuit Unit Tests ─────────────────────────────────────────\n")
    ok_all = True

    # ── 1. L, R, C always positive ────────────────────────────────────────────
    rlc = RLCNeuron(size=32, L_init=1.0, R_init=2.0, C_init=1.0)
    ok = bool((rlc.L > 0).all() and (rlc.R > 0).all() and (rlc.C > 0).all())
    ok_all &= check(ok, f"L={rlc.L.mean():.3f}, R={rlc.R.mean():.3f}, C={rlc.C.mean():.3f}  all > 0")

    # ── 2. Critical damping at init ────────────────────────────────────────────
    zeta = rlc.damping_ratio.mean().item()
    ok = abs(zeta - 1.0) < 0.01
    ok_all &= check(ok, f"critical damping at init  ζ={zeta:.4f}  (expected ≈ 1.000)")

    # ── 3. Natural frequency formula ω₀ = 1/√(LC) ────────────────────────────
    L_val = rlc.L.mean().item()
    C_val = rlc.C.mean().item()
    omega_expected = 1.0 / math.sqrt(L_val * C_val)
    omega_computed = rlc.omega_0.mean().item()
    ok = abs(omega_expected - omega_computed) < 1e-4
    ok_all &= check(ok, f"ω₀ = 1/√(LC) = {omega_expected:.4f}  computed={omega_computed:.4f}")

    # ── 4. Charge accumulates with applied voltage ────────────────────────────
    rlc.eval()
    V = torch.ones(1, 32) * 2.0    # constant positive voltage
    q_prev = torch.zeros(1, 32)
    state  = None
    charges = []
    for _ in range(10):
        q, state = rlc(V, state)
        charges.append(q.mean().item())

    # Under constant positive V, charge should increase from 0
    ok = charges[-1] > charges[0]
    ok_all &= check(ok, f"charge accumulates: q[0]={charges[0]:.4f} → q[9]={charges[-1]:.4f}")

    # ── 5. Friction gate fires on CHARGE, not raw voltage ─────────────────────
    # Measure sparsity on the charge q directly — the final output passes through
    # W_out which has a bias term, masking exact zeros in the output tensor.
    block = RLCFrictionBlock(d_model=64, d_ff=128, mu_s_init=0.3)
    block.eval()

    x_weak   = torch.randn(1, 4, 64) * 0.01   # tiny voltage → tiny charge → stuck
    x_strong = torch.randn(1, 4, 64) * 50.0   # large voltage → large charge → breaks free

    V_weak   = block.W_gate(x_weak)
    q_weak,  _ = block.rlc(V_weak)
    sparsity_weak = block.friction.measure_sparsity(q_weak)

    V_strong = block.W_gate(x_strong)
    q_strong, _ = block.rlc(V_strong)
    sparsity_strong = block.friction.measure_sparsity(q_strong)

    ok = sparsity_weak > sparsity_strong
    ok_all &= check(ok,
        f"friction on charge q: weak→{sparsity_weak:.1%} zeros  "
        f"strong→{sparsity_strong:.1%} zeros  (more zeros when charge < μ_s)")

    # ── 6. Overdamped regime: no oscillation ─────────────────────────────────
    # High R → ζ >> 1 → charge rises smoothly without crossing zero
    rlc_over = RLCNeuron(size=8, L_init=1.0, R_init=20.0, C_init=1.0, dt=0.05)
    rlc_over.eval()
    V_step = torch.ones(1, 8)
    state  = None
    sign_changes = 0
    q_last = None
    for _ in range(30):
        q, state = rlc_over(V_step, state)
        if q_last is not None:
            sign_changes += int((q * q_last < 0).any().item())
        q_last = q.detach()
    ok = sign_changes == 0
    ok_all &= check(ok, f"overdamped (R=20): 0 sign changes in q  (got {sign_changes})")

    # ── 7. Underdamped regime: oscillation ────────────────────────────────────
    # Low R → ζ << 1 → charge oscillates.
    # With ω₀=1 and dt=0.05, one full cycle ≈ 2π/0.05 ≈ 126 steps; first sign
    # change (half cycle) ≈ 63 steps — run 80 to be safe.
    rlc_under = RLCNeuron(size=8, L_init=1.0, R_init=0.05, C_init=1.0, dt=0.05)
    rlc_under.eval()
    V_pulse = torch.zeros(1, 8)
    state = (torch.zeros(1, 8), torch.ones(1, 8) * 2.0)   # kick with initial current
    sign_changes = 0
    q_last = None
    for _ in range(80):
        q, state = rlc_under(V_pulse, state)
        if q_last is not None:
            sign_changes += int((q * q_last < 0).any().item())
        q_last = q.detach()
    ok = sign_changes > 0
    ok_all &= check(ok, f"underdamped (R=0.05): oscillation confirmed — {sign_changes} sign changes in 80 steps")

    # ── 8. State threads from block L to block L+1 ────────────────────────────
    # We verify on the RLC charge q directly — final output passes through
    # W_out bias which can mask differences when friction zeroes everything.
    b1 = RLCFrictionBlock(d_model=64, d_ff=128, dt=0.5)   # larger dt = more charge per step
    b2 = RLCFrictionBlock(d_model=64, d_ff=128, dt=0.5)
    b1.eval(); b2.eval()
    x = torch.randn(2, 8, 64) * 2.0   # strong enough to build charge

    # B1 produces a non-trivial circuit state
    _, state_out, _ = b1(x, rlc_state=None, momentum=None)
    q1, i1 = state_out

    # Compute what B2's RLC produces with B1's state vs zero state
    V2 = b2.W_gate(x)
    q_seeded, _   = b2.rlc(V2, state=(q1, i1))
    q_cold,   _   = b2.rlc(V2, state=None)

    # The seeded charge should differ from cold-start charge
    ok = not torch.allclose(q_seeded, q_cold, atol=1e-5)
    delta = (q_seeded - q_cold).abs().mean().item()
    ok_all &= check(ok, f"state threading: seeded q differs from cold q by Δ={delta:.5f}")

    # ── 9. Gradients flow through L, R, C ─────────────────────────────────────
    rlc_grad = RLCNeuron(size=16, L_init=1.0, R_init=2.0, C_init=1.0)
    V_g = torch.randn(4, 16, requires_grad=False)
    q_out, _ = rlc_grad(V_g)
    q_out.sum().backward()

    grad_L = rlc_grad.log_L.grad
    grad_R = rlc_grad.log_R.grad
    grad_C = rlc_grad.log_C.grad

    ok = all(
        g is not None and not g.isnan().any()
        for g in [grad_L, grad_R, grad_C]
    )
    ok_all &= check(ok,
        f"gradients through L/R/C: "
        f"∂/∂L={'✓' if grad_L is not None else '✗'}  "
        f"∂/∂R={'✓' if grad_R is not None else '✗'}  "
        f"∂/∂C={'✓' if grad_C is not None else '✗'}")

    # ── 10. RLCFrictionLM full forward + loss ─────────────────────────────────
    cfg = FrictionConfig.tiny()
    cfg.use_rlc = True
    model = RLCFrictionLM(cfg)
    model.eval()

    idx  = torch.randint(0, cfg.vocab_size, (2, 32))
    tgt  = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(idx, tgt)

    ok = logits.shape == (2, 32, cfg.vocab_size) and loss is not None and not loss.isnan()
    ok_all &= check(ok,
        f"RLCFrictionLM forward: logits{tuple(logits.shape)}  loss={loss.item():.4f}")

    # ── 11. Loss decreases with gradient step ─────────────────────────────────
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(5):
        _, loss = model(idx, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    ok = losses[-1] < losses[0]
    ok_all &= check(ok, f"loss decreases over 5 steps: {losses[0]:.4f} → {losses[-1]:.4f}")

    # ── 12. Circuit report: ω₀ and ζ per layer ───────────────────────────────
    model.eval()
    report = model.circuit_report(idx)
    has_all_layers = all(f"layer_{i}" in report for i in range(cfg.n_layers))
    first = report["layer_0"]
    ok = has_all_layers and "omega_0_mean" in first and "zeta_mean" in first
    ok_all &= check(ok,
        f"circuit report: {cfg.n_layers} layers  "
        f"ω₀={first['omega_0_mean']:.3f}  ζ={first['zeta_mean']:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 68)
    if ok_all:
        print("  ALL TESTS PASSED — RLC circuit physics are correct\n")
    else:
        print("  SOME TESTS FAILED — see above\n")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
