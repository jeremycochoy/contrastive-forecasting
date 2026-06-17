"""#350 correctness gate for the learnable log-bilinear main loss.

Checks, on the xshh_allt loss shape used by the experiment:
  1. W = (1/τ)·I reproduces the τ baseline (to fp tolerance) — both the default
     checkpoint path and the fused autograd Function path.
  2. The gradient reaches W (non-zero), and a finite-difference probe matches
     autograd for a couple of W entries.
  3. A non-identity W changes the loss (the knob is live).
Run: python3 experiments/2026-04-13_gift-eval/scripts/test_main_bilinear_W.py
"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.loss import contrastive_latent_loss  # noqa: E402

TAU = 0.10
B, T, C, H = 6, 8, 1, 16


def make_spec():
    return SimpleNamespace(train_configuration={
        "loss_shape": "cosine_similarity_batch_full_hh_negs_xshh_allt",
        "contrastive_divergence_temperature": TAU,
        "include_positive_in_denominator": True,
        "stopgrad_positive_h": True,
        "subtract_contrastive_floor": True,
    })


def identity_W(dev, dtype):
    w = torch.nn.Linear(H, H, bias=False).to(dev, dtype)
    with torch.no_grad():
        w.weight.copy_(torch.eye(H, device=dev, dtype=dtype) / TAU)
    return w


def loss_of(f, o, W=None):
    return contrastive_latent_loss((f, o), validation=False, spec=make_spec(),
                                   main_bilinear_W=W)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    f = torch.randn(B, T, C, H, device=dev, dtype=torch.float64)
    o = torch.randn(B, T, C, H, device=dev, dtype=torch.float64)

    base = loss_of(f, o)
    W = identity_W(dev, torch.float64)
    bil = loss_of(f, o, W)
    print(f"[1] baseline={base.item():.10f}  W=(1/τ)I={bil.item():.10f}  "
          f"|Δ|={abs(base - bil).item():.2e}")
    assert torch.allclose(base, bil, rtol=1e-9, atol=1e-9), "W=(1/τ)I != baseline"

    # Fused autograd Function path (XSHH_ALLT_FUSED=1).
    os.environ["XSHH_ALLT_FUSED"] = "1"
    bil_fused = loss_of(f, o, identity_W(dev, torch.float64))
    base_fused = loss_of(f, o)
    os.environ["XSHH_ALLT_FUSED"] = "0"
    print(f"[1b] fused baseline={base_fused.item():.10f}  "
          f"W=(1/τ)I={bil_fused.item():.10f}  "
          f"|Δ|={abs(base_fused - bil_fused).item():.2e}")
    assert torch.allclose(base_fused, bil_fused, rtol=1e-9, atol=1e-9), \
        "fused W=(1/τ)I != baseline"

    # Gradient reaches W, and matches finite differences for a few entries.
    W = identity_W(dev, torch.float64)
    loss_of(f, o, W).backward()
    g = W.weight.grad.clone()
    print(f"[2] ||grad_W||={g.norm().item():.4e}  (nonzero ⇒ W is trained)")
    assert g.norm().item() > 0, "no gradient on W"
    eps = 1e-6
    for (i, j) in [(0, 0), (1, 3), (5, 2)]:
        Wp = identity_W(dev, torch.float64)
        with torch.no_grad():
            Wp.weight[i, j] += eps
        lp = loss_of(f, o, Wp).item()
        Wm = identity_W(dev, torch.float64)
        with torch.no_grad():
            Wm.weight[i, j] -= eps
        lm = loss_of(f, o, Wm).item()
        fd = (lp - lm) / (2 * eps)
        print(f"    W[{i},{j}]: autograd={g[i, j].item():+.6e}  fd={fd:+.6e}")
        assert abs(fd - g[i, j].item()) < 1e-4, f"grad mismatch at {(i, j)}"

    # A non-identity W changes the loss.
    Wr = identity_W(dev, torch.float64)
    with torch.no_grad():
        Wr.weight.add_(0.5 * torch.randn_like(Wr.weight))
    diff = loss_of(f, o, Wr)
    print(f"[3] random W loss={diff.item():.6f}  baseline={base.item():.6f}  "
          f"(differ ⇒ knob is live)")
    assert not torch.allclose(diff, base, rtol=1e-3), "random W == baseline?!"
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
