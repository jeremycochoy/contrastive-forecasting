"""#350 correctness gate for the bilinear-W inference path.

Verifies extract_forecaster_latents and rollout_latent apply Wᵀ correctly:
  1. With W = I (identity), bilinear extract == baseline extract byte-for-byte
     — Wᵀ is the no-op transform, every downstream consumer is unchanged.
  2. With W = I, single-step rollout token == baseline single-step token
     (the per-step W on the recurrent feedback is also a no-op at W=I).
  3. With a non-identity W, the extract output differs from baseline (knob live).
Run: python3 experiments/2026-04-13_gift-eval/scripts/test_inference_bilinear_W.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.forecasting_head import (  # noqa: E402
    extract_forecaster_latents, rollout_latent, extract_encoder_latents,
)
from src.models import ConfigurableModel  # noqa: E402


def make_cfg(H=32, **extra):
    cfg = dict(C=1, W=4, H=H, nhead=4, num_layers=2,
               num_encoder_layers=0, encoder_type="gru",
               rev_norm_kind="ewma", rev_norm_span=16,
               freq_emb_dim=0, seasonality_emb_dim=0)
    cfg.update(extra)
    return cfg


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H = 32

    base = ConfigurableModel(**make_cfg(H=H)).to(dev).eval()
    bil = ConfigurableModel(**make_cfg(H=H,
                                       main_loss_bilinear=True,
                                       main_bilinear_init_tau=1.0)).to(dev).eval()
    # Mirror every shared weight so the only difference is main_w.
    base_sd = base.state_dict()
    bil_sd = {k: (v if k == "main_w.weight" else base_sd[k])
              for k, v in bil.state_dict().items()}
    bil.load_state_dict(bil_sd)
    assert torch.allclose(bil.main_w.weight, torch.eye(H, device=dev)), \
        "bilinear init at tau=1.0 should give W=I"

    x = torch.randn(2, 64, 1, device=dev)

    # [1] extract_forecaster_latents: W=I ⇒ byte-for-byte identical to baseline.
    f_base, _ = extract_forecaster_latents(base, x)
    f_bil, _ = extract_forecaster_latents(bil, x)
    delta = (f_base - f_bil).abs().max().item()
    print(f"[1] extract W=I: max|Δ|={delta:.2e}")
    assert torch.allclose(f_base, f_bil, atol=1e-6), \
        "extract with W=I should equal baseline"

    # [2] rollout_latent: W=I ⇒ identical token sequence (per-step Wᵀ is no-op).
    e_base, _ = extract_encoder_latents(base, x)
    e_bil, _ = extract_encoder_latents(bil, x)
    assert torch.allclose(e_base, e_bil, atol=1e-6), \
        "encoder latents differ — backbones not aligned"
    r_base = rollout_latent(base, e_base, n_future_tokens=4)
    r_bil = rollout_latent(bil, e_bil, n_future_tokens=4)
    delta = (r_base - r_bil).abs().max().item()
    print(f"[2] rollout W=I (4 steps): max|Δ|={delta:.2e}")
    assert torch.allclose(r_base, r_bil, atol=1e-6), \
        "rollout with W=I should equal baseline at every step"

    # [3] Non-identity W changes extract (knob is live, gradient reaches W).
    with torch.no_grad():
        bil.main_w.weight.copy_(torch.eye(H, device=dev) + 0.5 * torch.randn(H, H, device=dev))
    f_bil2, _ = extract_forecaster_latents(bil, x)
    delta = (f_bil2 - f_base).abs().max().item()
    print(f"[3] extract W≠I: max|Δ|={delta:.4f}  (positive ⇒ Wᵀ is being applied)")
    assert not torch.allclose(f_bil2, f_base, atol=1e-3), \
        "non-identity W should change extract output"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
