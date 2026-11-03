"""Causality of the backbone: future values must not change any earlier latent.

Usage: CONTEXT=4096 causality_check.py CODE_DIR NUM_LAYERS CONFIG LABEL=CKPT [...]

For 8 test windows of CONFIG: the input context + truth, and the same context
followed by random noise in place of the truth. Per stage (the normalised
input, the encoder latents, the forecaster outputs), the largest relative
change over the context positions (0 if causal), and the first position that
changes.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/jupyter/cf_runs/qkgain_test/scripts")
import latent_spectrum as ls  # noqa: E402
import rollout_spread as rs  # noqa: E402


def first_change(a, b, tol=1e-4):
    """Largest relative change over the context, and the first index that changes."""
    rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-12)   # (B, T)
    worst = rel.max(0).values
    changed = (worst > tol).nonzero()
    return worst, int(changed[0]) if len(changed) else None


def stages(backbone, x, ids):
    from src.forecasting_head import extract_encoder_latents
    with torch.no_grad():
        x_norm = backbone.rev_norm(x.cuda(), mode="norm")
        h, _ = extract_encoder_latents(backbone, x.cuda(), **ids)
        f = backbone.transformer.forecaster_forward(h, fp32_tail=False, cache_mask=False)
    return {"normalised input": x_norm[..., 0:1], "encoder latents": h, "forecaster outputs": f}


def main():
    code_dir, num_layers, config = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    att = ls.load_batch_module(code_dir)
    import eval_gift_eval_official as ev
    from src.freq_embedding import freq_labels
    dataset = rs.config_dataset(ev, config)
    _, full = rs.windows(ev, dataset, 8)
    noisy = full.clone()
    gen = np.random.default_rng(0)
    noisy[:, rs.T_RAW:] = torch.from_numpy(gen.normal(0, 50, size=noisy[:, rs.T_RAW:].shape)).float()
    fid, sid = freq_labels(dataset.freq, "v2")
    ids = dict(freq_ids=torch.full((8,), fid, device="cuda"), seasonality_ids=torch.full((8,), sid, device="cuda"))
    for item in sys.argv[4:]:
        label, path = item.split("=", 1)
        backbone = rs.ls_backbone(att, path, num_layers)
        a, b = stages(backbone, full, ids), stages(backbone, noisy, ids)
        for name in a:
            per_step = name == "normalised input"
            n_ctx = rs.T_RAW if per_step else rs.T_RAW // backbone.W
            worst, first = first_change(a[name], b[name])
            print(f"{label} {name}: largest relative change over the {n_ctx} context positions "
                  f"{worst[:n_ctx].max().item():.1e}; first changed position {first} "
                  f"(the future starts at {n_ctx})", flush=True)


if __name__ == "__main__":
    main()
