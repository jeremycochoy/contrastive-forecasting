"""Spread of the latents along the iterated forecast (strategy B4) on GIFT-Eval test windows.

Usage: [CONTEXT=4096] rollout_spread.py OUT_TSV CODE_DIR NUM_LAYERS CONFIG N_WINDOWS LABEL=CKPT [...]
CONTEXT is the count of context values (the official eval gives 1024, its
predictor default; the backbones train on 4096).

For N_WINDOWS test windows of CONFIG (evenly spaced over its test windows), the
eval's context (the last 4096 values, zeros on the left) and three latents for
each future patch k:
  rolled   the B4 rollout (rollout_latent): the forecaster reads its own outputs;
  onestep  the forecaster's prediction of patch k from the true past;
  true     the encoder latent of the true future (the encoder reads context + truth).
Per k: the effective rank across the windows of each latent, the mean cosine of
rolled and of onestep with true, the norm ratio rolled / true, and the mean
relative change from patch k-1 to patch k of rolled and of true.
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/jupyter/cf_runs/qkgain_test/scripts")
import latent_spectrum as ls  # noqa: E402

T_RAW = int(os.environ.get("CONTEXT", "4096"))  # the eval as scored gives 1024: its predictor default


def config_dataset(ev, config):
    """The gift_eval dataset of a config name, as the eval builds it."""
    for ds_name, term in ev.get_all_dataset_configs():
        if ev.get_ds_config_name(ds_name, term)[0] == config:
            multivariate = ev.GiftDataset(name=ds_name, term=term, to_univariate=False).target_dim > 1
            return ev.GiftDataset(name=ds_name, term=term, to_univariate=multivariate)
    raise SystemExit(f"no GIFT-Eval config {config}")


def windows(ev, dataset, n):
    """(context, context + truth) tensors of n evenly spaced test windows."""
    horizon = dataset.prediction_length
    ctx_prep = ev.ContrastiveForecasterPredictor(None, None, horizon, "cpu", t_raw=T_RAW, context_pad="zeros")
    full_prep = ev.ContrastiveForecasterPredictor(None, None, horizon, "cpu", t_raw=T_RAW + horizon,
                                                  context_pad="zeros")
    inputs, labels = list(dataset.test_data.input), list(dataset.test_data.label)
    ctx, full = [], []
    for i in np.linspace(0, len(inputs) - 1, min(n, len(inputs))).round().astype(int):
        target = ctx_prep._fill_missing(np.asarray(inputs[i]["target"], dtype=np.float32))
        truth = ev.fill_horizon(labels[i]["target"], target[-1])
        ctx.append(ctx_prep._prepare_context(target))
        full.append(full_prep._prepare_context(np.concatenate([target, truth])))
    return torch.cat(ctx), torch.cat(full)


def eff_rank(z):
    z = (z - z.mean(0, keepdim=True)).double()
    eig = torch.linalg.eigvalsh(z.T @ z).clamp_min(0)
    p = eig / eig.sum()
    return math.exp(-(p[p > 0] * p[p > 0].log()).sum().item())


def step_change(x):
    """Mean relative change from patch k-1 to patch k, per k (first k: nan)."""
    rel = (x[:, 1:] - x[:, :-1]).norm(dim=-1) / x[:, :-1].norm(dim=-1)
    return [float("nan")] + rel.mean(0).tolist()


CHUNK = 25  # windows per forward pass: 200 windows of 4816 values do not fit a 24 GB card


def latents(att, ev, path, num_layers, ctx, full, freq):
    """The three latents of all windows, CHUNK windows at a time, and the
    relative gap between the context latents of the two passes (0 if causal)."""
    backbone = ls_backbone(att, path, num_layers)
    parts = [chunk_latents(backbone, ctx[i:i + CHUNK], full[i:i + CHUNK], freq)
             for i in range(0, len(ctx), CHUNK)]
    z = {k: torch.cat([p[0][k] for p in parts]) for k in parts[0][0]}
    gap_sq = sum(p[1] for p in parts) / sum(p[2] for p in parts)
    return z, gap_sq ** 0.5


def chunk_latents(backbone, ctx, full, freq):
    from src.forecasting_head import extract_encoder_latents, rollout_latent
    from src.freq_embedding import freq_labels
    fid, sid = freq_labels(freq, "v2")
    ids = dict(freq_ids=torch.full((len(ctx),), fid, device="cuda"),
               seasonality_ids=torch.full((len(ctx),), sid, device="cuda"))
    m = (full.shape[1] - T_RAW) // backbone.W
    with torch.no_grad():
        e_ctx, _ = extract_encoder_latents(backbone, ctx.cuda(), **ids)
        rolled = rollout_latent(backbone, e_ctx, m)
        h_full, _ = extract_encoder_latents(backbone, full.cuda(), **ids)
        f_all = backbone.transformer.forecaster_forward(h_full, fp32_tail=False, cache_mask=False)
    n = e_ctx.shape[1]
    z = {"rolled": rolled.float().cpu(), "onestep": f_all[:, n - 1:n - 1 + m].float().cpu(),
         "true": h_full[:, n:n + m].float().cpu()}
    return z, (h_full[:, :n] - e_ctx).pow(2).sum().item(), e_ctx.pow(2).sum().item()


def ls_backbone(att, path, num_layers):
    job = [a if a != "6" else str(num_layers) for a in att.JOB_ARGS]
    args = att.tfh.parse_args(["--backbone-path", path] + job)
    sd = torch.load(path, map_location="cuda", weights_only=True)
    config, _ = att.tfh.backbone_config(args, sd)
    return att.tfh.load_frozen_backbone(args, config, sd, "cuda")


def rows(label, config, z):
    cos = torch.nn.functional.cosine_similarity
    roll, one, true = z["rolled"], z["onestep"], z["true"]
    ch_roll, ch_true = step_change(roll), step_change(true)
    for k in range(roll.shape[1]):
        yield (label, config, k + 1, eff_rank(roll[:, k]), eff_rank(one[:, k]), eff_rank(true[:, k]),
               cos(roll[:, k], true[:, k], dim=-1).mean().item(), cos(one[:, k], true[:, k], dim=-1).mean().item(),
               (roll[:, k].norm(dim=-1).mean() / true[:, k].norm(dim=-1).mean()).item(), ch_roll[k], ch_true[k])


def main():
    out, code_dir, num_layers, config, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    att = ls.load_batch_module(code_dir)
    import eval_gift_eval_official as ev
    dataset = config_dataset(ev, config)
    ctx, full = windows(ev, dataset, n)
    header = ("label\tconfig\tk\trank_rolled\trank_onestep\trank_true\tcos_rolled\tcos_onestep\t"
              "norm_ratio\tchange_rolled\tchange_true\n")
    with open(out, "a") as fh:
        if fh.tell() == 0:
            fh.write(header)
        for item in sys.argv[6:]:
            label, path = item.split("=", 1)
            z, gap = latents(att, ev, path, num_layers, ctx, full, dataset.freq)
            print(f"{label} {config}: {len(ctx)} windows, {z['rolled'].shape[1]} future patches, "
                  f"context latents differ by {gap:.1e} between the two passes", flush=True)
            for r in rows(label, config, z):
                fh.write("\t".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + "\n")


if __name__ == "__main__":
    main()
