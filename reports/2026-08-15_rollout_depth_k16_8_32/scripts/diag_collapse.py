#!/usr/bin/env python3
"""#401 diagnosis — is the saved backbone a constant function?

The three phase-1 GIFT-Eval scores are 2x to 7x worse than every number #373
published on this pipeline. The trainer's own diagnostics say the k = 8 and
k = 16 runs collapsed: AUC 0.50, Top1 at chance, u_batch 0.017 against
#373's 0.146. Those numbers are measured inside the training loop.

This script measures the SAVED CHECKPOINT instead, through the loader the
GIFT-Eval uses, on real GIFT-Eval windows. It closes the loop between the
training metric and the artefact the eval consumed.

Per backbone it reports, on the encoder latent h:

    pair_cos    mean cosine between the h of two DIFFERENT series. 1.0 means
                every input maps to one direction: the encoder is constant.
    dim_std     mean per-dimension standard deviation across the batch.
    eff_rank    participation ratio of the covariance spectrum,
                (sum l)^2 / sum l^2, in [1, H]. 1.0 is a single direction.
    cos_err_d0  1 - cos(f_t, h_{t+1}), the trainer's own `cos_err_d0`, on
                these windows. A collapsed encoder makes it ~0 for free.

`--all` keeps every one of those measurements unchanged and only widens the
subject list. It walks the checkpoint tree and measures EVERY periodic
backbone the study wrote, at k = 0, 8, 16 and 32, so no cell of the grid
rests on an unmeasured checkpoint. The first five subjects stay in the list,
so an `--all` run reproduces the narrow run row for row.

Usage:
    python3 diag_collapse.py --out results/diag/collapse.csv
    python3 diag_collapse.py --all --out results/diag/collapse_all.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parent.parent
sys.path.insert(0, str(REPO))

from src.models import ConfigurableModel                      # noqa: E402
from src.checkpoint import prepare_backbone_state_dict        # noqa: E402
from src.forecasting_head import (                            # noqa: E402
    extract_encoder_latents,
    extract_forecaster_latents,
)

T_RAW = 4096

# The subjects. Each row is (label, k, stop, checkpoint).
#   The three #401 backbones the phase-1 scores were measured on, plus the
#   two known-good controls: #393's k = 0 parent of this very cell, and
#   #373's own G1 subject.
CF401 = "/home/jupyter/checkpoints_backup/cf-401"
SUBJECTS = [
    ("393 parent  k=0  bb40k", 0, 40,
     "/home/jupyter/checkpoints_backup/cf-393/arm6_v2_combab_alignS/"
     "leg_40k/cf393_arm6_v2_combab_alignS_40k.pth"),
    ("379 B5pub   k=0  bb40k", 0, 40,
     "/home/jupyter/checkpoints_backup/cf-379/runs/bb_small_arm4_combab_"
     "xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_40k.pth"),
    ("401 arm     k=8  bb40k", 8, 40,
     f"{CF401}/k8/arm6_v2_combab_alignS/leg_40k/"
     "cf393_arm6_v2_combab_alignS_cf373k8_40k.pth"),
    ("401 arm     k=8  bb100k", 8, 100,
     f"{CF401}/k8/arm6_v2_combab_alignS/leg_100k/"
     "cf393_arm6_v2_combab_alignS_cf373k8_100k.pth"),
    ("401 arm     k=16 bb40k", 16, 40,
     f"{CF401}/k16/arm6_v2_combab_alignS/leg_40k/"
     "cf393_arm6_v2_combab_alignS_cf373k16_40k.pth"),
]

CF393 = "/home/jupyter/checkpoints_backup/cf-393/arm6_v2_combab_alignS"

# Where each arm keeps its periodic backbones. The k = 0 row is the parent
# this study branched from, so the ladder has a measured k = 0 point.
ARM_DIRS = [
    (0, CF393, "cf393_arm6_v2_combab_alignS"),
    (8, f"{CF401}/k8/arm6_v2_combab_alignS",
     "cf393_arm6_v2_combab_alignS_cf373k8"),
    (16, f"{CF401}/k16/arm6_v2_combab_alignS",
     "cf393_arm6_v2_combab_alignS_cf373k16"),
    (32, f"{CF401}/k32/arm6_v2_combab_alignS",
     "cf393_arm6_v2_combab_alignS_cf373k32"),
]

# The three legs the study scores. A leg's last step is the "stop" the
# GIFT-Eval head was trained on: bb40k, bb100k, bb200k.
LEGS = [("leg_40k", 40), ("leg_100k", 100), ("leg_200k", 200)]


def discover():
    """Every periodic backbone the study wrote, in (k, step) order.

    This adds subjects. It changes no measurement. A checkpoint that is not
    on disk yet is skipped, so the k = 32 200k leg joins the table as its
    periodic files land.
    """
    rows = []
    for k, root, stem in ARM_DIRS:
        for leg, stop in LEGS:
            d = Path(root) / leg
            if not d.is_dir():
                continue
            for p in sorted(d.glob(f"{stem}_*k.pth"),
                            key=lambda q: int(q.stem.rsplit("_", 1)[1][:-1])):
                step = int(p.stem.rsplit("_", 1)[1][:-1])
                rows.append((f"k={k:<2} step {step:>3}k", k, stop, str(p)))
    return rows


# Real windows, from datasets that span the benchmark's frequencies and
# domains. Synthetic input would leave open whether the encoder separates
# real series. These are the series the score is computed on.
DATASETS = [
    ("ett1", "H"), ("ett2", "H"), ("electricity", "H"), ("solar", "H"),
    ("jena_weather", "H"), ("us_births", "D"), ("saugeenday", "D"),
    ("bitbrains_fast_storage", "H"),
]


def load_backbone(path, device):
    """Build and load the backbone exactly as eval_gift_eval_official.py does.

    Same auto-detection, same `prepare_backbone_state_dict`, same STRICT
    `load_state_dict`. A strict load is itself a check: an architecture that
    did not match the checkpoint would raise here rather than score badly.
    """
    sd = torch.load(path, map_location=device, weights_only=True)
    cfg = dict(C=1, H=64, W=16, encoder_type="gru", num_layers=3, nhead=8,
               ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
               rev_norm_kind="ewma", rev_norm_span=128)
    w = sd.get("freq_embedding.embedding.weight")
    cfg["freq_emb_dim"] = w.shape[1] if w is not None else 0
    sw = sd.get("seasonality_embedding.embedding.weight")
    cfg["seasonality_emb_dim"] = sw.shape[1] if sw is not None else 0
    enc = {int(k.split(".")[2]) for k in sd
           if k.startswith("transformer.encoder_layers.")}
    cfg["num_encoder_layers"] = (max(enc) + 1) if enc else 0
    if any(k.endswith(".q_norm.weight") for k in sd):
        cfg["qk_norm"] = True
    if any(k.endswith(".attn_out_rms.weight") for k in sd):
        cfg["attn_out_norm"] = True
    model = ConfigurableModel(**cfg)
    model.load_state_dict(prepare_backbone_state_dict(sd, "student"))
    return model.to(device).eval(), cfg


def real_windows(n_per_ds=8):
    """One (n, T_RAW, 1) tensor of real GIFT-Eval context windows."""
    from datasets import load_from_disk
    root = Path("/home/jupyter/workspaces/gift-eval-data")
    out = []
    for name, freq in DATASETS:
        d = root / name / freq
        if not d.is_dir():
            continue
        ds = load_from_disk(str(d))
        col = "target" if "target" in ds.column_names else ds.column_names[-1]
        taken = 0
        for row in ds:
            t = np.asarray(row[col], dtype=np.float64)
            if t.ndim > 1:
                t = t[0]
            t = t[np.isfinite(t)]
            if t.size < T_RAW // 4:
                continue
            if t.size >= T_RAW:
                w = t[-T_RAW:]
            else:
                w = np.concatenate([np.full(T_RAW - t.size, t[0]), t])
            out.append(w.astype(np.float32))
            taken += 1
            if taken >= n_per_ds:
                break
    x = torch.from_numpy(np.stack(out)).unsqueeze(-1)   # (n, T_RAW, 1)
    return x


@torch.no_grad()
def measure(model, x, device):
    """Latent spread of h, and the depth-0 rollout error, on these windows.

    `extract_encoder_latents` and `extract_forecaster_latents` are the two
    the GIFT-Eval and the head trainer call, so h and f here are the same
    tensors the score was computed from. `compute_metrics` in src/models.py
    reads them as f_t against h_{t+1}, which is the trainer's `ff` column.
    """
    x = x.to(device)
    h = extract_encoder_latents(model, x)[0]
    f = extract_forecaster_latents(model, x)[0]
    h = h.reshape(h.shape[0], h.shape[1], -1)
    f = f.reshape(f.shape[0], f.shape[1], -1)
    n, T, H = h.shape

    # One vector per series: the last latent, the state a forecast rolls from.
    v = h[:, -1, :].float()
    vn = torch.nn.functional.normalize(v, dim=-1)
    cos = vn @ vn.T
    off = ~torch.eye(n, dtype=torch.bool, device=cos.device)
    pair_cos = cos[off].mean().item()

    dim_std = v.std(dim=0).mean().item()
    c = torch.cov(v.T.double())
    lam = torch.linalg.eigvalsh(c).clamp(min=0)
    eff_rank = ((lam.sum() ** 2) / (lam.pow(2).sum() + 1e-30)).item()

    # cos_err_d0 = 1 - cos(f_t, h_{t+1}), the trainer's own column.
    fc = torch.nn.functional.cosine_similarity(
        f[:, :-1, :].float(), h[:, 1:, :].float(), dim=-1)
    return dict(pair_cos=pair_cos, dim_std=dim_std, eff_rank=eff_rank,
                cos_err_d0=(1.0 - fc.mean()).item(), n_series=n, H=H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/diag/collapse.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--per-dataset", type=int, default=8)
    ap.add_argument("--all", action="store_true",
                    help="measure every periodic backbone on disk too")
    a = ap.parse_args()

    subjects = list(SUBJECTS)
    if a.all:
        have = {s[3] for s in subjects}
        subjects += [r for r in discover() if r[3] not in have]

    x = real_windows(a.per_dataset)
    print(f"{x.shape[0]} real windows of {x.shape[1]} steps "
          f"from {len(DATASETS)} datasets\n")

    cols = ["label", "k", "stop_k", "step_k", "pair_cos", "dim_std",
            "eff_rank", "cos_err_d0", "n_series", "H", "checkpoint"]
    rows = []
    print(f"{'backbone':<24} {'pair_cos':>9} {'dim_std':>9} {'eff_rank':>9} "
          f"{'cos_err_d0':>11}")
    for label, k, stop, path in subjects:
        if not Path(path).is_file():
            print(f"{label:<24} {'MISSING':>9}   {path}")
            continue
        model, cfg = load_backbone(path, a.device)
        m = measure(model, x, a.device)
        print(f"{label:<24} {m['pair_cos']:9.5f} {m['dim_std']:9.5f} "
              f"{m['eff_rank']:9.3f} {m['cos_err_d0']:11.5f}")
        step = Path(path).stem.rsplit("_", 1)[1]
        step = int(step[:-1]) if step.endswith("k") and step[:-1].isdigit() \
            else stop
        rows.append(dict(label=label, k=k, stop_k=stop, step_k=step,
                         checkpoint=path, **m))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r[c] for c in cols})
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
