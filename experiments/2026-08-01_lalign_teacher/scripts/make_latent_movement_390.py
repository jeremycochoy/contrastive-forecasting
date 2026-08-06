#!/usr/bin/env python3
"""Latent movement between adjacent checkpoints, for #390's ten retrained arms.

Same measure and same fixed batch as #379's
``reports/2026-07-21_split_pred_rep_small/plots/_make_latent_movement.py``:

    drift_h = mean over (b, t, c) of  1 - cos(h_t(model_j), h_t(model_i))
    drift_e = mean over (b, t, c) of  1 - cos(e_t(model_j), e_t(model_i))

over adjacent periodic checkpoints of one run. The batch is #379's committed
``plots/_latent_movement_batch.pt``, so the ten teacher-target arms land on
exactly the scale of the twenty arms this experiment did not retrain. Running
#379's script unchanged against `--arms arm1` reproduces its committed rows
bit for bit, which is what pins the two sets to one scale.

The only thing that differs from #379 is the run names: #390's carry the
`_alignteacher` suffix, and #379's RUNS table is hardcoded.

    PYTHONPATH=<repo> python3 make_latent_movement_390.py \
        --runs-dir <wt>/experiments/2026-08-01_lalign_teacher/runs \
        --batch <repo>/reports/2026-07-21_split_pred_rep_small/plots/_latent_movement_batch.pt \
        --out <repo>/reports/2026-08-04_lalign_teacher/results/latent_movement_390.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.eval_latent_movement import (           # noqa: E402
    compute_latents, load_backbone, mean_one_minus_cos, small_backbone_kwargs,
)

# arm slug -> (label, backbone run name). Names follow arm_names.sh's rule:
# bb_small_<arm>_lalign_<lrep|lrepmoco>_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher
BASE = "_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher"
VARIANT_LABEL = {
    "": "",
    "_tr1": ", all τ=1.0",
    "_nse": ", sigreg_e=0",
    "_ncpc": ", cpc=0",
    "_combab": ", τ_rep=1.0 + cpc=0",
}
ARMS: list[tuple[str, str, str]] = []
for arm, family, loss in (("arm5", "lrep", "L_align→teacher + L_rep"),
                          ("arm6_v2", "lrepmoco", "L_align→teacher + L_rep_moco")):
    for variant, extra in VARIANT_LABEL.items():
        slug = f"{arm}{variant}"
        pretty = arm.replace("arm", "arm ").replace("_v2", " v2")
        label = f"{pretty}{variant.replace('_', ' ')}  ({loss}{extra})"
        ARMS.append((slug, label, f"bb_small_{slug}_lalign_{family}{BASE}"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--batch", required=True,
                   help="#379's committed fixed batch (.pt)")
    p.add_argument("--out", required=True)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def list_periodic_checkpoints(runs_dir: Path, name: str) -> list[tuple[int, Path]]:
    """``[(step, path), ...]`` sorted by step for ``<name>[_rN]_<k>k.pth``.

    Identical rule to #379's: optimizer companions, ``_final``, ``_best_*``
    are skipped, so only periodic training milestones appear. A resumed wave
    writes ``<name>_r2_50k.pth``, which the ``<name>_*k.pth`` glob still
    catches and whose trailing token is still ``50k``.
    """
    out: list[tuple[int, Path]] = []
    for p in runs_dir.glob(f"{name}_*k.pth"):
        stem = p.stem
        if stem.endswith("_optimizer"):
            continue
        tail = stem.rsplit("_", 1)[-1]
        if not tail.endswith("k") or not tail[:-1].isdigit():
            continue
        out.append((int(tail[:-1]) * 1000, p))
    out.sort(key=lambda kv: kv[0])
    return out


def arm_movements(runs_dir: Path, name: str, batch: torch.Tensor,
                  device: str) -> list[tuple[int, float, float]]:
    cks = list_periodic_checkpoints(runs_dir, name)
    if len(cks) < 2:
        return []
    kwargs = small_backbone_kwargs(C=batch.shape[-1])
    x = batch.to(device)
    B = x.shape[0]
    # Frozen "unknown" bucket for both conditioning embeddings, so adjacent
    # checkpoints see identical conditioning. Same as #379.
    freq_ids = torch.zeros(B, dtype=torch.long, device=device)
    seasonality_ids = torch.zeros(B, dtype=torch.long, device=device)
    prev_model = load_backbone(str(cks[0][1]), kwargs, device)
    prev_h, prev_e = compute_latents(prev_model, x, freq_ids, seasonality_ids)
    del prev_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    out: list[tuple[int, float, float]] = []
    for cur_step, cur_path in cks[1:]:
        cur_model = load_backbone(str(cur_path), kwargs, device)
        cur_h, cur_e = compute_latents(cur_model, x, freq_ids, seasonality_ids)
        out.append((cur_step,
                    mean_one_minus_cos(prev_h, cur_h),
                    mean_one_minus_cos(prev_e, cur_e)))
        del cur_model
        prev_h, prev_e = cur_h, cur_e
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise SystemExit(f"runs dir not found: {runs_dir}")
    batch = torch.load(args.batch, map_location="cpu", weights_only=False)
    print(f"fixed batch: shape={tuple(batch.shape)} from {args.batch}")

    rows: list[tuple] = []
    missing: list[str] = []
    for slug, label, name in ARMS:
        pts = arm_movements(runs_dir, name, batch, args.device)
        if not pts:
            missing.append(slug)
            print(f"  skip {slug}: fewer than 2 periodic checkpoints under {name}")
            continue
        for st, h, e in pts:
            rows.append((slug, label, st, f"{h:.6f}", f"{e:.6f}"))
        print(f"  {label}: {len(pts)} pairs → "
              f"h_last={pts[-1][1]:.4f}  e_last={pts[-1][2]:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm_slug", "label", "step_later", "drift_h", "drift_e"])
        w.writerows(rows)
    tmp.replace(out)
    print(f"wrote {out}  ({len(rows)} pairs, "
          f"{len({r[0] for r in rows})}/{len(ARMS)} arms)")
    if missing:
        print(f"MISSING arms: {' '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
