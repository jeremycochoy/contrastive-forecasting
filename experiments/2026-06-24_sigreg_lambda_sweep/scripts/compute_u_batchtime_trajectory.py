#!/usr/bin/env python3
"""Retroactive cross-(batch × time) dim-usage on EVERY existing checkpoint.

#363 follow-up to ``compute_u_batchtime_retro.py``. The single-checkpoint
script pins one ``u_batchtime`` value per arm at the FINAL backbone; this
trajectory variant walks every saved checkpoint (intermediate ``_*k.pth``,
``_best_loss.pth``, ``_best_gap.pth``, ``_final.pth``, ``_FINAL.pth``) so
the report can plot a real trajectory on the dim-usage panel for arms
trained before the prospective ``u_batchtime`` CSV column was added.

Same dataset / batch / seed / eval mode as ``compute_u_batchtime_retro.py``
— this script imports the resolver, the deterministic batch loader, the
backbone builder, and the e_t/h_t extraction helpers from there. The
single-checkpoint script's behaviour is therefore unchanged.

Output: ``results/u_batchtime_trajectory.csv`` with
    arm, recipe, step, ckpt_kind, backbone_ckpt, u_batchtime, u_batchtime_e
one row per existing checkpoint.

Usage:
    PYTHONPATH=. python3 \\
      experiments/2026-06-24_sigreg_lambda_sweep/scripts/\\
        compute_u_batchtime_trajectory.py
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Iterable

import torch

from src.metrics import u_batchtime

# Reuse the single-checkpoint script's helpers verbatim so both scripts
# evaluate against an identical held-out batch and forward pass.
from compute_u_batchtime_retro import (
    REPO_ROOT,
    RESULTS_DIR,
    _pick_device,
    _resolve_checkpoint,
    load_held_out_batch,
    build_backbone,
    extract_e_and_h,
)


CSV_OUT = os.path.join(RESULTS_DIR, "u_batchtime_trajectory.csv")


# -- Per-arm prefixes. Each ``prefix_rel`` is the *file* stem (relative
# to repo root, no trailing ``_<variant>.pth``); ``enumerate_checkpoints``
# probes every variant in turn using the same multi-root resolver the
# single-checkpoint script uses, so files spread across the current
# worktree, the ``contrastive-forecasting-<issue>`` main checkouts, or
# the elisa main checkout are all picked up. Arms 5 and 6 (emb10_enc10
# / emb10000_enc10) are deliberately omitted: they were still in flight
# on 2026-06-27 and their files belong to the queued launchers — the
# next iteration of this script picks them up once their FINAL.pth
# exists.
ARMS = [
    dict(
        arm="emb100_enc01",
        recipe="λ_e=10.0, λ_h=0.1",
        prefix_rel=("experiments/2026-06-24_sigreg_lambda_sweep/runs/"
                    "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_"
                    "cpc_emb100_enc01"),
    ),
    dict(
        arm="emb100_enc10",
        recipe="λ_e=10.0, λ_h=1.0",
        prefix_rel=("experiments/2026-06-24_sigreg_lambda_sweep/runs/"
                    "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_"
                    "cpc_emb100_enc10"),
    ),
    dict(
        arm="emb100_enc100",
        recipe="λ_e=10.0, λ_h=10.0",
        prefix_rel=("experiments/2026-06-24_sigreg_lambda_sweep/runs/"
                    "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_"
                    "cpc_emb100_enc100"),
    ),
    dict(
        arm="emb1000_enc01",
        recipe="λ_e=100.0, λ_h=0.1",
        prefix_rel=("experiments/2026-06-24_sigreg_lambda_sweep/runs/"
                    "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_"
                    "cpc_emb1000_enc01"),
    ),
    # Prior B=512 anchors. Same recipe family; bundled so the trajectory
    # lines up with the report's anchor rows in dim_usage.png.
    dict(
        arm="anchor_emb01",
        recipe="λ_e=0.1, λ_h=0.1 (#355 anchor)",
        prefix_rel=("reports/2026-06-20_lejepa_sigreg/runs/"
                    "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_"
                    "cpc"),
    ),
    dict(
        arm="anchor_emb10",
        recipe="λ_e=1.0, λ_h=0.1 (#359 anchor)",
        prefix_rel=("reports/2026-06-22_lejepa_sigreg_emb10/runs/"
                    "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_"
                    "cpc_emb10"),
    ),
]


# Periodic checkpoint suffixes (save-every=2500, total=12500 steps).
# train.py names them with ``step // 1000`` so 2500→_2k.pth, 5000→_5k.pth,
# 7500→_7k.pth, 10000→_10k.pth, 12500→_12k.pth. The step the suffix maps
# back to is taken from the companion ``_optimizer.pth`` if present;
# otherwise we fall back to the table below (the periodic-save call site
# in ``experiments/2026-04-27_freq-embedding/scripts/train.py`` line
# ~1545: ``f"{run_name}_{step // 1000}k.pth"``).
PERIODIC_SUFFIX_TO_STEP = {
    "2k": 2500,
    "5k": 5000,
    "7k": 7500,
    "10k": 10000,
    "12k": 12500,
}

NAMED_VARIANTS = ("best_loss", "best_gap", "final", "FINAL")


def _step_from_optimizer(ckpt_path: str) -> int | None:
    """Read ``step`` from the companion ``<ckpt>_optimizer.pth`` if present.

    Returns ``None`` if the companion file is missing or unreadable —
    callers fall back to filename-inference or ``""`` (N/A) in the CSV.
    Note: for ``_best_loss.pth`` / ``_best_gap.pth`` ``state["step"]`` is
    the step at which the best metric was reached (also recorded as
    ``best_loss_step`` / ``best_step``).
    """
    root, ext = os.path.splitext(ckpt_path)
    opt = f"{root}_optimizer{ext}"
    if not os.path.exists(opt):
        return None
    try:
        s = torch.load(opt, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [traj] warn: failed to read step from {opt}: {e}",
              file=sys.stderr)
        return None
    v = s.get("step")
    return int(v) if isinstance(v, (int, float)) else None


def enumerate_checkpoints(prefix_rel: str,
                          resolver=_resolve_checkpoint) -> list[dict]:
    """Return existing checkpoints for one arm in deterministic order.

    ``prefix_rel`` is the file stem relative to the repo root, with no
    trailing variant suffix (e.g.
    ``experiments/.../runs/bb_..._emb100_enc01``). For each candidate
    variant (periodic _Nk.pth + best_loss/best_gap/final/FINAL) we
    call ``resolver`` to find the first existing absolute path across
    the worktrees the single-checkpoint script also searches; missing
    variants are silently skipped.

    Each returned entry is ``{ckpt_kind, step, path}``:
    * ``ckpt_kind`` is one of ``step_2500``, ``step_5000``, ``step_7500``,
      ``step_10000``, ``step_12500``, ``best_loss``, ``best_gap``,
      ``final``, ``FINAL``.
    * ``step`` is the integer step at which the file was saved if it can
      be read from the companion ``_optimizer.pth``; otherwise the
      filename-inferred step for periodic variants; otherwise ``None``
      (recorded as ``""`` in the CSV for ``best_*`` when the optimizer
      file is missing).
    * ``path`` is the absolute checkpoint path.
    """
    out: list[dict] = []
    # Periodic — in step order.
    for suffix, fallback_step in PERIODIC_SUFFIX_TO_STEP.items():
        p = resolver(f"{prefix_rel}_{suffix}.pth")
        if p is None:
            continue
        step = _step_from_optimizer(p)
        if step is None:
            step = fallback_step
        out.append(dict(ckpt_kind=f"step_{fallback_step}",
                        step=step, path=p))
    # Named variants — order: best_loss, best_gap, final (lowercase), FINAL.
    for variant in NAMED_VARIANTS:
        p = resolver(f"{prefix_rel}_{variant}.pth")
        if p is None:
            continue
        step = _step_from_optimizer(p)
        out.append(dict(ckpt_kind=variant, step=step, path=p))
    return out


def collect_arm_checkpoints(arm: dict) -> list[dict]:
    """Resolve and enumerate this arm's checkpoints across worktrees.

    Returns ``[]`` if nothing matches — callers print a warning rather
    than aborting.
    """
    cks = enumerate_checkpoints(arm["prefix_rel"])
    if not cks:
        print(f"  [traj] {arm['arm']}: no checkpoints found for "
              f"{arm['prefix_rel']}_*.pth", file=sys.stderr)
    return cks


def eval_checkpoint(ckpt_path: str, x_dev, freq_ids, seas_ids,
                    device: str) -> tuple[float, float]:
    """Build the backbone matching ``ckpt_path`` and return
    ``(u_batchtime_h, u_batchtime_e)``."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    bb = build_backbone(sd)
    sd_strict = {k: v for k, v in sd.items()
                 if not k.startswith("cpc_w1")
                 and not k.startswith("teacher_")}
    bb.load_state_dict(sd_strict)
    bb.to(device)
    e_lat, h_lat = extract_e_and_h(bb, x_dev, freq_ids, seas_ids)
    u_bt_h = u_batchtime(h_lat).item()
    u_bt_e = u_batchtime(e_lat).item()
    del bb
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return u_bt_h, u_bt_e


def iter_rows(arms: Iterable[dict], x_dev, freq_ids, seas_ids,
              device: str) -> Iterable[dict]:
    """Yield CSV rows in (arm × checkpoint) order."""
    for arm in arms:
        cks = collect_arm_checkpoints(arm)
        if not cks:
            continue
        print(f"  [traj] {arm['arm']:14s}  {len(cks)} ckpt(s)")
        for ck in cks:
            try:
                u_bt_h, u_bt_e = eval_checkpoint(
                    ck["path"], x_dev, freq_ids, seas_ids, device)
            except Exception as e:
                print(f"  [traj] {arm['arm']} {ck['ckpt_kind']}: FAILED {e}",
                      file=sys.stderr)
                continue
            step_str = "" if ck["step"] is None else str(ck["step"])
            yield dict(
                arm=arm["arm"], recipe=arm["recipe"],
                step=step_str, ckpt_kind=ck["ckpt_kind"],
                backbone_ckpt=os.path.relpath(ck["path"], REPO_ROOT),
                u_batchtime=f"{u_bt_h:.6f}",
                u_batchtime_e=f"{u_bt_e:.6f}",
            )


def main() -> None:
    device = _pick_device()
    print(f"  [traj] device={device}")

    x_dev, freq_ids, seas_ids = load_held_out_batch(device)
    print(f"  [traj] batch shape={tuple(x_dev.shape)}  "
          f"freq_ids={tuple(freq_ids.shape) if freq_ids is not None else None}  "
          f"seas_ids={tuple(seas_ids.shape) if seas_ids is not None else None}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cols = ["arm", "recipe", "step", "ckpt_kind",
            "backbone_ckpt", "u_batchtime", "u_batchtime_e"]
    n_rows = 0
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in iter_rows(ARMS, x_dev, freq_ids, seas_ids, device):
            w.writerow(row)
            f.flush()
            n_rows += 1
            print(f"  [traj] {row['arm']:14s} {row['ckpt_kind']:11s} "
                  f"step={row['step']:>5s}  "
                  f"u_bt(h)={row['u_batchtime']}  "
                  f"u_bt(e)={row['u_batchtime_e']}")
    print(f"  [traj] wrote {CSV_OUT}  ({n_rows} rows)")


if __name__ == "__main__":
    main()
