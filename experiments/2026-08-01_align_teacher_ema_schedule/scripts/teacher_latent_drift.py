#!/usr/bin/env python3
"""Post-hoc h_t drift on student *and* EMA-teacher latents (#388).

Extends the #382 probe (PR #387) to two experiments at once: the eight
#382 arms and the four #388 runs. Every curve therefore comes from one
pipeline, one probe batch and one cadence, so a #382 arm and a #388 arm
can be drawn on the same axes.

For each run it walks the saved 5000-step checkpoints, runs the fixed
ARMA probe batch through the student encoder and (when the checkpoint
carries teacher weights) through the EMA teacher, and writes the
:func:`src.metrics.drift_pair` decomposition for

  * ``adjacent``   — against the previous checkpoint,
  * ``vs_initial`` — against the run's first checkpoint.

Output: ``drift.csv``

  run, arm, alpha, latent, kind, step, step_ref, delta_step,
  drift_cos, drift_cos_aligned, rot_gap, cka

``run`` is the training run; ``arm`` its loss term; ``alpha`` the EMA
momentum schedule (``none`` for arms without a teacher, ``const_0.9``,
``sched_0.9_1.0``); ``latent`` is ``student_h`` or ``teacher_h``.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.arma import generate_arma_batch
from src.checkpoint import _detect_backbone_config
from src.forecasting_head import (extract_encoder_latents,
                                  extract_teacher_encoder_latents)
from src.metrics import drift_pair
from src.models import ConfigurableModel

# (run, arm, alpha, source, checkpoint subdirectory, checkpoint prefix).
# `source` selects which --runs-* root the subdirectory lives under.
RUNS = [
    ("pred",                "pred",         "none",          "r382", "pred",       "lti_pred"),
    ("rep",                 "rep",          "none",          "r382", "rep",        "lti_rep"),
    ("align",               "align",        "none",          "r382", "align",      "lti_align"),
    ("sigreg_e",            "sigreg_e",     "none",          "r382", "sigreg_e",   "lti_sigreg_e"),
    ("sigreg_h",            "sigreg_h",     "none",          "r382", "sigreg_h",   "lti_sigreg_h"),
    ("cpc",                 "cpc",          "none",          "r382", "cpc",        "lti_cpc"),
    ("pred_moco",           "pred_moco",    "const_0.9",     "r382", "pred_moco",  "lti_pred_moco"),
    ("rep_moco",            "rep_moco",     "const_0.9",     "r382", "rep_moco",   "lti_rep_moco"),
    ("align_teacher_a09",   "align_teacher", "const_0.9",    "r388", "align_teacher_a09",   "ats_align_teacher_a09"),
    ("align_teacher_sched", "align_teacher", "sched_0.9_1.0", "r388", "align_teacher_sched", "ats_align_teacher_sched"),
    ("pred_moco_sched",     "pred_moco",    "sched_0.9_1.0", "r388", "pred_moco_sched",     "ats_pred_moco_sched"),
    ("rep_moco_sched",      "rep_moco",     "sched_0.9_1.0", "r388", "rep_moco_sched",      "ats_rep_moco_sched"),
]

# Small #379 backbone — the fields a state_dict cannot disambiguate.
# Mirrors both experiments' run_arm.sh.
BASE_CFG = dict(C=1, H=64, W=16, nhead=8, num_layers=3,
                encoder_type="gru", ffn_mult=4.0, activation="gelu",
                depthwise_conv=3, dropout=0.1,
                rev_norm_kind="ewma", rev_norm_span=128)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--runs-382", required=True,
                   help="Root holding one subdirectory per #382 arm.")
    p.add_argument("--runs-388", required=True,
                   help="Root holding one subdirectory per #388 run.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--probe-batch-size", type=int, default=64,
                   help="Must match the training probe (run_arm.sh: 64).")
    p.add_argument("--t-raw", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20260722,
                   help="Training probe seed. One fixed ARMA draw shared "
                        "by every checkpoint of every run.")
    p.add_argument("--only", default=None,
                   help="Comma-separated run names; default all of them.")
    return p.parse_args()


def find_checkpoints(run_dir, prefix):
    """Return [(step, path), ...] sorted by step for `<prefix>_<N>k.pth`."""
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)k\.pth$")
    snaps = []
    if not os.path.isdir(run_dir):
        return snaps
    for name in os.listdir(run_dir):
        m = pat.match(name)
        if m:
            snaps.append((int(m.group(1)) * 1000, os.path.join(run_dir, name)))
    return sorted(snaps)


def load_model(path, device):
    """Build a ConfigurableModel matching `path` and load it whole.

    Unlike ``load_backbone_from_checkpoint`` this keeps the ``teacher_*``
    weights: the ema flags are switched on exactly when those keys are
    present, so ``load_state_dict`` stays strict.
    """
    sd = torch.load(path, map_location=device, weights_only=True)
    cfg = _detect_backbone_config(sd, BASE_CFG)
    cfg["ema_embedding"] = any(k.startswith("teacher_input_to_latent")
                               for k in sd)
    cfg["ema_encoder"] = any(k.startswith("teacher_encoder_layers")
                             for k in sd)
    cfg["cpc_infonce"] = any(k.startswith("cpc_w1") for k in sd)
    model = ConfigurableModel(**cfg).to(device).eval()
    model.load_state_dict(sd)
    for p in model.parameters():
        p.requires_grad = False
    return model, bool(cfg["ema_embedding"] or cfg["ema_encoder"])


@torch.no_grad()
def extract_latents(model, probe_x, has_teacher):
    """{latent_name: h} with h as (B*C, T, H) fp16 on CPU."""
    student, _ = extract_encoder_latents(model, probe_x)
    out = {"student_h": student.detach().to(torch.float16).cpu()}
    if has_teacher:
        teacher, _ = extract_teacher_encoder_latents(model, probe_x)
        out["teacher_h"] = teacher.detach().to(torch.float16).cpu()
    return out


def metrics_row(m):
    return [f"{m['drift_cos'].item():.6f}",
            f"{m['drift_cos_aligned'].item():.6f}",
            f"{m['rot_gap'].item():.6f}",
            f"{m['cka'].item():.6f}"]


def probe_run(writer, run_dir, prefix, run, arm, alpha, probe_x, device):
    snaps = find_checkpoints(run_dir, prefix)
    if not snaps:
        print(f"[{run}] no checkpoints under {run_dir} — skipped")
        return 0
    print(f"[{run}] {len(snaps)} checkpoints ({snaps[0][0]}…{snaps[-1][0]})")
    initial, prev, prev_step, n = {}, {}, None, 0
    for step, path in snaps:
        model, has_teacher = load_model(path, device)
        cur = extract_latents(model, probe_x, has_teacher)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for name, h in cur.items():
            if name not in initial:
                initial[name] = h
                continue
            hc = h.to(device).float()
            adj = drift_pair(prev[name].to(device).float(), hc)
            writer.writerow([run, arm, alpha, name, "adjacent", step,
                             prev_step, step - prev_step] + metrics_row(adj))
            ini = drift_pair(initial[name].to(device).float(), hc)
            writer.writerow([run, arm, alpha, name, "vs_initial", step,
                             snaps[0][0], step - snaps[0][0]]
                            + metrics_row(ini))
            n += 2
        prev, prev_step = cur, step
    return n


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    roots = {"r382": args.runs_382, "r388": args.runs_388}
    wanted = set(args.only.split(",")) if args.only else None

    probe_x, _ = generate_arma_batch(
        batch_size=args.probe_batch_size, T_raw=args.t_raw, C=1,
        seed=args.seed, dimension=4)
    probe_x = probe_x.to(device)
    print(f"[probe] shape={tuple(probe_x.shape)} seed={args.seed}")

    drift_path = os.path.join(args.out_dir, "drift.csv")
    total = 0
    with open(drift_path, "w", newline="") as fd:
        wd = csv.writer(fd)
        wd.writerow(["run", "arm", "alpha", "latent", "kind", "step",
                     "step_ref", "delta_step", "drift_cos",
                     "drift_cos_aligned", "rot_gap", "cka"])
        for run, arm, alpha, source, subdir, prefix in RUNS:
            if wanted and run not in wanted:
                continue
            total += probe_run(wd, os.path.join(roots[source], subdir),
                               prefix, run, arm, alpha, probe_x, device)
            fd.flush()
    print(f"[out] {drift_path} ({total} rows)")


if __name__ == "__main__":
    main()
