#!/usr/bin/env python3
"""Offline latent-drift probe over a set of backbone checkpoints.

Runs the same :func:`src.metrics.drift_pair` decomposition the training
loop writes, but on already-saved snapshots. Reuses
:func:`src.checkpoint.load_backbone_from_checkpoint` so the state_dict
autodetects (freq / seasonality / encoder-layer count / QK-norm /
attn-out RMSNorm / CPC forecaster kind / learnable τ / patch-stats
width) match the head trainer's contract byte-for-byte.

Usage
-----
    python3 scripts/latent_drift.py \\
        --manifest manifest.csv \\
        --out drift.csv \\
        --d-model 384 --n-heads 6 --num-layers 6 \\
        --t-raw 4096 --n-channels 1 \\
        --rev-norm-kind ewma --rev-norm-span 128

Manifest CSV — one row per checkpoint::

    arm,step,path
    arm1,2000,/path/to/bb_..._2k.pth
    arm1,12500,/path/to/bb_..._final.pth
    arm1,25000,/path/to/bb_..._r2_final.pth
    ...

Output CSV — one row per (arm, step_a → step_b) comparison::

    arm,step_a,step_b,delta_step,kind,
    drift_cos,drift_cos_aligned,rot_gap,cka

``kind`` is ``"adjacent"`` (step_b vs the previous step within the arm)
or ``"vs_first"`` (step_b vs the arm's smallest step). Both kinds are
emitted for every non-first step, so callers can plot cumulative or
incremental curves without recomputing.

Probe batch: a fixed ARMA draw (``--seed``, default 20260722) shared
across ALL checkpoints in a single run — the metric is a comparison,
so both operands must see the same input.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import torch

# Make the src/ package importable when this script is run from a repo
# checkout without an installed package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.arma import generate_arma_batch
from src.checkpoint import load_backbone_from_checkpoint
from src.forecasting_head import extract_encoder_latents
from src.metrics import drift_pair


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--manifest", required=True,
                   help="CSV with columns arm,step,path (header row required).")
    p.add_argument("--out", required=True,
                   help="Output CSV path.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--probe-batch-size", type=int, default=64,
                   help="Probe batch size. Fixed ARMA draw.")
    p.add_argument("--seed", type=int, default=20260722,
                   help="Seed for the fixed ARMA probe batch. Every "
                        "checkpoint in this run sees the same batch.")
    # Backbone architecture — MUST match the checkpoints under test.
    # State-dict autodetect fills in what it can; these fields cannot
    # be inferred from the weights alone.
    p.add_argument("--t-raw", type=int, default=4096)
    p.add_argument("--n-channels", type=int, default=1)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--patch-width", type=int, default=16,
                   help="Patch width W. #374 arms use 16.")
    p.add_argument("--encoder-type", default="gru",
                   choices=["mlp", "mlp_wide", "residual_silu", "gru", "conv"])
    p.add_argument("--rev-norm-kind", default="ewma",
                   choices=["ewma", "revin", "none"])
    p.add_argument("--rev-norm-span", type=int, default=128)
    p.add_argument("--verbose", action="store_true",
                   help="Print each checkpoint's autodetected config.")
    return p.parse_args()


def _read_manifest(path):
    """Return {arm: [(step, path), …]} sorted by step within each arm."""
    entries = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        expected = {"arm", "step", "path"}
        missing = expected - set(reader.fieldnames or ())
        if missing:
            raise SystemExit(
                f"Manifest {path} is missing required columns: {sorted(missing)}")
        for row in reader:
            arm = row["arm"].strip()
            step = int(row["step"])
            ckpt = row["path"].strip()
            if not os.path.exists(ckpt):
                raise SystemExit(f"Manifest row references missing file: {ckpt}")
            entries[arm].append((step, ckpt))
    for arm in entries:
        entries[arm].sort(key=lambda kv: kv[0])
    return entries


def _generate_probe(batch_size, t_raw, n_channels, seed, device):
    x, _ = generate_arma_batch(
        batch_size=batch_size, T_raw=t_raw, C=n_channels,
        seed=seed, dimension=4)
    return x.to(device)


@torch.no_grad()
def _extract_h(backbone, probe_x):
    h, _ = extract_encoder_latents(backbone, probe_x)
    return h.detach().to(torch.float16).cpu()


def _row(arm, step_a, step_b, kind, m):
    return [
        arm, step_a, step_b, step_b - step_a, kind,
        f"{m['drift_cos'].item():.6f}",
        f"{m['drift_cos_aligned'].item():.6f}",
        f"{m['rot_gap'].item():.6f}",
        f"{m['cka'].item():.6f}",
    ]


def main():
    args = parse_args()
    device = torch.device(args.device)
    manifest = _read_manifest(args.manifest)
    print(f"[drift] manifest: {sum(len(v) for v in manifest.values())} "
          f"checkpoints across {len(manifest)} arm(s)")

    probe_x = _generate_probe(
        args.probe_batch_size, args.t_raw, args.n_channels,
        args.seed, device)
    print(f"[drift] probe: shape={tuple(probe_x.shape)}, "
          f"seed={args.seed}, batch_size={args.probe_batch_size}")

    load_kwargs = dict(
        C=args.n_channels, H=args.d_model, W=args.patch_width,
        nhead=args.n_heads, num_layers=args.num_layers,
        encoder_type=args.encoder_type,
        rev_norm_kind=args.rev_norm_kind,
        rev_norm_span=args.rev_norm_span,
        verbose=args.verbose,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "arm", "step_a", "step_b", "delta_step", "kind",
            "drift_cos", "drift_cos_aligned", "rot_gap", "cka",
        ])
        for arm, snaps in manifest.items():
            print(f"[drift] arm={arm}: {len(snaps)} snapshot(s)")
            first_h = None
            first_step = None
            prev_h = None
            prev_step = None
            for step, ckpt in snaps:
                backbone, _ = load_backbone_from_checkpoint(
                    ckpt, device, **load_kwargs)
                h = _extract_h(backbone, probe_x)
                del backbone
                torch.cuda.empty_cache() if device.type == "cuda" else None
                if first_h is None:
                    first_h = h
                    first_step = step
                    prev_h = h
                    prev_step = step
                    print(f"  [{arm}] initial h at step={step}")
                    continue
                cur = h.to(device).float()
                adj = drift_pair(prev_h.to(device).float(), cur)
                writer.writerow(_row(arm, prev_step, step, "adjacent", adj))
                if prev_step != first_step:
                    ini = drift_pair(first_h.to(device).float(), cur)
                    writer.writerow(_row(arm, first_step, step, "vs_first", ini))
                f.flush()
                print(f"  [{arm}] step {prev_step}→{step}: "
                      f"drift_cos={adj['drift_cos']:.4f}, "
                      f"aligned={adj['drift_cos_aligned']:.4f}, "
                      f"rot_gap={adj['rot_gap']:.4f}, "
                      f"cka={adj['cka']:.4f}")
                prev_h = h
                prev_step = step
    print(f"[drift] wrote {args.out}")


if __name__ == "__main__":
    main()
