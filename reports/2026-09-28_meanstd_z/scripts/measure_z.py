#!/usr/bin/env python3
"""How far the target part of a window lies from its context (#421).

The #421 split and scaler give each window one loc and one scale from its
context. This script reads real batches of the GiftEvalPretrain stream and
measures, per window, the max |z| over the observed values of the target
part, where z = (x - loc) / scale. It trains nothing.

Two views of the same batches:

    stream   the stream rows alone, as the stream gives them
    trainer  every row of the trainer's batch (stream rows, forked-arma
             rows, crossfade triplets), after the sign flip and the mixup
             of the #419 Moirai command (p = 0.3, alpha = 0.2)

Writes z_summary.json and z_quantiles.tsv, and z_per_window.tsv.gz.

Usage (CPU only, data reading only):
    HF_TOKEN=... CUDA_VISIBLE_DEVICES= python3 measure_z.py --batches 80 --out results
"""

import argparse
import collections
import gzip
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from src.dataloader import create_mixed_forked_arma_dataloader  # noqa: E402
from src.forecasting_head import mean_std_inputs  # noqa: E402
from src.freq_embedding import FREQ_NAMES_V2  # noqa: E402
from src.gift_pretrain import stream_factory  # noqa: E402
from src.norm import MEAN_STD_MINIMUM_SCALE, RevMeanStdNorm  # noqa: E402

SIZES = (8, 16, 32, 64, 128)
FLOOR = MEAN_STD_MINIMUM_SCALE ** 0.5
QUANTILES = (0.5, 0.9, 0.99, 0.999)
SHARES_ABOVE = (10, 20, 50, 100, 1000)


def scaler():
    """What `mean_std_inputs` reads of a model: the #421 normaliser with
    zero padding, and the base patch size."""
    return SimpleNamespace(rev_norm=RevMeanStdNorm(1, skip_leading_zeros=True),
                           W=16)


def window_rows(x, freq_ids):
    """Per window: (max |z| of the target part, scale, patch size, freq id).
    A window with no target (length < 2 P) gives NaN as its max |z|."""
    model = scaler()
    x_norm, sizes, target = mean_std_inputs(model, x, freq_ids, SIZES)
    real = target & ~model.rev_norm.pad_mask
    max_z = x_norm.abs().masked_fill(~real, 0.0).amax(dim=(1, 2))
    max_z[~real.any(dim=(1, 2))] = float("nan")
    return list(zip(max_z.tolist(), model.rev_norm.stdev.view(-1).tolist(),
                    sizes.tolist(), freq_ids.tolist()))


def mixup(x, rng):
    """The trainer's mixup on X at p = 0.3, alpha = 0.2 (maybe_mixup)."""
    if rng.random() >= 0.3:
        return x, False
    a = float(rng.beta(0.2, 0.2))
    perm = torch.from_numpy(rng.permutation(x.shape[0]))
    return a * x + (1 - a) * x[perm], True


def loader(args):
    """The trainer's batches of the #419 Moirai command, at batch 256."""
    real = stream_factory([args.seed, 0], 1, "v2", args.index, args.root)
    return create_mixed_forked_arma_dataloader(
        repo_id=None, batch_size=256, C=1, mix_ratio=0.0078125,
        crossfade_ratio=0.0, cross_triplets=1, path_in_repo=None,
        split="train", skip_rows=0, T_raw=4096, seed=args.seed + 10_000,
        emit_freq_ids=True, real_rows=real)


def summary(rows):
    """Quantiles of max |z| and the shares of the windows with a target."""
    z = np.array([r[0] for r in rows], dtype=np.float64)
    scale = np.array([r[1] for r in rows], dtype=np.float64)
    has = ~np.isnan(z)
    zt, st = z[has], scale[has]
    out = {"windows": int(len(z)), "with_target": int(has.sum()),
           "no_target_share": float(1 - has.mean()),
           "floor_share": float(np.mean(st <= FLOOR * 1.01)),
           "max": float(zt.max())}
    out.update({f"p{100 * q:g}": float(np.quantile(zt, q)) for q in QUANTILES})
    out.update({f"above_{t}": float(np.mean(zt > t)) for t in SHARES_ABOVE})
    return out


def by_class(rows, threshold):
    """Windows and the share above ``threshold``, per v2 frequency name."""
    groups = collections.defaultdict(list)
    for z, _, _, fid in rows:
        if not np.isnan(z):
            groups[FREQ_NAMES_V2[fid]].append(z)
    return {name: {"windows": len(v),
                   "above": float(np.mean(np.array(v) > threshold))}
            for name, v in sorted(groups.items())}


def measure(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    stream, trainer, mixed = [], [], []
    start = time.time()
    for i, (x, freq_ids, _) in enumerate(loader(args)):
        if i == args.batches:
            break
        stream += window_rows(x[:254], freq_ids[:254])
        x = x * torch.where(torch.rand(x.shape[0], 1, 1) < 0.5, 1.0, -1.0)
        x, was_mixed = mixup(x, rng)
        batch = window_rows(x, freq_ids)
        trainer += batch
        mixed += [was_mixed] * len(batch)
    return stream, trainer, mixed, time.time() - start


def write(args, stream, trainer, mixed, seconds):
    os.makedirs(args.out, exist_ok=True)
    unmixed = [r for r, m in zip(trainer, mixed) if not m]
    report = {"batches": args.batches, "seconds": round(seconds, 1),
              "floor": FLOOR, "stream": summary(stream),
              "trainer": summary(trainer),
              "trainer_unmixed": summary(unmixed),
              "trainer_mixed": summary([r for r, m in zip(trainer, mixed) if m]),
              "stream_by_class_above_100": by_class(stream, 100)}
    with open(os.path.join(args.out, "z_summary.json"), "w") as f:
        json.dump(report, f, indent=1)
    write_table(args.out, report)
    with gzip.open(os.path.join(args.out, "z_per_window.tsv.gz"), "wt") as f:
        f.write("view\tmax_z\tscale\tpatch_size\tfreq\tmixed\n")
        for r in stream:
            f.write(f"stream\t{r[0]:.6g}\t{r[1]:.6g}\t{r[2]}\t{FREQ_NAMES_V2[r[3]]}\t0\n")
        for r, m in zip(trainer, mixed):
            f.write(f"trainer\t{r[0]:.6g}\t{r[1]:.6g}\t{r[2]}\t{FREQ_NAMES_V2[r[3]]}\t{int(m)}\n")
    return report


def write_table(out, report):
    """One row per view: the quantiles and the shares."""
    keys = (["windows", "with_target", "no_target_share", "floor_share"]
            + [f"p{100 * q:g}" for q in QUANTILES] + ["max"]
            + [f"above_{t}" for t in SHARES_ABOVE])
    views = ("stream", "trainer", "trainer_unmixed", "trainer_mixed")
    with open(os.path.join(out, "z_quantiles.tsv"), "w") as f:
        f.write("view\t" + "\t".join(keys) + "\n")
        for v in views:
            f.write(v + "\t" + "\t".join(f"{report[v][k]:.6g}" for k in keys)
                    + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--batches", type=int, default=80)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--out", default=os.path.join(HERE, "..", "results"))
    p.add_argument("--index", default=None,
                   help="The record-batch index (default: the shipped one).")
    p.add_argument("--root", default=None,
                   help="A local copy of the dataset to read instead of "
                        "Hugging Face (tests).")
    args = p.parse_args()
    report = write(args, *measure(args))
    print(json.dumps({k: report[k] for k in
                      ("stream", "trainer", "trainer_unmixed", "trainer_mixed")},
                     indent=1))


if __name__ == "__main__":
    main()
