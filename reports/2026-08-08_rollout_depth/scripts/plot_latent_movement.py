#!/usr/bin/env python3
"""#373 — latent movement between adjacent checkpoints, k = 3 against k = 0.

The parent's `latent_movement.png`, same measure and same fixed batch:

    movement_h = mean over (b, t, c) of  1 − cos(h_t(step_j), h_t(step_i))
    movement_e = mean over (b, t, c) of  1 − cos(e_t(step_j), e_t(step_i))

over adjacent periodic checkpoints of one run. The batch is #379's
committed `_latent_movement_batch.pt`, so these numbers sit on the scale of
both parent reports.

Why it is in this study: at k = 3 the f-bearing terms carry four times
their baseline weight against the f-free `L_rep` and SIGReg. If the encoder
starts moving faster, that ratio shift is the first place it shows.

Usage: plot_latent_movement.py --batch <_latent_movement_batch.pt> \\
           --out-csv results/latent_movement.csv \\
           --out plots/latent_movement.png \\
           --run <arm>:<k>:<run name>=<checkpoint dir> [--run ...]
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
sys.path.insert(0, str(HERE))

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

import cell_colours as cc                     # noqa: E402
import runs as R                              # noqa: E402                              # noqa: E402
from src.eval_latent_movement import (                 # noqa: E402
    compute_latents, load_backbone, mean_one_minus_cos, small_backbone_kwargs,
)

plt.rcParams.update(cc.rc())


def periodic(dirpath, name):
    """`[(step, path)]` for `<name>[_rN]_<k>k.pth`, sorted, no companions."""
    out = []
    for p in Path(dirpath).rglob(f"{name}*k.pth"):
        step = R.ckpt_step(name, p.name)
        if step is not None:
            out.append((step, p))
    return sorted(set(out))


def draw(rows, out):
    """The figure, from the rows the CSV carries.

    Colour is the ARM, not the cell: B5 trained two backbones and one hue at
    one lightness put two dashed curves on the panel with nothing to tell
    them apart. `arm_colour` keeps the hue and steps the lightness, and the
    retracted arm draws thin and says so in the key. The key sits under the
    figure, so it covers no curve.
    """
    fig, (axH, axE) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    for ax, key, ylab in ((axH, "movement_h", "1 − cos on h$_t$"),
                          (axE, "movement_e", "1 − cos on e$_t$")):
        for r in rows:
            ax.plot([r["from"], r["to"]], [float(r[key])] * 2,
                    color=cc.arm_colour(r["cell"]),
                    linestyle=cc.style(r["k"]),
                    linewidth=cc.width(r["cell"], base=2.6))
        ax.set_xlabel("backbone step interval")
        ax.set_ylabel(ylab)
    axH.set_title("Encoder-output latent")
    axE.set_title("Patch-embedding latent")
    cells = sorted({r["cell"] for r in rows},
                   key=lambda c: cc.ORDER.index(cc.cell_of(c))
                   if cc.cell_of(c) in cc.ORDER else 99)
    handles = [Line2D([], [], color=cc.arm_colour(c),
                      linewidth=cc.width(c, base=2.6),
                      label=cc.label(c) + ("  ✗ retracted"
                                           if cc.retracted(c) else ""))
               for c in cells]
    handles += [Line2D([], [], color=cc.INK_SOFT, linestyle=cc.style(kk),
                       label=f"k = {kk}, the rollout depth trained")
                for kk in (0, 3)]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("Latent movement between adjacent checkpoints", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out} ({len(rows)} interval(s))")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", default=[],
                   metavar="ARM:K:NAME=DIR")
    p.add_argument("--batch")
    p.add_argument("--from-csv", action="store_true",
                   help="redraw from --out-csv and load no checkpoint. The "
                        "numbers are committed; only re-deriving them needs "
                        "the checkpoint store.")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    if args.from_csv:
        with open(args.out_csv) as fh:
            rows = [dict(r, k=int(r["k"]), **{"from": int(r["from"]),
                                              "to": int(r["to"])})
                    for r in csv.DictReader(fh)]
        return draw(rows, args.out)

    if not args.run or not args.batch:
        raise SystemExit("ABORT: --run and --batch are required without "
                         "--from-csv")
    x = torch.load(args.batch, map_location="cpu", weights_only=False)
    kwargs = small_backbone_kwargs()
    # Class 0 (unknown) for both conditioning tables, so every checkpoint of
    # every cell sees identical conditioning. Both parents' latent-movement
    # scripts pass exactly this; `compute_latents` has no default and raises
    # without it.
    B = x.shape[0]
    freq_ids = torch.zeros(B, dtype=torch.long, device=args.device)
    seasonality_ids = torch.zeros(B, dtype=torch.long, device=args.device)
    rows = []

    for spec in args.run:
        head, dirpath = spec.split("=", 1)
        cell, ktxt, name = head.split(":")
        k = int(ktxt)
        # The run name comes in with the argument. Globbing the directory
        # for it does not work: group A's launcher puts every depth of a
        # cell, and its re-weighting control, into one `leg_40k` directory,
        # so a glob mixes four runs into one movement curve.
        cks = sorted(set(periodic(dirpath, name)))
        if len(cks) < 2:
            print(f"  skip {spec}: {len(cks)} periodic checkpoint(s), need 2",
                  file=sys.stderr)
            continue

        prev = None
        for step, path in cks:
            model = load_backbone(str(path), kwargs, args.device)
            h, e = compute_latents(model, x.to(args.device),
                                   freq_ids, seasonality_ids)
            if prev is not None:
                rows.append({
                    "cell": cell, "k": k,
                    "from": prev[0], "to": step,
                    "movement_h": f"{mean_one_minus_cos(prev[1], h):.6f}",
                    "movement_e": f"{mean_one_minus_cos(prev[2], e):.6f}"})
                print(f"{cell} k={k}  {prev[0]}->{step}  "
                      f"h={rows[-1]['movement_h']}  e={rows[-1]['movement_e']}")
            prev = (step, h, e)
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("ABORT: no run had two periodic checkpoints")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["cell", "k", "from", "to",
                                           "movement_h", "movement_e"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {args.out_csv}")
    return draw(rows, args.out)


if __name__ == "__main__":
    sys.exit(main())
