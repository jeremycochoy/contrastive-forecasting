#!/usr/bin/env python3
"""#373 figure 4 — how well the composed forecaster tracks the encoder.

On one fixed batch, roll the forecaster out on its own output for d = 1..D
tokens and compare each rolled latent with the true encoder latent d steps
ahead:

    fidelity(d) = mean over (b, c) of  cos(rollout_d, h_{T0+d})

`rollout_latent` is the eval's own operator, so this measures exactly what
GIFT-Eval composes, with no quantile head in the way. That is the point: a
head can mask a bad rollout, and the head is where every other number in
this study is read.

The batch is #379's committed `_latent_movement_batch.pt`, the same one the
two parent reports' latent-movement figures use, so a `k = 3` curve and a
`k = 0` curve are on one scale and comparable to the parents' diagnostics.

Usage:
  rollout_fidelity.py --out results/rollout_fidelity.csv \\
      --batch <repo>/reports/2026-07-21_split_pred_rep_small/plots/_latent_movement_batch.pt \\
      --run <label>=<checkpoint.pth> [--run ...]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.eval_latent_movement import (           # noqa: E402
    load_backbone, small_backbone_kwargs,
)
from src.forecasting_head import (               # noqa: E402
    extract_encoder_latents, rollout_latent,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True,
                   metavar="LABEL=CKPT",
                   help="repeatable; one curve per run")
    p.add_argument("--batch", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--horizon", type=int, default=16,
                   help="tokens to roll out; the card asks for d = 1..16")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def fidelity_curve(model, x: torch.Tensor, horizon: int,
                   device: str) -> list[float]:
    """`cos(rollout_d, h_{T0+d})` for d = 1..horizon, averaged over (b*c)."""
    # (B*C, T, H) encoder latents, the layout rollout_latent consumes.
    h = extract_encoder_latents(model, x.to(device))
    if isinstance(h, tuple):
        h = h[0]
    T = h.shape[1]
    if T <= horizon:
        raise SystemExit(f"ABORT: batch has {T} patch positions, "
                         f"need more than horizon={horizon}")
    # Cut the context so the last `horizon` true latents are the target. The
    # rollout never sees them.
    ctx, truth = h[:, :T - horizon, :], h[:, T - horizon:, :]
    rolled = rollout_latent(model, ctx, horizon)          # (B*C, horizon, H)
    return [float(F.cosine_similarity(rolled[:, d, :], truth[:, d, :],
                                      dim=-1).mean())
            for d in range(horizon)]


def main() -> int:
    args = parse_args()
    x = torch.load(args.batch, map_location="cpu", weights_only=False)
    kwargs = small_backbone_kwargs()

    rows = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"ABORT: --run wants LABEL=CKPT, got {spec!r}")
        label, ckpt = spec.split("=", 1)
        if not Path(ckpt).is_file():
            print(f"  skip {label}: no checkpoint at {ckpt}", file=sys.stderr)
            continue
        model = load_backbone(ckpt, kwargs, args.device)
        curve = fidelity_curve(model, x, args.horizon, args.device)
        for d, v in enumerate(curve, start=1):
            rows.append({"run": label, "d": d, "cos": f"{v:.6f}"})
        print(f"{label:<24} d=1 {curve[0]:.4f}  d=4 {curve[3]:.4f}  "
              f"d={args.horizon} {curve[-1]:.4f}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("ABORT: no run produced a curve")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "d", "cos"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
