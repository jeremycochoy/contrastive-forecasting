#!/usr/bin/env python3
"""V6 (broken loss) vs V7 (fixed loss) backbone training comparison."""
import re
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt


PAT = re.compile(
    r"\[Step (\d+)\] loss=([\d.]+) \| train FF=([\d.\-]+) FP=([\d.\-]+) CB=([\d.\-]+) "
    r"\| val FF=([\d.\-]+) FP=([\d.\-]+) CB=([\d.\-]+) \| gap=([\d.\-]+)"
)


def parse(path):
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        m = PAT.search(line)
        if not m:
            continue
        step = int(m.group(1))
        rows.append({
            "step": step,
            "loss": float(m.group(2)),
            "train_ff": float(m.group(3)),
            "train_fp": float(m.group(4)),
            "train_cb": float(m.group(5)),
            "val_ff": float(m.group(6)),
            "val_fp": float(m.group(7)),
            "val_cb": float(m.group(8)),
            "gap": float(m.group(9)),
        })
    if not rows:
        return None
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6-log", default="experiments/contrastive-arma-correlation/logs/corrV6.log")
    ap.add_argument("--v7-log", default="experiments/contrastive-arma-correlation/logs/corrV7.log")
    ap.add_argument("--out", default="experiments/contrastive-arma-correlation/plots/v6_v7_compare.png")
    args = ap.parse_args()

    v6 = parse(args.v6_log)
    v7 = parse(args.v7_log)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    if v6 is not None:
        ax.plot(v6["step"], v6["loss"], label=f"V6 (broken loss)", lw=1.4, color="C3")
    if v7 is not None:
        ax.plot(v7["step"], v7["loss"], label=f"V7 (fixed loss)", lw=1.4, color="C0")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Backbone loss")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if v6 is not None:
        ax.plot(v6["step"], v6["val_cb"], label="V6 (broken)", lw=1.4, color="C3")
    if v7 is not None:
        ax.plot(v7["step"], v7["val_cb"], label="V7 (fixed)", lw=1.4, color="C0")
    ax.set_xlabel("step"); ax.set_ylabel("val CB"); ax.set_title("Cross-batch similarity (lower = more discriminative)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if v6 is not None:
        ax.plot(v6["step"], v6["val_ff"], label="V6 val FF", lw=1.4, color="C3")
        ax.plot(v6["step"], v6["val_fp"], label="V6 val FP", lw=1.0, ls="--", color="C3")
    if v7 is not None:
        ax.plot(v7["step"], v7["val_ff"], label="V7 val FF", lw=1.4, color="C0")
        ax.plot(v7["step"], v7["val_fp"], label="V7 val FP", lw=1.0, ls="--", color="C0")
    ax.set_xlabel("step"); ax.set_ylabel("FF / FP"); ax.set_title("Forecast vs Persistence")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if v6 is not None:
        ax.plot(v6["step"], v6["val_ff"] - v6["val_fp"], label="V6 gap", lw=1.4, color="C3")
    if v7 is not None:
        ax.plot(v7["step"], v7["val_ff"] - v7["val_fp"], label="V7 gap", lw=1.4, color="C0")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("step"); ax.set_ylabel("gap = FF - FP"); ax.set_title("Forecast gap")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("V6 (broken loss) vs V7 (fixed loss): cross-channel h×h_hat negative re-added",
                 fontsize=11, y=0.998)
    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out} (V6: {len(v6['step']) if v6 is not None else 0} pts, "
          f"V7: {len(v7['step']) if v7 is not None else 0} pts)")


if __name__ == "__main__":
    main()
