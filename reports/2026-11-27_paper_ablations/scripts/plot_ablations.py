"""The figures of the paper ablations (#482): the GM-MASE of each row against
the reference, from the score files of the follower.

A score file holds one number: the GM-Relative MASE over the 97 GIFT-Eval
configurations of one stop of one run, with the control head of 30k steps
(``score_<run>_bb<k>k_h30k_control.txt``). The reference scores are those of
#425 (hp64/ff, read only). The ablation scores are in cf-482/ff.

Outputs (default: the results/ and plots/ folders of the report):
  results/ablation_gm_mase.csv     one line per row and stop, with the reference
  plots/ablation_gm_mase_166k.png  the GM-MASE of each row at 166k
  plots/ablation_gm_mase_steps.png the GM-MASE of each row against the steps

Usage: python3 plot_ablations.py [--ref-ff <folder>] [--ablation-ff <folder>] [--out <report folder>]
"""
import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
STOPS = [10, 25, 50, 75, 100, 125, 150, 166]
FINAL = 166
REF_FF = "/home/jupyter/cf_runs/hp64/ff"
ABLATION_FF = "/home/jupyter/cf_runs/cf-482/ff"
# The short label of each row, for the axes.
LABELS = {1: "No synthetic series", 2: "No crossfade series", 3: "No mixup",
          4: "Instance norm.", 5: "No time-scale emb.", 6: "TimesFM patch emb.",
          7: "No sphere", 8: "No convolution", 9: "No DropKey", 10: "No dropout",
          11: "No EMA teacher", 12: "No SIGReg on $e$", 13: "No SIGReg on $h$",
          14: r"No $\mathcal{L}_{\mathrm{rep}}$",
          15: r"$\mathcal{L}_{\mathrm{align}}$ at distance 1"}
ROW_HUE, REF_HUE, INK, GRID = "#257de0", "#8c8c8c", "#191919", "#e6e6e6"


def load_rows():
    """The ROWS table of make_runs.py: (row, short, part, ours, replaced, edits)."""
    spec = importlib.util.spec_from_file_location("make_runs", HERE / "make_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(ff, run, k):
    """The GM-MASE of a run at a stop, or None when its score file is not there."""
    path = Path(ff) / "results" / f"score_{run}_bb{k}k_h30k_control.txt"
    return float(path.read_text().split()[0]) if path.is_file() else None


def table(ref_ff, ablation_ff):
    """One dict per row and stop: the row, its labels, the stop, its GM-MASE
    (None when not scored yet) and the GM-MASE of the reference."""
    runs = load_rows()
    out = []
    for row, short, part, ours, replaced, _ in runs.ROWS:
        name = runs.run_name(row, short)
        for k in STOPS:
            out.append(dict(row=row, run=name, part=part, ours=ours,
                            replaced=replaced, stop_k=k,
                            gm_mase=score(ablation_ff, name, k),
                            reference=score(ref_ff, runs.REF, k)))
    return out


def write_csv(lines, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(lines[0]))
        writer.writeheader()
        for line in lines:
            writer.writerow({k: ("" if v is None else v) for k, v in line.items()})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def positions(rows):
    """The x of each row: one gap between two parts of the table."""
    xs, x, last = {}, 0.0, None
    for line in rows:
        if last is not None and line["part"] != last:
            x += 0.8
        xs[line["row"]], x, last = x, x + 1.0, line["part"]
    return xs


def plot_final(lines, out):
    """The GM-MASE of each row at 166k, with a stem from the reference line."""
    final = [line for line in lines if line["stop_k"] == FINAL]
    ref = final[0]["reference"]
    xs = positions(final)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    if ref is not None:
        ax.axhline(ref, color=REF_HUE, ls="--", lw=1.5, zorder=1)
        ax.text(max(xs.values()) + 1.4, ref, f"reference {ref:.4f}", color=INK,
                fontsize=9, ha="right", va="bottom")
    for line in final:
        x, value = xs[line["row"]], line["gm_mase"]
        if value is None:
            ax.text(x, ref if ref is not None else 1.0, "not\nscored\n", color=REF_HUE,
                    fontsize=7, ha="center", va="bottom")
            continue
        if ref is not None:
            ax.plot([x, x], [ref, value], color=ROW_HUE, lw=2, zorder=2)
        ax.plot(x, value, "o", color=ROW_HUE, ms=8, mec="white", mew=2, zorder=3)
    for part in dict.fromkeys(line["part"] for line in final):
        span = [xs[line["row"]] for line in final if line["part"] == part]
        ax.text(sum(span) / len(span), 1.02, part, transform=ax.get_xaxis_transform(),
                ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks([xs[line["row"]] for line in final])
    ax.set_xticklabels([LABELS[line["row"]] for line in final], rotation=35, ha="right")
    ax.set_ylabel("GM-MASE at 166k (lower is better)", color=INK)
    ax.set_xlim(min(xs.values()) - 0.6, max(xs.values()) + 1.6)
    style(ax)
    fig.suptitle("Each ablation at 166k steps against the reference", color=INK, y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_steps(lines, out):
    """The GM-MASE of each row against the steps, one panel per row, with the
    reference curve in each panel."""
    rows = list(dict.fromkeys(line["row"] for line in lines))
    values = [v for line in lines for v in (line["gm_mase"], line["reference"])
              if v is not None]
    low, high = (min(values), max(values)) if values else (0.9, 1.0)
    pad = 0.08 * (high - low or 0.05)
    fig, axes = plt.subplots(3, 5, figsize=(15, 8.4), sharex=True, sharey=True)
    for ax, row in zip(axes.flat, rows):
        mine = [line for line in lines if line["row"] == row]
        ref = [(line["stop_k"], line["reference"]) for line in mine
               if line["reference"] is not None]
        got = [(line["stop_k"], line["gm_mase"]) for line in mine
               if line["gm_mase"] is not None]
        if ref:
            ax.plot(*zip(*ref), color=REF_HUE, lw=2, marker="o", ms=5, label="reference")
        if got:
            ax.plot(*zip(*got), color=ROW_HUE, lw=2, marker="o", ms=6,
                    mec="white", mew=1.5, label="ablation")
        ax.set_title(f"{row}. {LABELS[row]}", color=INK, fontsize=10)
        ax.set_ylim(low - pad, high + pad)
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("steps (thousands)", color=INK)
        ax.set_xticks([10, 50, 100, 166])
    for ax in axes[:, 0]:
        ax.set_ylabel("GM-MASE", color=INK)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", frameon=False, ncol=2)
    fig.suptitle("GM-MASE of each ablation against the steps, with the reference",
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref-ff", default=REF_FF)
    p.add_argument("--ablation-ff", default=ABLATION_FF)
    p.add_argument("--out", type=Path, default=HERE.parent)
    a = p.parse_args(argv)
    lines = table(a.ref_ff, a.ablation_ff)
    write_csv(lines, a.out / "results" / "ablation_gm_mase.csv")
    (a.out / "plots").mkdir(parents=True, exist_ok=True)
    plot_final(lines, a.out / "plots" / "ablation_gm_mase_166k.png")
    plot_steps(lines, a.out / "plots" / "ablation_gm_mase_steps.png")
    scored = sum(line["gm_mase"] is not None for line in lines)
    print(f"{scored} of {len(lines)} row stops scored; wrote {a.out}/results and {a.out}/plots")


if __name__ == "__main__":
    main()
