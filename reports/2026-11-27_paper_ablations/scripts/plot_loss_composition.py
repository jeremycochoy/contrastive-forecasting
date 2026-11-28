"""The loss composition of the paper ablations (#482): the weighted parts of
the training loss of the reference, of its second training (another seed) and
of each of the 16 rows, over the steps.

The trainer (experiments/2026-04-27_freq-embedding/scripts/train.py) logs
``loss`` as the sum of these parts, with the flags of the reference
(--align-loss-weight 10, --train-rollout-depth 3 --train-rollout-reduce sum,
SIGReg weights 1, CPC and align-MoCo weights 0):

  align_dist1      10 x l_align: L_align at distance 1 (rollout depth 0)
  align_dist2to4   L_align at distances 2 to 4 (depths 1 to 3), each at the
                   weight 10. It has no column: loss minus the other parts.
  sigreg_e         1 x sigreg_e: SIGReg on e
  sigreg_h         1 x sigreg_h: SIGReg on h
  rep              rep_w x l_rep: L_rep at the weight of the step (0.1)

A part that a row removes has an empty column (rows 12, 13 and 14). A run at
rollout depth 0 (row 15) has no cos_err_d1 column, thus no part for distances
2 to 4. A removed part gets no area and an empty cell in the table.

Inputs (read only): the losses CSVs of the reference (#425) in --ref-ckpt, and
of each row in <--runs-root>/<run>/ckpt/. Each resume writes a new leg (_r2,
_r3, ...). The legs go in the order of their first steps, because the box move
of the reference at 50k named its new leg r2 again. A later leg replaces the
rows of the earlier legs from its first step (src/loss_parts.py). The script
keeps only the columns of the parts.

Each value is the mean over BINS log-spaced bins of steps from 1k to 166k. A
partial last bin centres on its logged steps, so a run that trains draws up to
its last step. All the panels of a figure have one y range. The stack range is
the range of the parts and the total after the warmup (10k), with a margin. A
panel with a higher total in the warmup gives its peak.

Outputs (default: the results/ and plots/ folders of the report):
  results/loss_parts_binned.tsv   the binned parts of each run
  plots/loss_composition.png      the stacked parts and the total of each run
  plots/loss_small_terms.png      the total and the two SIGReg parts, log y

Usage: python3 plot_loss_composition.py [--ref-ckpt <folder>]
           [--runs-root <folder>] [--out <report folder>]
"""
import argparse
import csv
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from src.loss_parts import log_binned_means, read_loss_legs  # noqa: E402


def load_script(name):
    """A script of this folder as a module, with a name that no other module
    has."""
    spec = importlib.util.spec_from_file_location(f"{name}_482_loss",
                                                  HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ABLATIONS = load_script("plot_ablations")
RUNS = ABLATIONS.load_rows()
REF, ROWS, run_name = RUNS.REF, RUNS.ROWS, RUNS.run_name
LABELS, INK, style = ABLATIONS.LABELS, ABLATIONS.INK, ABLATIONS.style
REF2 = ABLATIONS.REF2

REF_CKPT = Path("/home/jupyter/cf_runs/hp64") / REF / "ckpt"
RUNS_ROOT = Path("/home/jupyter/cf_runs/cf-482")
FIRST, LAST, BINS = 1000, 166_000, 60
WARMUP = 10_000  # the warmup of the learning rate: the stack range omits it
TICKS_K = [1, 10, 25, 50, 166]  # 100k and 166k touch in a small panel
COLUMNS = 4  # the panels in a row of a figure
# part -> (column, weight): the weight is a number, or the column of the
# weight of the step.
LOGGED = {"align_dist1": ("l_align", 10.0), "sigreg_e": ("sigreg_e", 1.0),
          "sigreg_h": ("sigreg_h", 1.0), "rep": ("l_rep", "rep_w")}
REST = "align_dist2to4"  # no column: loss minus the logged parts
# part -> (hue, legend label), from the bottom of the stack up. The hues pass
# the color checks of adjacent parts, also when a row removes one part.
PARTS = {
    "align_dist1": ("#257de0", r"$10\,\mathcal{L}_{\mathrm{align}}$, distance 1"),
    "align_dist2to4": ("#f2a600",
                       r"$10\,\mathcal{L}_{\mathrm{align}}$, distances 2 to 4"),
    "sigreg_e": ("#008a17", "SIGReg on $e$"),
    "sigreg_h": ("#6a4ce2", "SIGReg on $h$"),
    "rep": ("#ee7fac", r"$0.1\,\mathcal{L}_{\mathrm{rep}}$"),
}
SMALL_TERMS = ["sigreg_e", "sigreg_h"]
X_LABEL = "Training step (batch 256, log scale)"


def first_row(path):
    """The first row of a losses CSV as a dict, or None for a CSV with no row."""
    with open(path) as f:
        return next(csv.DictReader(f), None)


def loss_legs(folder, run):
    """The losses CSVs of a run (``<run>_losses.csv``, ``<run>_r<k>_losses.csv``)
    in the order of their first steps. A CSV with no row is not a leg."""
    name = re.compile(rf"{re.escape(run)}(_r\d+)?_losses\.csv")
    firsts = {p: first_row(p) for p in Path(folder).glob("*_losses.csv")
              if name.fullmatch(p.name)}
    legs = sorted((p for p, row in firsts.items() if row),
                  key=lambda p: int(firsts[p]["step"]))
    if not legs:
        raise SystemExit(f"no losses CSV of {run} in {folder}")
    return legs


def run_parts(row):
    """The parts that a run logs, from its first row, in the order of PARTS."""
    logged = {p for p, (column, _) in LOGGED.items() if row.get(column)}
    if "cos_err_d1" in row:
        logged.add(REST)
    return [p for p in PARTS if p in logged]


def read_columns(parts):
    """The columns of the CSV that the parts and the total read."""
    names = ["loss"]
    for part in parts:
        names += [x for x in LOGGED.get(part, ()) if isinstance(x, str)]
    return names


def weighted(values, parts):
    """{part: values}: each logged part times its weight, and REST as the loss
    minus the logged parts."""
    out = {}
    for part in (p for p in parts if p in LOGGED):
        column, weight = LOGGED[part]
        out[part] = values[column] * (values[weight] if isinstance(weight, str)
                                      else weight)
    if REST in parts:
        out[REST] = values["loss"] - sum(out.values())
    return out


def read_parts(legs):
    """``(steps, {"total": loss, part: values})`` of each logged step of the
    legs of a run."""
    parts = run_parts(first_row(legs[0]))
    steps, values = read_loss_legs(legs, read_columns(parts))
    return steps, {"total": values["loss"], **weighted(values, parts)}


def binned(steps, parts):
    """``(centres, means)``: the parts over BINS log-spaced bins of steps."""
    return log_binned_means(steps, parts, FIRST, LAST, BINS)


def largest_residual(parts):
    """The largest |total - sum of the parts| over the steps or the bins."""
    rest = parts["total"] - sum(v for k, v in parts.items() if k != "total")
    return float(np.nanmax(np.abs(rest), initial=0.0))


def stack_layers(means):
    """``[(part, low, high)]``: the signed stack of the parts in the order of
    PARTS. A positive value goes on top of the parts below it, a negative
    value below zero. A layer with no height is left out."""
    up = np.zeros(len(means["total"]))
    down, layers = up, []
    for part in (p for p in PARTS if p in means):
        value = np.nan_to_num(means[part])
        top, bottom = up + np.maximum(value, 0), down + np.minimum(value, 0)
        layers += [(part, up, top), (part, bottom, down)]
        up, down = top, bottom
    return [(p, low, high) for p, low, high in layers if np.any(high > low)]


def y_range(curves):
    """``(low, high)`` of the stack panels: the parts and the totals of all
    runs after the warmup, with a margin of 5 %."""
    lows, highs = [0.0], [0.0]
    for centres, means in curves.values():
        late = centres > WARMUP
        layers = stack_layers(means)
        highs += [np.nanmax(means["total"][late], initial=0.0)]
        highs += [high[late].max(initial=0.0) for _, _, high in layers]
        lows += [low[late].min(initial=0.0) for _, low, _ in layers]
    return 1.05 * min(lows), 1.05 * max(highs)


def draw_stack(ax, centres, means, top):
    """The signed stack of the parts of one run and the total as a line. A
    label gives the peak of the total when it is above ``top``."""
    for part, low, high in stack_layers(means):
        ax.fill_between(centres, low, high, color=PARTS[part][0], alpha=0.85,
                        lw=0.4, edgecolor="white", gid=part)
    ax.plot(centres, means["total"], color=INK, lw=1.2, gid="total")
    peak = int(np.nanargmax(means["total"])) if len(centres) else None
    if peak is not None and means["total"][peak] > top:
        ax.annotate(f"peak {means['total'][peak]:.1f}", (centres[peak], top),
                    xytext=(5, -3), textcoords="offset points", ha="left",
                    va="top", fontsize=8, color=INK,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.9))


def panel_grid(count):
    """A figure with one panel for each of ``count`` runs, COLUMNS panels in a
    row, with one x range and one y range, and the steps on a log scale."""
    rows = -(-count // COLUMNS)
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(3.7 * COLUMNS, 2.6 * rows + 1.2),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    ax = axes.flat[0]
    ax.set_xscale("log")
    ax.set_xlim(FIRST, LAST)
    ax.set_xticks([k * 1000 for k in TICKS_K], [f"{k}k" for k in TICKS_K])
    ax.minorticks_off()
    for ax in axes.flat:
        style(ax)
    return fig, axes


def finish(fig, handles, title, y_label, out):
    """One legend above the panels, with the title on its first line, and the
    axis labels. Writes the figure to ``out``."""
    legend = fig.legend(handles=handles, loc="outside upper center",
                        ncol=len(handles), frameon=False, fontsize=9.5,
                        title=title, title_fontsize=12)
    legend.get_title().set_color(INK)
    fig.supxlabel(X_LABEL, color=INK, fontsize=10)
    fig.supylabel(y_label, color=INK, fontsize=10)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_composition(curves, titles, out):
    """One panel for each run: the stacked parts and the total."""
    low, high = y_range(curves)
    fig, axes = panel_grid(len(curves))
    for ax, (run, (centres, means)) in zip(axes.flat, curves.items()):
        draw_stack(ax, centres, means, high)
        ax.set_title(titles[run], color=INK, fontsize=10)
    axes.flat[0].set_ylim(low, high)
    handles = [Line2D([], [], color=INK, lw=1.2, label="Total")]
    handles += [Patch(color=PARTS[p][0], alpha=0.85, label=PARTS[p][1])
                for p in reversed(PARTS)]
    finish(fig, handles, "Parts of the training loss",
           "Loss, mean of a bin of steps (no unit)", out)


def log_range(curves):
    """``(low, high)`` of the log y axis: all the positive values of the totals
    and of the SIGReg parts, with a margin."""
    values = np.concatenate([means[k] for _, means in curves.values()
                             for k in ["total", *SMALL_TERMS] if k in means])
    values = values[np.isfinite(values) & (values > 0)]
    return values.min() / 1.3, values.max() * 1.3


def plot_small_terms(curves, titles, out):
    """One panel for each run: the total and the two SIGReg parts, on a log y
    axis."""
    fig, axes = panel_grid(len(curves))
    for ax, (run, (centres, means)) in zip(axes.flat, curves.items()):
        ax.plot(centres, means["total"], color=INK, lw=1.2, gid="total")
        for part in (p for p in SMALL_TERMS if p in means):
            ax.plot(centres, means[part], color=PARTS[part][0], lw=1.6, gid=part)
        ax.set_title(titles[run], color=INK, fontsize=10)
    axes.flat[0].set_yscale("log")
    axes.flat[0].set_ylim(*log_range(curves))
    handles = [Line2D([], [], color=INK, lw=1.2, label="Total")]
    handles += [Line2D([], [], color=PARTS[p][0], lw=1.6, label=PARTS[p][1])
                for p in SMALL_TERMS]
    finish(fig, handles, "Total and SIGReg parts of the training loss",
           "Loss, mean of a bin of steps (no unit, log scale)", out)


def write_table(path, curves):
    """One line for each run and bin: the bin centre, the total and the parts.
    A part that a run removes has an empty cell."""
    names = ["total", *PARTS]
    with open(path, "w") as f:
        f.write("\t".join(["run", "bin_centre_step", *names]) + "\n")
        for run, (centres, means) in curves.items():
            for i, centre in enumerate(centres):
                cells = [f"{means[k][i]:.6f}" if k in means else "" for k in names]
                f.write("\t".join([run, f"{centre:.0f}", *cells]) + "\n")


def panels(ref_ckpt, runs_root):
    """``[(title, run, folder)]``: the reference, its second training, then
    the rows in the order of ROWS, with the labels of plot_ablations.py."""
    rows = [(f"{row}. {LABELS[row]}", run_name(row, short),
             Path(runs_root) / run_name(row, short) / "ckpt")
            for row, short, *_ in ROWS]
    return [("Reference, seed 1", REF, Path(ref_ckpt)),
            ("Reference, seed 2", REF2, Path(runs_root) / REF2 / "ckpt")] + rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref-ckpt", type=Path, default=REF_CKPT)
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--out", type=Path, default=HERE.parent)
    a = p.parse_args(argv)
    curves, titles = {}, {}
    for title, run, folder in panels(a.ref_ckpt, a.runs_root):
        steps, parts = read_parts(loss_legs(folder, run))
        curves[run], titles[run] = binned(steps, parts), title
        rest = (f", {REST} {parts[REST].min():.3g} to {parts[REST].max():.3g}"
                if REST in parts else "")
        print(f"{run}: last step {steps[-1]}, largest |total - parts| "
              f"{largest_residual(parts):.1e}{rest}")
    for sub in ("results", "plots"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    write_table(a.out / "results" / "loss_parts_binned.tsv", curves)
    plot_composition(curves, titles, a.out / "plots" / "loss_composition.png")
    plot_small_terms(curves, titles, a.out / "plots" / "loss_small_terms.png")
    print(f"wrote {a.out}/results/loss_parts_binned.tsv and 2 figures in {a.out}/plots")


if __name__ == "__main__":
    main()
