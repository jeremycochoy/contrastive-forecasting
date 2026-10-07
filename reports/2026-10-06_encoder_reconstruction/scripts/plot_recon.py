"""#425: the GM-Relative MASE of the reconstruction head through training,
beside the forecast of the same checkpoints.

Each GM-Relative MASE graph of the #412 report that holds runs of ours gives
two figures here:

* ``recon_<graph>.png``: the reconstruction curves only.
* ``overlay_<graph>.png``: the forecast curves at 50% opacity, the
  reconstruction curves in the same colour at full opacity, and a thin
  vertical line from the forecast point to the reconstruction point of each
  checkpoint.

The graphs: the selected runs, all runs, ours with one patch size, and ours
with patch sizes. The Moirai graph has no run of ours. The groups, the
labels, the colours and the x scale are those of #412 (``plot_gm_rates.py``
and ``run_style.py``), so a run looks the same in every figure.

The report holds the figures only. So a figure keeps each fact in its plot
or in its legend, never in its title or in an annotation. The legend gives
the first and the last R score of each run, the floor of each scaling setup
with its score, and the key to the line styles.

Each reconstruction figure draws the floor of each scaling setup of its runs
as a thin grey line: the R score of a head that gives the normalised value 0
(floors.sh), so each value is the mean that normalised it. An R score
compares with the floor of its own setup. An overlay draws no floor: the
floors sit among the forecast curves, and a reader could take them for
forecast scores.

The y axis is logarithmic. The R scores and the forecast scores are about
ten times apart, and each graph shows the change of a score through
training, so a ratio keeps the same height at each level. No point leaves
the chart.

Reads results/recon_trajectories.tsv and results/forecast_425.tsv
(collect.py), results/floors.tsv (floors.sh), and #412's
results/gm_trajectories.tsv.
"""
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

STUDY = Path(__file__).resolve().parent.parent
BASE = STUDY.parent / "2026-09-06_moirai_small_size"
sys.path.insert(0, str(BASE / "scripts"))
import plot_gm_rates as base  # noqa: E402
from run_style import colour, line, tagged  # noqa: E402

FORECAST = [BASE / "results" / "gm_trajectories.tsv",
            STUDY / "results" / "forecast_425.tsv"]
RECON = STUDY / "results" / "recon_trajectories.tsv"
FLOORS = STUDY / "results" / "floors.tsv"
PLOTS = STUDY / "plots"
OURS = base.GROUPS[:2]
GRAPHS = {
    "selected": [(name, [run for run in runs if run[0] not in base.OFF_MAIN])
                 for name, runs in OURS],
    "all": OURS,
    "ours_one_patch_size": OURS[:1],
    "ours_patch_sizes": OURS[1:],
}
# The title of each graph: the runs it holds.
GRAPH_TITLES = {
    "selected": "selected runs",
    "all": "all runs",
    "ours_one_patch_size": "ours, one patch size",
    "ours_patch_sizes": "ours, patch sizes 8 to 128",
}
FORECAST_ALPHA = 0.5
LINK_WIDTH = 0.6
FLOOR_COLOUR = "0.55"
FLOOR_WIDTH = 0.8
# One line style for each floor, in the order of floors.tsv, so that a floor
# looks the same in every figure. The runs of ours draw solid lines.
FLOOR_STYLES = [(0, (6, 2, 1, 2)), (0, (1, 1.6)), (0, (5, 3))]
# The floor of each setup of floors.sh in the legend. A setup that is not
# here shows its floors.tsv label.
FLOOR_NAMES = {
    "ewma_zero_pad": "EWMA, new data with zero padding (BLK, OEF)",
    "ewma_old": "EWMA, old data",
    "meanstd": "mean/std",
}
KEY_COLOUR = "0.3"
# The y ticks of the log axis: those inside the range of a figure.
Y_TICKS = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7,
           1.0, 1.5, 2.0, 3.0, 5.0]
Y_PAD = 1.12   # the room above the top point and under the low point


def load(paths):
    """``{arm: {data seen: score}}`` from tables of arm, stop in thousands
    and score. A later table wins for a stop that two tables hold."""
    points = defaultdict(dict)
    for path in paths:
        if not Path(path).is_file():
            continue
        for row in open(path):
            arm, stop_k, score = row.split()
            seen = int(stop_k) * 1000 * base.XSCALE.get(arm, 1)
            points[arm][seen] = float(score)
    return points


def load_floors(path):
    """The floor of each scaling setup: ``[{setup, label, arms, score,
    style}]``, or [] when floors.sh has not run."""
    if not Path(path).is_file():
        return []
    rows = [line.rstrip("\n").split("\t") for line in open(path)][1:]
    return [{"setup": setup, "label": label, "arms": set(arms.split(",")),
             "score": float(score),
             "style": FLOOR_STYLES[i % len(FLOOR_STYLES)]}
            for i, (setup, label, arms, score) in enumerate(rows)]


def floors_of(arms, floors):
    """The floors of the setups of ``arms``."""
    return [floor for floor in floors if floor["arms"] & set(arms)]


def floor_legend(floor):
    """The legend label of a floor: its setup and its score."""
    name = FLOOR_NAMES.get(floor["setup"], floor["label"])
    return f"{name}: {floor['score']:.4f}"


def draw_floors(ax, floors):
    """Each floor as a thin grey line. Returns the legend entries."""
    entries = []
    for floor in sorted(floors, key=lambda floor: -floor["score"]):
        handle = ax.axhline(floor["score"], color=FLOOR_COLOUR, lw=FLOOR_WIDTH,
                            ls=floor.get("style", "--"), zorder=1)
        entries.append((handle, floor_legend(floor)))
    return entries


def draw_run(ax, arm, width, marker, points, alpha=1.0):
    """One run's line, or None when it has no point."""
    if not points:
        return None
    x = sorted(points)
    handle, = ax.plot(x, [points[v] for v in x], line(arm), color=colour(arm),
                      lw=width, marker=marker, ms=8, alpha=alpha, zorder=3)
    return handle


def draw_links(ax, arm, forecast, recon):
    """A thin vertical line at each checkpoint with both scores."""
    for seen in sorted(set(forecast) & set(recon)):
        ax.plot([seen, seen], [forecast[seen], recon[seen]],
                color=colour(arm), lw=LINK_WIDTH, zorder=2)


def run_label(arm, label, recon):
    """The legend label of a run: its code, its label, and its first and
    last R score."""
    scores = [recon[seen] for seen in sorted(recon)]
    if len(scores) == 1:
        numbers = f"R {scores[0]:.4f}"
    else:
        numbers = f"R {scores[0]:.4f} → {scores[-1]:.4f}"
    return f"{tagged(arm, label)}.  {numbers}"


def key_entries(overlay):
    """The key to the line styles: what R is, and in an overlay, the
    forecast and the line that joins the two scores of a checkpoint."""
    blank = Line2D([], [], linestyle="none")
    entries = [(Line2D([], [], color=KEY_COLOUR, lw=3, marker="o", ms=7),
                "R: the encoder reads the context and the true\n"
                "horizon. A head decodes the horizon from the\n"
                "encoder latents. One dot for each checkpoint."),
               (blank, "R a → b: R at the first and the last checkpoint")]
    if overlay:
        entries += [
            (Line2D([], [], color=KEY_COLOUR, lw=3, marker="o", ms=7,
                    alpha=FORECAST_ALPHA),
             "B4: the forecast of the same checkpoint"),
            (Line2D([], [], color=KEY_COLOUR, lw=0, marker="|", ms=16,
                    mew=LINK_WIDTH * 1.5),
             "The B4 score and the R score of one checkpoint"),
        ]
    return entries


def draw_legend(ax, columns, key):
    """The runs under the axes, one column per entry of ``columns``:
    ``(name, [(handle, label)])``, under the name in bold. The key at the
    right of the axes, under its name in bold. The figure has no tight
    layout, so neither legend takes room from the axes: the saved figure
    grows to hold them."""
    rows = 1 + max(len(entries) for _, entries in columns)
    blank = Line2D([], [], linestyle="none")
    handles, labels = [], []
    for name, entries in columns:
        column = [(blank, name)] + list(entries)
        column += [(blank, "")] * (rows - len(column))
        handles += [handle for handle, _ in column]
        labels += [label for _, label in column]
    # A figure legend: an axes legend that ax.add_artist keeps is clipped to
    # the axes, and this one sits under them.
    runs = ax.figure.legend(handles, labels, ncol=len(columns), fontsize=10,
                            loc="upper center", bbox_to_anchor=(0.5, -0.1),
                            bbox_transform=ax.transAxes, framealpha=0.94,
                            handlelength=3.2)
    for text in runs.get_texts()[::rows]:
        text.set_fontweight("bold")
    name, entries = key
    legend = ax.legend([blank] + [handle for handle, _ in entries],
                       [name] + [label for _, label in entries], fontsize=10,
                       loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       framealpha=0.94, handlelength=3.2)
    legend.get_texts()[0].set_fontweight("bold")
    return runs, legend


def style_axes(ax, title, low, high):
    ax.set_xscale("log")
    ticks = [40000, 100000, 200000, 400000, 665000, 1000000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["40k", "100k", "200k", "400k", "665k", "1,000k"])
    ax.set_xlim(left=28000)
    ax.set_xlabel("Data seen, in batch-64 steps (log scale). A step at batch "
                  "256 (OMB, OMF, OBM, OBW and OAL) counts 4.")
    ax.set_yscale("log")
    ax.set_ylim(low, high)
    ax.yaxis.set_major_locator(FixedLocator(
        [tick for tick in Y_TICKS if low <= tick <= high]))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("GM-Relative MASE, 97-config GIFT-Eval "
                  "(log scale, lower is better)")
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(alpha=0.3)


def draw_figure(groups, forecast, recon, out, overlay, floors=(), title=None):
    """One figure. Returns it, or None, and draws nothing, when no run of
    the groups has a reconstruction score."""
    # A run with no reconstruction score (ABC keeps no checkpoint) has no
    # pair to show, so its forecast stays in the #412 figures only.
    runs = [run for _, members in groups for run in members if recon.get(run[0])]
    if not runs:
        return None
    floors = [] if overlay else floors_of([arm for arm, *_ in runs], floors)
    fig, ax = plt.subplots(figsize=(12.5, 9.0))
    handles = {}
    for arm, _, width, marker in runs:
        if overlay:
            draw_run(ax, arm, width, marker, forecast.get(arm),
                     alpha=FORECAST_ALPHA)
            draw_links(ax, arm, forecast.get(arm, {}), recon.get(arm, {}))
        handles[arm] = draw_run(ax, arm, width, marker, recon.get(arm))
    values = [v for arm, *_ in runs for v in recon[arm].values()]
    if overlay:
        values += [v for arm, *_ in runs for v in forecast.get(arm, {}).values()]
    values += [floor["score"] for floor in floors]
    if title is None:
        title = ("Reconstruction (R) and forecast (B4)" if overlay
                 else "Reconstruction (R)")
    style_axes(ax, title, min(values) / Y_PAD, max(values) * Y_PAD)
    floor_entries = draw_floors(ax, floors)
    columns = [(name, [(handles[arm], run_label(arm, label, recon[arm]))
                       for arm, label, *_ in members if arm in handles])
               for name, members in groups]
    columns = [(name, entries) for name, entries in columns if entries]
    key = key_entries(overlay)
    if floor_entries:
        key += [(Line2D([], [], linestyle="none"),
                 "Floor: R of a head that outputs the mean")] + floor_entries
    draw_legend(ax, columns, ("How to read", key))
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return fig


def main():
    forecast, recon = load(FORECAST), load([RECON])
    floors = load_floors(FLOORS)
    PLOTS.mkdir(parents=True, exist_ok=True)
    for name, groups in GRAPHS.items():
        runs = GRAPH_TITLES[name]
        draw_figure(groups, forecast, recon, PLOTS / f"recon_{name}.png",
                    False, floors, f"Reconstruction (R): {runs}")
        draw_figure(groups, forecast, recon, PLOTS / f"overlay_{name}.png",
                    True, floors, f"Reconstruction (R) and forecast (B4): {runs}")


if __name__ == "__main__":
    main()
