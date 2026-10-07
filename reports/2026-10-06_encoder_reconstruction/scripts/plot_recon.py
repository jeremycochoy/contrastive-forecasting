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

The y axis is logarithmic: each graph shows the change of a score through
training, so a ratio keeps the same height at each level. The range of an
axis holds its points between two ticks, so no point leaves the chart.

The R scores are far below the forecast scores. On one axis, each family of
curves is a flat band. So an overlay breaks its y axis: the forecast curves
have the top panel, the reconstruction curves have the bottom panel, and
each panel has the range of its points. The line of a checkpoint goes from
one panel to the other.

The floor of a scaling setup is the R score of a head that gives the
normalised value 0 (floors.sh), so each value is the mean that normalised
it. An R score compares with the floor of its own setup. The legend gives
the floors. The plot does not draw them: a floor line takes the height of
the chart from the R curves.

Each R score comes from one head with one seed, so the figures cannot show
the noise between two heads. snapshot_score.sh scores an earlier snapshot of
some heads (the head of the best training loss). A hollow marker shows that
score under or above the dot of its checkpoint: the distance between the
two is the change of R in the last steps of one head training.

Reads results/recon_trajectories.tsv, results/forecast_425.tsv and
results/snapshots/scores.tsv (collect.py), results/floors.tsv (floors.sh),
and #412's results/gm_trajectories.tsv.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FixedLocator, NullLocator
from matplotlib.transforms import offset_copy

STUDY = Path(__file__).resolve().parent.parent
BASE = STUDY.parent / "2026-09-06_moirai_small_size"
sys.path.insert(0, str(BASE / "scripts"))
import plot_gm_rates as base  # noqa: E402
from run_style import CODE, colour, line, tagged  # noqa: E402

FORECAST = [BASE / "results" / "gm_trajectories.tsv",
            STUDY / "results" / "forecast_425.tsv"]
RECON = STUDY / "results" / "recon_trajectories.tsv"
FLOORS = STUDY / "results" / "floors.tsv"
SNAPSHOTS = STUDY / "results" / "snapshots" / "scores.tsv"
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
# The floor of each setup of floors.sh in the legend. A setup that is not
# here shows its floors.tsv label.
FLOOR_NAMES = {
    "ewma_zero_pad": "EWMA, new data with zero padding (BLK, OEF)",
    "ewma_old": "EWMA, old data",
    "meanstd": "mean/std",
}
KEY_COLOUR = "0.3"
METRIC = "GM-Relative MASE, 97-config GIFT-Eval (log scale, lower is better)"
# The ticks of a log y axis: in each decade, the values of one ladder. An
# axis takes the first ladder that gives it Y_MIN_TICKS ticks or more and
# that keeps the empty share of its height under Y_MAX_EMPTY. So a narrow
# range of scores still has ticks to read a point.
Y_LADDERS = [
    (1, 2, 5),
    (1, 1.5, 2, 3, 5, 7),
    (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8),
    (1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.2, 2.4, 2.6, 2.8,
     3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5),
]
Y_MIN_TICKS = 4
Y_MAX_EMPTY = 0.35
Y_MARGIN = 0.04   # room past the end ticks, as a share of the log range
# The labels of the two panels of an overlay, and the label of their common
# y axis at their left: x in widths of the axes.
PANEL_LABEL_X, METRIC_LABEL_X = -0.062, -0.092
LEGEND_DROP = 52  # points from the axes down to the legend of the runs


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
    """The floor of each scaling setup: ``[{setup, label, arms, score}]``,
    or [] when floors.sh has not run."""
    if not Path(path).is_file():
        return []
    rows = [line.rstrip("\n").split("\t") for line in open(path)][1:]
    return [{"setup": setup, "label": label, "arms": set(arms.split(",")),
             "score": float(score)} for setup, label, arms, score in rows]


def load_snapshots(path):
    """The R score of an earlier snapshot of a head:
    ``{arm: {data seen: (head step, score)}}``. The final head is the dot
    of the checkpoint, so its control score is not a snapshot here."""
    points = defaultdict(dict)
    if not Path(path).is_file():
        return points
    for row in list(open(path))[1:]:
        arm, stop_k, snapshot, step, score = row.split()
        if snapshot != "final":
            seen = int(stop_k) * 1000 * base.XSCALE.get(arm, 1)
            points[arm][seen] = (int(step), float(score))
    return points


def floors_of(arms, floors):
    """The floors of the setups of ``arms``."""
    return [floor for floor in floors if floor["arms"] & set(arms)]


def floor_legend(floor):
    """The legend label of a floor: its setup and its score."""
    name = FLOOR_NAMES.get(floor["setup"], floor["label"])
    return f"{name}: {floor['score']:.4f}"


def y_axis(values):
    """``(low, high, ticks)`` of a log axis that holds ``values``. The ticks
    go from the tick under the values to the tick above them, on the first
    ladder that gives Y_MIN_TICKS ticks or more and leaves no more than
    Y_MAX_EMPTY of that range empty. The limits add a small margin."""
    low, high = min(values), max(values)
    for ladder in Y_LADDERS:
        grid = sorted(m * 10.0 ** e for e in range(-5, 4) for m in ladder)
        under = max(t for t in grid if t <= low * (1 + 1e-9))
        above = min(t for t in grid if t >= high * (1 - 1e-9))
        ticks = [t for t in grid if under <= t <= above]
        empty = 1 - math.log(high / low) / math.log(above / under) \
            if above > under else 1.0
        if len(ticks) >= Y_MIN_TICKS and empty <= Y_MAX_EMPTY:
            break
    margin = (above / under) ** Y_MARGIN if above > under else 1.1
    return under / margin, above * margin, ticks


def draw_run(ax, arm, width, marker, points, alpha=1.0):
    """One run's line, or None when it has no point."""
    if not points:
        return None
    x = sorted(points)
    handle, = ax.plot(x, [points[v] for v in x], line(arm), color=colour(arm),
                      lw=width, marker=marker, ms=8, alpha=alpha, zorder=3)
    return handle


def draw_snapshots(ax, arm, marker, recon, snapshots):
    """A hollow marker at the R score of an earlier snapshot of a head, and
    a line from it to the dot of the final head. Both lie under the dot, so
    two equal scores show the dot."""
    for seen, (_, score) in snapshots.items():
        if seen not in recon:
            continue
        ax.plot([seen, seen], [score, recon[seen]], color=colour(arm),
                lw=1.4, zorder=2.5)
        ax.plot([seen], [score], linestyle="none", marker=marker, ms=8,
                mfc="white", mec=colour(arm), mew=1.6, zorder=2.6)


def draw_links(top, bottom, arm, forecast, recon):
    """A thin vertical line at each checkpoint with both scores, from its
    forecast point in the top panel to its R point in the bottom panel. It
    lies under the two panels, so no curve hides behind it."""
    for seen in sorted(set(forecast) & set(recon)):
        top.figure.add_artist(ConnectionPatch(
            (seen, forecast[seen]), (seen, recon[seen]),
            coordsA=top.transData, coordsB=bottom.transData,
            color=colour(arm), lw=LINK_WIDTH, zorder=-1))


def draw_break(top, bottom):
    """The marks of a break in the y axis between two panels."""
    top.spines["bottom"].set_visible(False)
    bottom.spines["top"].set_visible(False)
    top.tick_params(axis="x", which="both", bottom=False)
    marks = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=10, mew=1,
                 linestyle="none", color="k", clip_on=False)
    top.plot([0, 1], [0, 0], transform=top.transAxes, **marks)
    bottom.plot([0, 1], [1, 1], transform=bottom.transAxes, **marks)


def run_label(arm, label, recon):
    """The legend label of a run: its code, its label, and its first and
    last R score."""
    scores = [recon[seen] for seen in sorted(recon)]
    if len(scores) == 1:
        numbers = f"R {scores[0]:.4f}"
    else:
        numbers = f"R {scores[0]:.4f} → {scores[-1]:.4f}"
    return f"{tagged(arm, label)}.  {numbers}"


def key_entries(overlay, floors, snapshots=()):
    """The key: what R is, the earlier snapshots of a head, the floor of
    each scaling setup, and in an overlay, the forecast, the line that
    joins the two scores of a checkpoint and the break of the y axis.
    ``snapshots``: ``(run code, stop label, head step, score, score of
    the final head)`` of each."""
    blank = Line2D([], [], linestyle="none")
    entries = [(Line2D([], [], color=KEY_COLOUR, lw=3, marker="o", ms=7),
                "R: the encoder reads the context and the true\n"
                "horizon. A head decodes the horizon from the\n"
                "encoder latents. One dot for each checkpoint,\n"
                "one head with one seed for each dot."),
               (blank, "R a → b: R at the first and the last checkpoint")]
    if snapshots:
        text = "The same head at an earlier head step:\n"
        text += "\n".join(
            f"{code} {stop}: R {score:.4f} at step {step:,},\n"
            f"    {final:.4f} at step 30,000 (the dot)"
            for code, stop, step, score, final in snapshots)
        entries.append((Line2D([], [], color=KEY_COLOUR, lw=0, marker="o",
                               ms=7, mfc="white", mew=1.6), text))
    if overlay:
        entries += [
            (Line2D([], [], color=KEY_COLOUR, lw=3, marker="o", ms=7,
                    alpha=FORECAST_ALPHA),
             "B4, top panel: the forecast of the run"),
            (Line2D([], [], color=KEY_COLOUR, lw=0, marker="|", ms=16,
                    mew=LINK_WIDTH * 1.5),
             "The B4 score and the R score of one checkpoint"),
            (Line2D([], [], color="k", lw=0, marker=[(-1, -0.6), (1, 0.6)],
                    ms=10, mew=1),
             "A break in the y axis: each panel has\n"
             "the range of its scores"),
        ]
    if floors:
        entries.append((blank, "Floor of R: R of a head that gives the mean\n"
                               "of the scaling. It reads no latent."))
        entries += [(blank, "    " + floor_legend(floor)) for floor in
                    sorted(floors, key=lambda floor: -floor["score"])]
    return entries


def draw_legend(ax, key_ax, columns, key):
    """The runs under ``ax``, one column per entry of ``columns``:
    ``(name, [(handle, label)])``, under the name in bold. The key at the
    right of ``key_ax``, under its name in bold. The figure has no tight
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
    # the axes, and this one sits under them, at the same distance for each
    # height of the axes.
    under = offset_copy(ax.transAxes, fig=ax.figure, y=-LEGEND_DROP,
                        units="points")
    runs = ax.figure.legend(handles, labels, ncol=len(columns), fontsize=10,
                            loc="upper center", bbox_to_anchor=(0.5, 0.0),
                            bbox_transform=under, framealpha=0.94,
                            handlelength=3.2)
    for text in runs.get_texts()[::rows]:
        text.set_fontweight("bold")
    name, entries = key
    legend = key_ax.legend([blank] + [handle for handle, _ in entries],
                           [name] + [label for _, label in entries],
                           fontsize=10, loc="upper left",
                           bbox_to_anchor=(1.01, 1.0), framealpha=0.94,
                           handlelength=3.2)
    legend.get_texts()[0].set_fontweight("bold")
    return runs, legend


def style_x(ax):
    ax.set_xscale("log")
    ticks = [40000, 100000, 200000, 400000, 665000, 1000000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["40k", "100k", "200k", "400k", "665k", "1,000k"])
    ax.set_xlim(left=28000)
    ax.set_xlabel("Data seen, in batch-64 steps (log scale). A step at batch "
                  "256 (OMB, OMF, OBM, OBW and OAL) counts 4.")


def style_y(ax, values, label):
    """A log y axis that holds ``values`` between two ticks."""
    low, high, ticks = y_axis(values)
    ax.set_yscale("log")
    ax.set_ylim(low, high)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)


def stop_label(arm, seen):
    """The stop of a checkpoint as its run counts it: 1080k."""
    return f"{seen // (1000 * base.XSCALE.get(arm, 1))}k"


def draw_figure(groups, forecast, recon, out, overlay, floors=(), title=None,
                snapshots=None):
    """One figure. Returns it, or None, and draws nothing, when no run of
    the groups has a reconstruction score."""
    # A run with no reconstruction score (ABC keeps no checkpoint) has no
    # pair to show, so its forecast stays in the #412 figures only.
    runs = [run for _, members in groups for run in members if recon.get(run[0])]
    if not runs:
        return None
    arms = [arm for arm, *_ in runs]
    snapshots = snapshots or {}
    if overlay:
        fig, (top, ax) = plt.subplots(2, 1, sharex=True, figsize=(12.5, 11.5),
                                      gridspec_kw={"hspace": 0.06})
    else:
        fig, ax = plt.subplots(figsize=(12.5, 9.0))
        top = ax
    handles = {}
    for arm, _, width, marker in runs:
        if overlay:
            draw_run(top, arm, width, marker, forecast.get(arm),
                     alpha=FORECAST_ALPHA)
        handles[arm] = draw_run(ax, arm, width, marker, recon.get(arm))
        draw_snapshots(ax, arm, marker, recon[arm], snapshots.get(arm, {}))
    r_values = [v for arm in arms for v in recon[arm].values()]
    r_values += [score for arm in arms for seen, (_, score)
                 in snapshots.get(arm, {}).items() if seen in recon[arm]]
    style_x(ax)
    if overlay:
        style_y(top, [v for arm in arms for v in forecast.get(arm, {}).values()]
                or [1.0], "Forecast (B4)")
        style_y(ax, r_values, "Reconstruction (R)")
        # The two panel labels at one distance from the axes, and the label
        # of their common y axis at their left.
        for panel in (top, ax):
            panel.yaxis.set_label_coords(PANEL_LABEL_X, 0.5)
        box = ax.get_position()
        fig.supylabel(METRIC, x=box.x0 + METRIC_LABEL_X * box.width,
                      fontsize=10)
        draw_break(top, ax)
        # The lines of the checkpoints lie under the panels, so the panels
        # have no background of their own.
        top.patch.set_visible(False)
        ax.patch.set_visible(False)
        for arm in arms:
            draw_links(top, ax, arm, forecast.get(arm, {}), recon[arm])
    else:
        style_y(ax, r_values, METRIC)
    if title is None:
        title = ("Reconstruction (R) and forecast (B4)" if overlay
                 else "Reconstruction (R)")
    top.set_title(title, fontsize=13, pad=10)
    columns = [(name, [(handles[arm], run_label(arm, label, recon[arm]))
                       for arm, label, *_ in members if arm in handles])
               for name, members in groups]
    columns = [(name, entries) for name, entries in columns if entries]
    shown = [(CODE[arm], stop_label(arm, seen), step, score, recon[arm][seen])
             for arm in arms for seen, (step, score)
             in sorted(snapshots.get(arm, {}).items()) if seen in recon[arm]]
    key = key_entries(overlay, floors_of(arms, floors), shown)
    draw_legend(ax, top, columns, ("How to read", key))
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return fig


def main():
    forecast, recon = load(FORECAST), load([RECON])
    floors, snapshots = load_floors(FLOORS), load_snapshots(SNAPSHOTS)
    PLOTS.mkdir(parents=True, exist_ok=True)
    for name, groups in GRAPHS.items():
        runs = GRAPH_TITLES[name]
        draw_figure(groups, forecast, recon, PLOTS / f"recon_{name}.png",
                    False, floors, f"Reconstruction (R): {runs}", snapshots)
        draw_figure(groups, forecast, recon, PLOTS / f"overlay_{name}.png",
                    True, floors, f"Reconstruction (R) and forecast (B4): {runs}",
                    snapshots)


if __name__ == "__main__":
    main()
