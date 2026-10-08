"""#425: the GM-Relative MASE of the reconstruction head through training,
beside the forecast of the same checkpoints.

The two heads have separate figures (owner, 10-08): no figure mixes the
transformer head and the linear head. Each GM-Relative MASE graph of the
#412 report that holds runs of ours gives four figures here:

* ``recon_<graph>.png``: the R curves of the transformer head.
* ``overlay_<graph>.png``: the forecast curves at 50% opacity, the same R
  curves at full opacity, and a thin vertical line from the forecast point
  to the R point of each checkpoint.
* ``recon_<graph>_linear.png`` and ``overlay_<graph>_linear.png``: the same
  two figures for the linear head.

The graphs: the selected runs, all runs, ours with one patch size, ours
with patch sizes, and the Moirai group. Each graph draws every run of its
groups that has a score of its head, with no fixed count of runs: a new
score in the tables gives a new curve, and a graph with no scored run
gives no figure. The groups, the labels, the colours and the x scale are
those of #412 (``plot_gm_rates.py`` and ``run_style.py``), so a run looks
the same in every figure. The two versions of a graph share one y range
(``y_extra``), so a reader compares the two heads at the same height.

The report holds the figures only. So a figure keeps each fact in its plot
or in its legend, never in its title or in an annotation. The legend gives
the first and the last R score of each run with their ratio, the floor of
each scaling setup with its score, and the key to the line styles. Each
entry of the key ("How to read") is one short line.

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
each floor. The chart holds a floor, as a thin grey line, only when it is
in the y range of the chart: a floor that is far above the R curves takes
the height of the chart from them.

snapshot_score.sh scores an earlier snapshot of some transformer heads (the
head of the best training loss). A hollow marker shows that score under or
above the dot of its checkpoint: the distance between the two is the change
of R in the last steps of one head training. The hollow markers lie above
the curves, so the curve of no run hides one. For some heads, the best
training loss is at the last head step. That snapshot is the final head, so
its score equals the dot and measures no change of R: load_snapshots keeps
it out, and the figure holds no marker for it. The linear figures hold no
hollow marker: no snapshot of a linear head has a score. The legend holds
no list of the scores: results/snapshots/scores.tsv gives each one, with
its head step and the R of the final head.

Reads results/recon_trajectories.tsv, results/forecast_425.tsv,
results/snapshots/scores.tsv and results/recon_lin_trajectories.tsv
(collect.py), results/floors.tsv (floors.sh), and #412's
results/gm_trajectories.tsv.
"""
import csv
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
LINEAR = STUDY / "results" / "recon_lin_trajectories.tsv"
PLOTS = STUDY / "plots"
OURS = base.GROUPS[:2]
GRAPHS = {
    "selected": [(name, [run for run in runs if run[0] not in base.OFF_MAIN])
                 for name, runs in base.GROUPS],
    "all": base.GROUPS,
    "ours_one_patch_size": base.GROUPS[:1],
    "ours_patch_sizes": base.GROUPS[1:2],
    "moirai": base.GROUPS[2:3],
}
# The title of each graph: the runs it holds.
GRAPH_TITLES = {
    "selected": "selected runs",
    "all": "all runs",
    "ours_one_patch_size": "ours, one patch size",
    "ours_patch_sizes": "ours, patch sizes 8 to 128",
    "moirai": "Moirai, our copy",
}
# The two heads: the suffix of their files and their name in a title.
HEAD_NAMES = {"": "transformer head", "_linear": "linear head"}
# A legend label that would repeat a banned word gets a new one here.
RELABEL = {base.P + "_lr10xb": "lr 5.6e-5, old data, second run"}
# The steps of a head training. A best head of this step is the final head.
HEAD_STEPS = 30000
FORECAST_ALPHA = 0.5
LINK_WIDTH = 0.6
# The marker of a snapshot has no face and is larger than the dot (ms=8),
# so a snapshot near its dot is a ring around it, not a cover over it.
SNAPSHOT_MS = 13
# A run with one R score draws after the runs with a line, above them and
# with a thin white marker edge, so the line of no other run hides its dot.
LONE_DOT = dict(zorder=3.3, mec="white", mew=0.8)
FLOOR_COLOUR = "0.55"
FLOOR_WIDTH = 0.8
# One line style for each floor, in the order of floors.tsv, so that a floor
# looks the same in every figure. The runs of ours draw solid lines.
FLOOR_STYLES = [(0, (6, 2, 1, 2)), (0, (1, 1.6)), (0, (5, 3))]
# The chart holds a floor when an R score of its setup is this many times
# under it, or nearer.
FLOOR_REACH = 3.0
# The floor of each setup of floors.sh in the legend. A setup that is not
# here shows its floors.tsv label.
FLOOR_NAMES = {
    "ewma_zero_pad": "EWMA, new data (BLK, OEF)",
    "ewma_old": "EWMA, old data",
    "meanstd": "mean/std",
}
KEY_COLOUR = "0.3"
METRIC = "GM-Relative MASE, 97-config GIFT-Eval (log scale, lower is better)"
# The first line of the key: what R is, for each head.
R_LINES = {False: "R: a head decodes the true horizon from its latents",
           True: "R: one linear map decodes the horizon from latents"}
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
    """The floor of each scaling setup: ``[{setup, label, arms, score,
    style}]``, or [] when floors.sh has not run."""
    if not Path(path).is_file():
        return []
    rows = [line.rstrip("\n").split("\t") for line in open(path)][1:]
    return [{"setup": setup, "label": label, "arms": set(arms.split(",")),
             "score": float(score),
             "style": FLOOR_STYLES[i % len(FLOOR_STYLES)]}
            for i, (setup, label, arms, score) in enumerate(rows)]


def load_snapshots(path):
    """The R score of an earlier snapshot of a head: ``{arm: {data seen:
    (head step, score)}}``, from the columns of the table of collect.py. A
    ``final`` row is a control of the eval path, not a snapshot. A ``best``
    head of the last head step is the final head: its score equals the dot
    of its checkpoint and measures no change of R, so it stays out too."""
    points = defaultdict(dict)
    if not Path(path).is_file():
        return points
    for row in csv.DictReader(open(path), delimiter="\t"):
        if row["snapshot"] != "final" and int(row["head_step"]) < HEAD_STEPS:
            arm = row["arm"]
            seen = int(row["stop_k"]) * 1000 * base.XSCALE.get(arm, 1)
            points[arm][seen] = (int(row["head_step"]),
                                 float(row["r_snapshot"]))
    return points


def floors_of(arms, floors):
    """The floors of the setups of ``arms``."""
    return [floor for floor in floors if floor["arms"] & set(arms)]


def near_floors(floors, recon):
    """The floors that a chart must hold: those with an R score of a run of
    their setup FLOOR_REACH times under them, or nearer. ``recon``:
    ``{arm: [R score]}``, the scores of each run."""
    near = []
    for floor in floors:
        scores = [score for arm in floor["arms"]
                  for score in recon.get(arm, ())]
        if scores and max(scores) * FLOOR_REACH >= floor["score"]:
            near.append(floor)
    return near


def draw_floors(ax, floors):
    """Each floor in the y range of ``ax`` as a thin grey line. Returns
    ``{setup: line}`` of the floors it draws."""
    low, high = ax.get_ylim()
    return {floor["setup"]: ax.axhline(
                floor["score"], color=FLOOR_COLOUR, lw=FLOOR_WIDTH,
                ls=floor.get("style", "--"), zorder=1)
            for floor in floors if low <= floor["score"] <= high}


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


def draw_run(ax, arm, width, marker, points, alpha=1.0, **style):
    """One run's line, or None when it has no point. ``style``: LONE_DOT
    for a run with one R score."""
    if not points:
        return None
    x = sorted(points)
    handle, = ax.plot(x, [points[v] for v in x], line(arm), color=colour(arm),
                      lw=width, marker=marker, ms=8, alpha=alpha,
                      **{"zorder": 3, **style})
    return handle


def draw_snapshots(ax, arm, marker, recon, snapshots):
    """A hollow marker at the R score of an earlier snapshot of a head, and
    a line from it to the dot of the final head. Both lie above the curves,
    so the curve of no run hides a snapshot. The marker has no face and is
    larger than the dot: where the two scores are almost equal, it is a
    ring around the dot, and the dot stays in view."""
    for seen, (_, score) in snapshots.items():
        if seen not in recon:
            continue
        ax.plot([seen, seen], [score, recon[seen]], color=colour(arm),
                lw=1.4, zorder=3.4)
        ax.plot([seen], [score], linestyle="none", marker=marker,
                ms=SNAPSHOT_MS, mfc="none", mec=colour(arm), mew=1.6,
                zorder=3.5)


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
    """The legend label of a run: its code, its label, its first and last
    R score, and the ratio of the last to the first."""
    return f"{tagged(arm, RELABEL.get(arm, label))}.  {score_span(recon)}"


def score_span(points):
    """The first and the last R score of a curve, and their ratio."""
    scores = [points[seen] for seen in sorted(points)]
    if len(scores) == 1:
        return f"R {scores[0]:.4f}"
    return (f"R {scores[0]:.4f} → {scores[-1]:.4f}, "
            f"×{scores[-1] / scores[0]:.2f}")


def key_entries(overlay, floors, snapshots=False, floor_lines=None,
                linear=False):
    """The key: what R is, the earlier snapshot of a head, the floor of each
    scaling setup, and in an overlay, the forecast, the line that joins the
    two scores of a checkpoint and the break of the y axis. Each entry is
    one short line. ``snapshots``: the chart holds a hollow marker of an
    earlier head step. ``floor_lines``: ``{setup: line}`` of the floors
    that the chart draws. ``linear``: the chart shows the linear head."""
    blank = Line2D([], [], linestyle="none")
    entries = [(Line2D([], [], color=KEY_COLOUR, lw=3, marker="o", ms=7),
                R_LINES[bool(linear)]),
               (blank, "One dot: one checkpoint, one head"),
               (blank, "R a → b, ×c: first and last checkpoint, c = b / a")]
    if snapshots:
        entries.append((Line2D([], [], color=KEY_COLOUR, lw=0, marker="o",
                               ms=SNAPSHOT_MS, mfc="none", mew=1.6),
                        "The same head at an earlier head step"))
    if overlay:
        entries += [
            (Line2D([], [], color=KEY_COLOUR, lw=3, marker="o", ms=7,
                    alpha=FORECAST_ALPHA),
             "B4: the forecast of the run"),
            (Line2D([], [], color=KEY_COLOUR, lw=0, marker="|", ms=16,
                    mew=LINK_WIDTH * 1.5),
             "One checkpoint: its B4 and its R"),
            (Line2D([], [], color="k", lw=0, marker=[(-1, -0.6), (1, 0.6)],
                    ms=10, mew=1),
             "A break in the y axis"),
        ]
    if floors:
        floor_lines = floor_lines or {}
        entries.append((blank, "Floor of R: a head that reads no latent"))
        if len(floor_lines) < len(floors):
            entries.append((blank, "A floor with no line is above the chart"
                            if floor_lines else
                            "Each floor is above the chart"))
        entries += [(floor_lines.get(floor["setup"], blank),
                     floor_legend(floor)) for floor in
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


def style_x(ax, arms):
    """The x axis. The label names the batch-256 runs of the figure only,
    and drops that sentence when the figure holds none."""
    ax.set_xscale("log")
    ticks = [40000, 100000, 200000, 400000, 665000, 1000000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["40k", "100k", "200k", "400k", "665k", "1,000k"])
    ax.set_xlim(left=28000)
    label = "Data seen, in batch-64 steps (log scale)."
    codes = [CODE[arm] for arm in arms if base.XSCALE.get(arm, 1) == 4]
    if codes:
        listed = codes[0] if len(codes) == 1 else \
            ", ".join(codes[:-1]) + " and " + codes[-1]
        label += f" A step at batch 256 ({listed}) counts 4."
    ax.set_xlabel(label)


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


def draw_figure(groups, forecast, recon, out, overlay, floors=(), title=None,
                snapshots=None, linear=False, y_extra=()):
    """One figure of one head. Returns it, or None, and draws nothing, when
    no run of the groups has an R score. ``recon``: the R scores of that
    head. ``linear``: the head is the linear head, so the key names it and
    the chart holds no snapshot marker. ``y_extra``: scores of the other
    head of the same graph, so the two versions share one y range."""
    snapshots = {} if linear else (snapshots or {})
    runs = [run for _, members in groups for run in members
            if recon.get(run[0])]
    if not runs:
        return None
    arms = [arm for arm, *_ in runs]
    recon = {arm: recon[arm] for arm in arms}
    if overlay:
        fig, (top, ax) = plt.subplots(2, 1, sharex=True, figsize=(12.5, 11.5),
                                      gridspec_kw={"hspace": 0.06})
    else:
        fig, ax = plt.subplots(figsize=(12.5, 9.0))
        top = ax
    handles = {}
    # The runs with one R score draw last: their dot lies above the lines.
    runs.sort(key=lambda run: len(recon[run[0]]) == 1)
    for arm, _, width, marker in runs:
        if overlay:
            draw_run(top, arm, width, marker, forecast.get(arm),
                     alpha=FORECAST_ALPHA)
        lone = LONE_DOT if len(recon[arm]) == 1 else {}
        handles[arm] = draw_run(ax, arm, width, marker, recon[arm], **lone)
        draw_snapshots(ax, arm, marker, recon[arm], snapshots.get(arm, {}))
    floors = floors_of(arms, floors)
    r_scores = {arm: list(recon[arm].values()) for arm in arms}
    r_values = [v for arm in arms for v in r_scores[arm]]
    r_values += [score for arm in arms for seen, (_, score)
                 in snapshots.get(arm, {}).items() if seen in recon[arm]]
    r_values += [floor["score"] for floor in near_floors(floors, r_scores)]
    r_values += list(y_extra)
    style_x(ax, arms)
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
    columns = []
    for name, members in groups:
        entries = [(handles[arm], run_label(arm, label, recon[arm]))
                   for arm, label, *_ in members if arm in handles]
        if entries:
            columns.append((name, entries))
    shown = any(seen in recon[arm] for arm in arms
                for seen in snapshots.get(arm, {}))
    key = key_entries(overlay, floors, shown, draw_floors(ax, floors), linear)
    draw_legend(ax, top, columns, ("How to read", key))
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return fig


def graph_values(groups, scores, snapshots):
    """The R scores of one head in one graph, with the snapshot scores that
    its charts hold: the y range that this head asks for."""
    values = []
    for _, members in groups:
        for arm, *_ in members:
            points = scores.get(arm, {})
            values += list(points.values())
            values += [score for seen, (_, score)
                       in snapshots.get(arm, {}).items() if seen in points]
    return values


def main():
    forecast, floors = load(FORECAST), load_floors(FLOORS)
    snapshots = load_snapshots(SNAPSHOTS)
    heads = {"": (load([RECON]), snapshots, False),
             "_linear": (load([LINEAR]), {}, True)}
    PLOTS.mkdir(parents=True, exist_ok=True)
    for name, groups in GRAPHS.items():
        runs = GRAPH_TITLES[name]
        shared = [v for scores, snaps, _ in heads.values()
                  for v in graph_values(groups, scores, snaps)]
        for suffix, (scores, snaps, linear) in heads.items():
            head = HEAD_NAMES[suffix]
            draw_figure(groups, forecast, scores,
                        PLOTS / f"recon_{name}{suffix}.png", False, floors,
                        f"R, {head}: {runs}", snaps, linear, shared)
            draw_figure(groups, forecast, scores,
                        PLOTS / f"overlay_{name}{suffix}.png", True, floors,
                        f"R and B4, {head}: {runs}", snaps, linear, shared)


if __name__ == "__main__":
    main()
