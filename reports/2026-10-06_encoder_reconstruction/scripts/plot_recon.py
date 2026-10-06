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

Reads results/recon_trajectories.tsv and results/forecast_425.tsv
(collect.py), and #412's results/gm_trajectories.tsv.
"""
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parent.parent
BASE = STUDY.parent / "2026-09-06_moirai_small_size"
sys.path.insert(0, str(BASE / "scripts"))
import plot_gm_rates as base  # noqa: E402
from run_style import colour, line  # noqa: E402

FORECAST = [BASE / "results" / "gm_trajectories.tsv",
            STUDY / "results" / "forecast_425.tsv"]
RECON = STUDY / "results" / "recon_trajectories.tsv"
PLOTS = STUDY / "plots"
OURS = base.GROUPS[:2]
GRAPHS = {
    "selected": [(name, [run for run in runs if run[0] not in base.OFF_MAIN])
                 for name, runs in OURS],
    "all": OURS,
    "ours_one_patch_size": OURS[:1],
    "ours_patch_sizes": OURS[1:],
}
YMAX = base.YMAX   # a forecast point above it sits on the top edge
FORECAST_ALPHA = 0.5
LINK_WIDTH = 0.6


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


def draw_run(ax, arm, width, marker, points, alpha=1.0, clip=None):
    """One run's line, or None when it has no point."""
    if not points:
        return None
    x = sorted(points)
    y = [points[v] if clip is None else min(points[v], clip) for v in x]
    handle, = ax.plot(x, y, line(arm), color=colour(arm), lw=width,
                      marker=marker, ms=8, alpha=alpha, zorder=3)
    return handle


def draw_links(ax, arm, forecast, recon):
    """A thin vertical line at each checkpoint with both scores."""
    for seen in sorted(set(forecast) & set(recon)):
        ax.plot([seen, seen], [min(forecast[seen], YMAX), recon[seen]],
                color=colour(arm), lw=LINK_WIDTH, zorder=2)


def style_axes(ax, overlay, low, high):
    ax.set_xscale("log")
    ticks = [40000, 100000, 200000, 400000, 665000, 1000000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["40k", "100k", "200k", "400k", "665k", "1,000k"])
    ax.set_xlim(left=28000)
    ax.set_xlabel("Data seen, in batch-64 steps (log scale). A step at batch "
                  "256 (OMB, OMF, OBM, OBW and OAL) counts 4.")
    ax.set_ylabel("GM-Relative MASE, 97-config GIFT-Eval (lower is better)")
    what = ("Faint lines: the forecast (B4). Solid lines: the reconstruction. "
            "A thin line joins the two scores of one checkpoint.\n"
            f"A forecast score above {YMAX} sits on the top edge."
            if overlay else "The reconstruction only.")
    ax.set_title("The encoder through training: the GM-Relative MASE of a head "
                 "that reconstructs the true horizon from its encoder latents.\n"
                 "The encoder reads the context and the true horizon. No "
                 "forecast. 1,024 context values, the B4 head settings.\n" + what,
                 pad=14)
    ax.grid(alpha=0.3)
    ax.set_ylim(low, high)


def draw_figure(groups, forecast, recon, out, overlay):
    """One figure. Returns False, and draws nothing, when no run of the
    groups has a reconstruction score."""
    # A run with no reconstruction score (ABC keeps no checkpoint) has no
    # pair to show, so its forecast stays in the #412 figures only.
    runs = [run for _, members in groups for run in members if recon.get(run[0])]
    if not runs:
        return False
    fig, ax = plt.subplots(figsize=(12.5, 9.0))
    handles = {}
    for arm, _, width, marker in runs:
        if overlay:
            draw_run(ax, arm, width, marker, forecast.get(arm),
                     alpha=FORECAST_ALPHA, clip=YMAX)
            draw_links(ax, arm, forecast.get(arm, {}), recon.get(arm, {}))
        handle = draw_run(ax, arm, width, marker, recon.get(arm))
        if handle is not None:
            handles[arm] = handle
    values = [v for arm, *_ in runs for v in recon.get(arm, {}).values()]
    if overlay:
        values += [min(v, YMAX) for arm, *_ in runs
                   for v in forecast.get(arm, {}).values()]
    span = max(values) - min(values) or 0.1
    style_axes(ax, overlay, max(0.0, min(values) - 0.08 * span),
               max(values) + 0.08 * span)
    base.draw_legend(ax, groups, handles)
    fig.tight_layout()
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return True


def main():
    forecast, recon = load(FORECAST), load([RECON])
    PLOTS.mkdir(parents=True, exist_ok=True)
    for name, groups in GRAPHS.items():
        draw_figure(groups, forecast, recon, PLOTS / f"recon_{name}.png", False)
        draw_figure(groups, forecast, recon, PLOTS / f"overlay_{name}.png", True)


if __name__ == "__main__":
    main()
