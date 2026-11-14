"""The width-64 GM-Relative MASE graph (#462): MPE at width 64 (this card),
MPE at width 384 (#421ew) and the width-64 runs of ~/cf_runs/hp64, at each
scored stop. Four panels: the GM over the 97 GIFT-Eval configs, then the GM
over the configs of each horizon.

A port of plot_hp64_scores.py, the graph of the hp64 session (11-14): the
runs, their cooldowns, the hues and the markers. The width-64 runs read the
score files and the per-config tables of hp64 (read only, control head, B4).
MPE and MPE at width 64 read their per-config tables (their own value heads,
A2V). The two MPE lines are the thick lines.

Usage: python3 plot_width64_gm_mase.py [out.png]
  HP64_FF=<folder> reads the width-64 runs from a copy with the layout of
  hp64/ff.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY.parent.parent))
from src.gift_scores import HORIZONS, gm_relative_mase, read_mase  # noqa: E402

FF = Path(os.environ.get("HP64_FF", "/home/jupyter/cf_runs/hp64/ff"))
SN = Path("/home/jupyter/workspaces/gift-eval/results/seasonal_naive/all_results.csv")
MPE_TABLES = STUDY.parent / "2026-09-06_moirai_small_size/results/per_config/radar"
NEW_TABLES = STUDY / "results/per_config"
OUT = STUDY / "plots/width64_gm_mase.png"
N_CONFIGS = 97
BASE = "g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere"
STOPS = [0, 10, 25, 35, 45, 50, 60, 75, 90, 100, 115, 125, 150, 166]  # 0: the 250-step save
MPE_STOPS = [10, 25, 50, 75, 100, 125, 150, 166]
LABELLED = [0.25, 10, 25, 50, 100, 166]
# The width-64 runs: legend name -> (name suffix, hue, marker), as in the hp64 graph
RUNS = {
    "base": ("", "#000000", "o"),
    "SIGReg-F": ("_sigf", "#1f77b4", "s"),
    "EMA-EXP": ("_emalog", "#ff7f0e", "^"),
    "10-L-ALIGN": ("_align10", "#2ca02c", "D"),
    "L2-B-NORM": ("_l2raw", "#9467bd", "v"),
    "red": ("_sigf_align10pw2", "#d62728", "*"),
    "red without SIGReg-F": ("_align10pw2", "#e377c2", "P"),
    "L2 run + SIGReg-F": ("_sigf_align10pw2_l2raw", "#17becf", "X"),
    "1/d run + SIGReg-F": ("_sigf_align10harm", "#8c564b", "h"),
    "L2 run": ("_align10pw2_l2raw", "#bcbd22", "X"),
    "1/d run": ("_align10harm", "#1b9e77", "h"),
}
# The cooldowns of the 5 runs of the search, from their 135k save:
# legend name -> (name suffix, line style, marker fill)
COOLDOWN_RUNS = ["base", "SIGReg-F", "EMA-EXP", "10-L-ALIGN", "L2-B-NORM"]
COOLDOWNS = {"exponential cooldown": ("_cd135", "--", "full"),
             "1-sqrt cooldown": ("_cdsqrt135", ":", "bottom")}
COOLDOWN_STOPS = [150, 166]
COOLDOWN_FROM = 125  # a cooldown line starts at the 125k point of its run
# legend name, hue, marker of the two MPE lines
MPE = ("MPE, width 384", "#000080", "*")
NEW = ("MPE, width 64", "#c51b7d", "o")
PANELS = [(None, "all 97 configs")] + [(h, f"{h} horizons") for h in HORIZONS]


def panel_scores(mase, naive):
    """{horizon or None: GM-Relative MASE} of one per-config table."""
    if len(mase) != N_CONFIGS:
        raise ValueError(f"{len(mase)} configs, not {N_CONFIGS}")
    return {h: gm_relative_mase(mase, naive, h) for h, _ in PANELS}


def hp64_scores(run, suffix, stops, naive):
    """{stop: panel scores} of a width-64 run, or of its cooldown ``suffix``,
    at each stop with a score file."""
    out = {}
    for stop in stops:
        name = f"{BASE}{RUNS[run][0]}_rollfix{suffix}_bb{stop}k_h30k_control"
        score = FF / "results" / f"score_{name}.txt"
        if score.is_file() and score.read_text().strip():
            mase = read_mase(FF / "heads/eval" / name / "gift/all_results.csv")
            out[stop] = panel_scores(mase, naive)
            assert abs(out[stop][None] - float(score.read_text().split()[0])) < 1e-4, name
    return out


def table_scores(pattern, naive):
    """{stop: panel scores} of the MPE stops whose per-config table exists."""
    paths = {stop: Path(pattern.format(stop)) for stop in MPE_STOPS}
    return {stop: panel_scores(read_mase(p), naive) for stop, p in paths.items()
            if p.is_file()}


def x_of(stop, offset):
    """x of a point: its stop, shifted a little for each run, so that the
    points of one stop stand side by side. 0 is the 250-step save."""
    return max(stop, 0.25) * offset


def draw_run(ax, key, points, offset, style):
    """One line: hollow markers for a width-64 run, filled for an MPE line."""
    if points:
        stops = sorted(points)
        ax.plot([x_of(s, offset) for s in stops], [points[s][key] for s in stops], **style)


def draw_cooldowns(ax, key, run, offset, scores, cooldowns):
    """The cooldowns of one run: filled or half-filled markers on a line from
    the 125k point of the run."""
    _, colour, marker = RUNS[run]
    for name, (_, ls, fill) in COOLDOWNS.items():
        points = cooldowns[name].get(run, {})
        if not points:
            continue
        start = [COOLDOWN_FROM] if COOLDOWN_FROM in scores else []
        merged = {**{s: scores[s] for s in start}, **points}
        draw_run(ax, key, merged, offset, dict(color=colour, ls=ls, lw=1.1, alpha=0.75))
        draw_run(ax, key, points, offset, dict(color=colour, marker=marker, ms=7, ls="none",
                                               fillstyle=fill, markerfacecoloralt="white"))


def draw_panel(ax, key, label, runs, cooldowns, mpe, new):
    for i, (run, (_, colour, marker)) in enumerate(RUNS.items()):
        offset = 0.93 + 0.014 * i
        draw_cooldowns(ax, key, run, offset, runs[run], cooldowns)
        draw_run(ax, key, runs[run], offset, dict(color=colour, marker=marker, mfc="none", ms=6,
                                                  lw=1.1, alpha=0.75, label=run))
    for name, (_, ls, fill) in COOLDOWNS.items():
        ax.plot([], [], color="grey", marker="o", ms=7, ls=ls, fillstyle=fill,
                markerfacecoloralt="white", label=name)
    for points, (name, colour, marker), size, ls in ((mpe, MPE, 12, "--"), (new, NEW, 8, "-")):
        draw_run(ax, key, points, 1.0, dict(color=colour, marker=marker, ms=size, lw=2.6, ls=ls,
                                            zorder=6, label=name))
    ax.axhline(1.0, color="grey", ls="--", lw=1, label="Seasonal Naive")
    set_axes(ax, label)


def set_axes(ax, label):
    ax.set_xscale("log")
    ax.set_xticks(LABELLED, [f"{s:g}" for s in LABELLED])
    ax.set_xticks([s for s in STOPS[1:] if s not in LABELLED], minor=True)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.2, 200)
    ax.grid(which="both", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Training step at batch 256 (thousands, log scale; 0.25: the 250-step save)")
    ax.set_ylabel(f"GM-Relative MASE, {label}")


def main(out=OUT):
    naive = read_mase(SN)
    runs = {run: hp64_scores(run, "", STOPS, naive) for run in RUNS}
    cooldowns = {name: {run: hp64_scores(run, suffix, COOLDOWN_STOPS, naive)
                        for run in COOLDOWN_RUNS}
                 for name, (suffix, _, _) in COOLDOWNS.items()}
    mpe = table_scores(str(MPE_TABLES / "moirai421ew_native{}k.csv"), naive)
    new = table_scores(str(NEW_TABLES / "cf462_mpe_w64_native{}k.csv"), naive)
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    for ax, (key, label) in zip(axes.flat, PANELS):
        draw_panel(ax, key, label, runs, cooldowns, mpe, new)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4,
               fontsize=10.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print("wrote", out, "| MPE at width 64, stops:", sorted(new) or "none yet")


if __name__ == "__main__":
    main(*sys.argv[1:])
