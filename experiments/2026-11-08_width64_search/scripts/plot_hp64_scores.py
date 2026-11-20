"""The forecast scores of the width-64 search of 11-08 (base, SIGReg-F, EMA-EXP, 10-L-ALIGN, L2-B-NORM: the user's
codenames of A, B, C, D1, D2. Width 64, 8 heads, 1 encoder and 3
forecaster layers, the red recipe, 166k steps at batch 256). For each scored stop: the GM-Relative MASE over the
97 GIFT-Eval configs (the score file of the follower) and by horizon (short, medium, long: the geometric mean of
MASE[0.5] / the MASE[0.5] of Seasonal Naive over the configs of that horizon). Stop 0k is the 250-step save.
Since 11-12 also the new run of 11-11 (10-L-ALIGN with L_align at 1, 2, 4, 8, 16, 32 patches ahead, + SIGReg-F)
(since 11-13 also its variant of 11-12 without SIGReg-F. Since 11-14 the L2 run and the 1/d run, each red with one change)
and two cooldowns of each of the 5 runs (from the 135k save, scored at 150k and 166k): the exponential one (filled
markers) and the 1-sqrt one (half-filled markers). Since 11-18 also the cooldowns of the 1/d run, and a second
reference line: the MPE recipe at width 64 (cf#462).

Usage: python3 plot_hp64_scores.py <output png> [late]
  prints one markdown table per scored stop, then draws one panel per horizon (the GM first). 'late' draws only
  the stops from 90k on a linear step axis (where the cooldowns branch off their runs).
  HP64_FF=<folder> reads the scores from a test copy with the layout of hp64/ff
  HP64_RUNS=<code name>,... draws only these runs and their cooldowns
"""
import csv
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

FF = Path(os.environ.get("HP64_FF", os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64") + "/ff"))
SN = Path(os.environ.get("SEASONAL_NAIVE_CSV", "/home/jupyter/workspaces/gift-eval/results/seasonal_naive/all_results.csv"))
COL = "eval_metrics/MASE[0.5]"
BASE = "g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere"
STOPS = [0, 10, 25, 35, 45, 50, 60, 75, 90, 100, 115, 125, 150, 166]
HORIZONS = ["short", "medium", "long"]
# MPE (the user's "Moire"): our Moirai small, width 384, run cf421ew_moirai_native of the report
# 2026-09-06_moirai_small_size, 166k steps at batch 256 (the same step axis). Its per-config tables of each stop
MPE_TABLES = Path(os.environ.get("MPE_TABLES", Path(__file__).resolve().parents[3]
                                 / "reports/2026-09-06_moirai_small_size/results/per_config/radar"))
MPE_STOPS = [10, 25, 50, 75, 100, 125, 150, 166]
MPE_LABEL = "MPE: our Moirai small (width 384), 166k steps at batch 256"
# W64: the MPE recipe at width 64 (1 encoder and 3 forecaster layers, 8 heads), 166k steps at batch 256. Its own
# value heads score it. Its per-config tables of each stop come from the report of cf#462
W64_TABLES = Path(os.environ.get("W64_TABLES", Path(__file__).resolve().parents[3]
                                 / "reports/2026-11-14_moirai_width64/results/per_config"))
W64_SCORES = W64_TABLES.parent  # score_cf462_mpe_w64_bb<k>k_native.txt: the GM of the run, checked here
W64_LABEL = "MPE recipe at width 64 (1 encoder, 3 forecaster layers; cf#462), 166k steps at batch 256"
LABELLED = [0.25, 10, 25, 50, 100, 166]  # the other stops get a tick with no label
# codename -> (name suffix, legend label, colour, marker, x offset factor: the points of one stop stand side by side)
RUNS = {
    "base": ("", "base: the red recipe at width 64", "#000000", "o", 0.955),
    "SIGReg-F": ("_sigf", "SIGReg-F: + SIGReg on f (read before the sphere), weight 1", "#1f77b4", "s", 0.977),
    "EMA-EXP": ("_emalog", "EMA-EXP: EMA τ linear in ln(1 + step)", "#ff7f0e", "^", 1.0),
    "10-L-ALIGN": ("_align10", "10-L-ALIGN: L_align x 10 (depths 0 to 3)", "#2ca02c", "D", 1.023),
    "L2-B-NORM": ("_l2raw", "L2-B-NORM: mean (f - h)^2 of the raw vectors in place of L_align", "#9467bd", "v", 1.046),
    "red": ("_sigf_align10pw2", "red: L_align x 10 at 1, 2, 4, 8, 16, 32 patches ahead + SIGReg on f",
                "#d62728", "*", 1.069),
    "red without SIGReg-F": ("_align10pw2", "red without SIGReg-F: L_align x 10 at 1, 2, 4, 8, 16, 32 patches ahead",
                             "#e377c2", "P", 1.092),
    "L2 run + SIGReg-F": ("_sigf_align10pw2_l2raw", "L2 run + SIGReg-F: red with mean (f - h)^2 of the raw vectors "
                          "in place of the cosine", "#17becf", "X", 1.115),
    "1/d run + SIGReg-F": ("_sigf_align10harm", "1/d run + SIGReg-F (stopped at 20k): red with the cosine at all "
                           "distances d = 1 to 32, weight 1/d", "#8c564b", "h", 1.138),
    "L2 run": ("_align10pw2_l2raw", "L2 run: red without SIGReg-F, with mean (f - h)^2 of the raw vectors in place "
               "of the cosine", "#bcbd22", "X", 1.161),
    "1/d run": ("_align10harm", "1/d run: red without SIGReg-F, with the cosine at all distances d = 1 to 32, "
                "weight 1/d", "#7f7f7f", "h", 1.184),
}
# HP64_RUNS=<key>,<key>,... draws only these runs of RUNS and their cooldowns (a view of a few runs)
ONLY = [k for k in os.environ.get("HP64_RUNS", "").split(",") if k]
if ONLY:
    RUNS = {k: RUNS[k] for k in ONLY}
# The cooldowns of the 5 runs of the search and of the 1/d run: each run again from its 135k save with its optimizer,
# every flag kept, lr = 1e-3 x 0.5^((step - 135k) / 5k) (exponential, MiniCPM. Folder and tag suffix _cd135) or
# lr = 1e-3 x (1 - sqrt((step - 135k) / 31k)) (1-sqrt, Hägele and others 2024. Suffix _cdsqrt135)
CD_RUNS = ["base", "SIGReg-F", "EMA-EXP", "10-L-ALIGN", "L2-B-NORM", "1/d run"]  # the 1/d run: since 11-18
CD_STOPS = [150, 166]
CD_FROM = 125  # the last scored stop before the 135k save: there the cooldown is still the same model as its run
CD_START = 135  # the save each cooldown starts from
LATE_FROM = 90  # the late view (second argument 'late'): the stops from 90k on a linear step axis
CD_LABEL = ("filled markers on a dashed line: the exponential cooldown of the run\n(the same run to its 135k save, "
            "then lr 1e-3 x 0.5^((step - 135k) / 5k); the line starts at the run's 125k point)\na larger open marker "
            "around a cooldown marker: the run itself, at almost the same score")
SQRT_LABEL = ("half-filled markers on a dotted line: the 1-sqrt cooldown of the run\n(the same run to its 135k save, "
              "then lr 1e-3 x (1 - sqrt((step - 135k) / 31k)), 0 at 166k)\na larger filled marker around a half-filled "
              "one: the exponential cooldown, at almost the same score")
SQRT_SHIFT = 0.011  # added to the run's x offset factor: the 1-sqrt points stand a little right of the exponential ones
# name -> (folder and tag suffix, line style, marker fill, x offset added to the run's, zorder, legend label)
COOLDOWNS = {
    "exponential": ("_cd135", "--", "full", 0.0, 5, CD_LABEL),
    "1-sqrt": ("_cdsqrt135", ":", "bottom", SQRT_SHIFT, 5.2, SQRT_LABEL),
}
RING_SHARE = 0.03  # points of one run and stop nearer than this share of the y range: the lower one is drawn larger


def tag(code, stop, suffix=""):
    return f"{BASE}{RUNS[code][0]}_rollfix{suffix}_bb{stop}k_h30k_control"


def mase(path):
    with open(path) as f:
        return {r["dataset"]: float(r[COL]) for r in csv.DictReader(f)}


def geo_mean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def scores(code, stop, naive, suffix=""):
    """{'gm': ..., 'short': ..., ...} of one run (or its cooldown: suffix) at one stop, or None before its score file."""
    name = tag(code, stop, suffix)
    score_file = FF / "results" / f"score_{name}.txt"
    if not score_file.is_file() or not score_file.read_text().strip():
        return None
    ratios = {c: v / naive[c] for c, v in mase(FF / "heads/eval" / name / "gift/all_results.csv").items()}
    out = {"gm": float(score_file.read_text().split()[0]), "configs": len(ratios)}
    for h in HORIZONS:
        out[h] = geo_mean([r for c, r in ratios.items() if c.endswith("/" + h)])
    assert abs(geo_mean(ratios.values()) - out["gm"]) < 1e-4, (code, stop, out)
    return out


def print_table(stop, rows):
    print(f"\nstop {stop}k ({'250 steps' if stop == 0 else f'{stop}k steps'}):")
    print("| Run | GM | short | medium | long |\n|---|---|---|---|---|")
    for code, s in rows.items():
        print(f"| {code} | {s['gm']:.4f} | {s['short']:.4f} | {s['medium']:.4f} | {s['long']:.4f} |")


def table_scores(naive, path_of):
    """{stop: {'gm', 'short', 'medium', 'long'}} from per-config tables, for each MPE stop whose table exists."""
    out = {}
    for stop in MPE_STOPS:
        if not path_of(stop).is_file():
            continue
        ratios = {c: v / naive[c] for c, v in mase(path_of(stop)).items()}
        out[stop] = {"gm": geo_mean(ratios.values())}
        for h in HORIZONS:
            out[stop][h] = geo_mean([r for c, r in ratios.items() if c.endswith("/" + h)])
    return out


def reference_lines(naive):
    """[(scores by stop, colour, legend label)]: MPE, and the MPE recipe at width 64 (its GM checked)."""
    mpe = table_scores(naive, lambda stop: MPE_TABLES / f"moirai421ew_native{stop}k.csv")
    w64 = table_scores(naive, lambda stop: W64_TABLES / f"cf462_mpe_w64_native{stop}k.csv")
    for stop, s in w64.items():
        runner_gm = float((W64_SCORES / f"score_cf462_mpe_w64_bb{stop}k_native.txt").read_text().split()[0])
        assert abs(s["gm"] - runner_gm) < 1e-4, (stop, s["gm"], runner_gm)
    print("\nMPE recipe at width 64:", {k: round(v["gm"], 4) for k, v in w64.items()})
    return [(mpe, "#000080", MPE_LABEL), (w64, "#008080", W64_LABEL)]


def xpos(stop, offset, late):
    """x of a point: its stop, shifted a little per run so that the points of one stop stand side by side."""
    return stop + (offset - 1) * 40 if late else max(stop, 0.25) * offset


def draw_panel(ax, key, table, cd_tables, refs, late=False):
    for code, (_, label, colour, marker, offset) in RUNS.items():
        points = sorted((xpos(stop, offset, late), s[key]) for (c, stop), s in table.items() if c == code)
        if points:
            ax.plot(*zip(*points), color=colour, marker=marker, ms=12 if marker == "*" else 7, lw=1.8, mfc="none",
                    mew=1.6, label=label)
        for family, cd_table in cd_tables.items():
            draw_cooldown(ax, key, table, cd_table, code, family, late)
    for family, cd_table in cd_tables.items():
        if cd_table:
            _, ls, fill, _, _, cd_label = COOLDOWNS[family]
            ax.plot([], [], color="grey", marker="o", ms=9, lw=1.8, ls=ls, fillstyle=fill, mfcalt="white",
                    label=cd_label)
    # the reference lines: dashed, with stars. zorder 5.7 puts them above the run markers (5.6): a star stays whole,
    # and the hollow markers show around it
    for ref, colour, label in refs:
        ax.plot(list(ref), [s[key] for s in ref.values()], color=colour, ls="--", marker="*", ms=10, lw=1.8,
                label=label, zorder=5.7)
    ax.axhline(1.0, color="grey", ls="--", lw=1, label="Seasonal Naive (1.0)")
    (set_late_axis(ax, key, table, cd_tables, refs) if late else set_log_axis(ax))
    draw_rings(ax, key, table, cd_tables, late)
    draw_halos(ax, key, cd_tables, late)
    draw_run_markers_on_top(ax, key, table, late)
    ax.set_title("All 97 configs (GM)" if key == "gm" else f"{key.capitalize()} horizons", loc="left")
    ax.set_ylabel("GM-Relative MASE (lower is better)")


def draw_cooldown(ax, key, table, cd_table, code, family, late):
    """One cooldown of one run: its markers on a line that starts at the run's CD_FROM point."""
    _, _, colour, marker, offset = RUNS[code]
    _, ls, fill, shift, zorder, _ = COOLDOWNS[family]
    points = sorted((xpos(stop, offset + shift, late), s[key]) for (c, stop), s in cd_table.items() if c == code)
    if not points:
        return
    start = [(xpos(CD_FROM, offset, late), table[(code, CD_FROM)][key])] if (code, CD_FROM) in table else []
    ax.plot(*zip(*(start + points)), color=colour, lw=1.8, ls=ls, zorder=4)
    ax.plot(*zip(*points), color=colour, marker=marker, ms=13 if marker == "*" else 9, ls="none", fillstyle=fill,
            mfcalt="white", zorder=zorder)


def draw_rings(ax, key, table, cd_tables, late):
    """The run's point again as a larger open marker where a cooldown point at the same stop would hide it."""
    low, high = ax.get_ylim()
    for cd_table in cd_tables.values():
        for (code, stop), cd in cd_table.items():
            run = table.get((code, stop))
            if run and abs(run[key] - cd[key]) < RING_SHARE * (high - low):
                _, _, colour, marker, offset = RUNS[code]
                ax.plot(xpos(stop, offset, late), run[key], color=colour, marker=marker, ms=16, mfc="none", mew=1.6,
                        zorder=5.5)


def draw_halos(ax, key, cd_tables, late):
    """The exponential cooldown's point again as a larger filled marker where the 1-sqrt point would hide it."""
    low, high = ax.get_ylim()
    for (code, stop), sq in cd_tables.get("1-sqrt", {}).items():
        ex = cd_tables["exponential"].get((code, stop))
        if ex and abs(ex[key] - sq[key]) < RING_SHARE * (high - low):
            _, _, colour, marker, offset = RUNS[code]
            ax.plot(xpos(stop, offset, late), ex[key], color=colour, marker=marker, ms=17 if marker == "*" else 13,
                    ls="none", zorder=5.1)


def draw_run_markers_on_top(ax, key, table, late):
    """The hollow run markers again over the filled cooldown markers of the other runs, which can hide them."""
    for code, (_, _, colour, marker, offset) in RUNS.items():
        points = [(xpos(stop, offset, late), s[key]) for (c, stop), s in table.items() if c == code]
        if points:
            ax.plot(*zip(*points), color=colour, marker=marker, ms=12 if marker == "*" else 7, ls="none",
                    mfc="none", mew=1.6, zorder=5.6)


def set_log_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(LABELLED, [f"{s:g}" for s in LABELLED])
    ax.set_xticks([s for s in STOPS[1:] if s not in LABELLED], minor=True)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.2, 220)  # the 166k marker of the last run (x offset 1.184, x = 196.5) stays whole
    ax.grid(axis="x", which="both", color="#e6e6e6")
    ax.set_xlabel("Training step at batch 256 (thousands, log scale; 0.25 = the 250-step save)")


def set_late_axis(ax, key, table, cd_tables, refs):
    """The stops from LATE_FROM on a linear axis, the y range from the points there, a line at the cooldown start."""
    points = [*table.items(), *(item for cd_table in cd_tables.values() for item in cd_table.items())]
    late = [s[key] for (_, stop), s in points if stop >= LATE_FROM]
    late += [s[key] for ref, _, _ in refs for k, s in ref.items() if k >= LATE_FROM]
    ax.set_xlim(LATE_FROM - 5, 178)  # the 166k marker of the last run (x = 166 + 0.184 x 40 = 173.4) stays whole
    low, high = min(late) - 0.01, max(late) + 0.01
    ax.set_ylim(low, high + 0.1 * (high - low))  # headroom: the cooldown-start label stays above the points
    ax.set_xticks(sorted({s for s in STOPS if s >= LATE_FROM} | {CD_START}))
    ax.axvline(CD_START, color="grey", ls=":", lw=1.2)
    ax.text(CD_START, 0.99, " the cooldowns start\n from the 135k save", transform=ax.get_xaxis_transform(),
            va="top", fontsize=9, color="dimgrey")
    ax.grid(axis="x", color="#e6e6e6")
    ax.set_xlabel("Training step at batch 256 (thousands)")


def cooldown_table(family, naive):
    """{(code, stop): scores} of one cooldown family. Prints one table per scored stop."""
    table = {}
    suffix = COOLDOWNS[family][0]
    for stop in CD_STOPS:
        rows = {code: s for code in CD_RUNS if code in RUNS and (s := scores(code, stop, naive, suffix))}
        table.update({(code, stop): s for code, s in rows.items()})
        if rows:
            print_table(stop, {f"{code} {family} cooldown": s for code, s in rows.items()})
    return table


def main(out, late=False):
    naive = mase(SN)
    table = {}
    for stop in STOPS:
        rows = {code: s for code in RUNS if (s := scores(code, stop, naive))}
        table.update({(code, stop): s for code, s in rows.items()})
        if rows:
            print_table(stop, rows)
    cd_tables = {family: cooldown_table(family, naive) for family in COOLDOWNS}  # a HP64_RUNS view too (11-18)
    refs = reference_lines(naive)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14.5))
    for ax, key in zip(axes.flat, ["gm"] + HORIZONS):
        draw_panel(ax, key, table, cd_tables, refs, late)
    # one legend under the 4 panels, so that it hides no point
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=10, frameon=False)
    title = ("Forecast GM-Relative MASE of the width-64 runs from 90k steps: each cooldown (exponential: filled "
             "markers, dashed; 1-sqrt: half-filled, dotted) against its run, with MPE (width 384) and the MPE recipe "
             "at width 64" if late else
             "Forecast GM-Relative MASE of the width-64 runs that differ in L_align (its distances, cosine or L2) and "
             "SIGReg on f, at each scored stop, with their cooldowns, MPE (width 384) and the MPE recipe at width 64"
             if ONLY else
             "Forecast GM-Relative MASE of the width-64 runs (the 5 runs of the search, red, its variant, the L2 runs "
             "and the 1/d runs) at each scored stop, with MPE (width 384) and the MPE recipe at width 64")
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.175, 1, 0.98))  # the legend under the panels needs 17.5 % of the height
    fig.savefig(out, dpi=110)
    print("\nwrote", out)


if __name__ == "__main__":
    main(sys.argv[1], late=len(sys.argv) > 2 and sys.argv[2] == "late")
