"""Forecast scores of the runs of 11-18 to 11-20 at each scored stop, as curves (the user, 11-20: 'what I need is
a curve so that I can see things'). Row 1 is width 384: green-384, grey-384 and MPE. Row 2 is width 64: the new loss
of PR #470 (the joint teacher run with tau 1.0), green, grey and the MPE recipe at width 64 of cf#462. The four
columns are the GM-Relative MASE over the 97 GIFT-Eval configs, then the short, medium and long horizons. Green and
grey keep their colours in both rows: the user names these runs by their colours.

Usage: python3 plot_gm_overview.py <output png>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import plot_hp64_scores as p  # noqa: E402
import plot_w384_scores as w  # noqa: E402

JOINT = "g166b_lr1em3_tau99995_bs256_sigfix_ss_sphere_joint_tau1_rollfix"
GREEN, GREY, NEW = "#2ca02c", "#7f7f7f", "#9467bd"
METRICS = [("gm", "All 97 configs (GM)"), ("short", "Short horizons"), ("medium", "Medium horizons"),
           ("long", "Long horizons")]


def joint_scores(stop, naive):
    """{'gm', 'short', 'medium', 'long'} of the new loss at one stop, or None before its score file."""
    name = f"{JOINT}_bb{stop}k_h30k_control"
    f = p.FF / "results" / f"score_{name}.txt"
    if not f.is_file() or not f.read_text().strip():
        return None
    ratios = {c: v / naive[c] for c, v in p.mase(p.FF / "heads/eval" / name / "gift/all_results.csv").items()}
    out = {"gm": float(f.read_text().split()[0])}
    for h in p.HORIZONS:
        out[h] = p.geo_mean([r for c, r in ratios.items() if c.endswith("/" + h)])
    return out


def curve(ax, table, key, colour, marker, label, ls="-", first=0):
    """One run: its value of ``key`` at each stop of ``table`` ({stop: scores or None}) from ``first``k on."""
    points = sorted((s, v[key]) for s, v in table.items() if v and s >= first)
    if points:
        ax.plot(*zip(*points), color=colour, marker=marker, ms=7, lw=2, ls=ls, mfc="none", mew=1.6, label=label)


def tables(naive):
    """{row title: [(scores by stop, colour, marker, label, line style)]}."""
    (mpe, *_), (w64, *_) = p.reference_lines(naive)
    w384 = [(mpe, "#000080", "*", "MPE: our Moirai small (width 384)", "--")]
    w384 += [({s: w.scores(suffix, s, naive) for s in p.MPE_STOPS}, colour, marker, label, "-")
             for suffix, label, colour, marker, _ in w.RUNS.values()]
    w64_runs = [(w64, "#008080", "*", "MPE recipe at width 64 (cf#462)", "--"),
                ({s: p.scores("10-L-ALIGN", s, naive) for s in p.STOPS}, GREEN, "D", "green: 10-L-ALIGN", "-"),
                ({s: p.scores("1/d run", s, naive) for s in p.STOPS}, GREY, "h", "grey: the 1/d run", "-"),
                ({s: joint_scores(s, naive) for s in p.STOPS}, NEW, "o",
                 "new loss (PR #470): green, one fraction\nper depth, teacher keys, tau 1.0", "-")]
    return {"Width 384": w384, "Width 64": w64_runs}


def axes_style(ax, first):
    """The Seasonal Naive line, the step ticks from ``first``k, the limits and the grid of one panel."""
    ax.axhline(1.0, color="grey", ls=":", lw=1.2)
    ticks = [s for s in (10, 25, 50, 75, 100, 125, 150, 166) if s >= first]
    ax.set_xticks(ticks, [str(s) for s in ticks])
    ax.set_xlim(first - 6, 172)
    ax.grid(color="#e6e6e6")
    ax.set_xlabel("Training step (thousands)")


def main(out):
    naive = p.mase(p.SN)
    fig, axes = plt.subplots(2, 4, figsize=(26, 11.5))
    for r, (row_title, runs) in enumerate(tables(naive).items()):
        first = 10 if r == 0 else 25  # width 64 from 25k: its 250-step and 10k points hide the later ones
        for c, (key, title) in enumerate(METRICS):
            ax = axes[r, c]
            for table, colour, marker, label, ls in runs:
                curve(ax, table, key, colour, marker, label, ls, first)
            axes_style(ax, first)
            ax.set_title(f"{row_title}: {title}" + ("" if r == 0 else " (from 25k)"), loc="left", fontsize=11)
            if c == 0:
                ax.set_ylabel("GM-Relative MASE (lower is better)")
        axes[r, 3].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False)
    fig.suptitle("Forecast GM-Relative MASE of the current runs at each scored stop (dotted line: Seasonal Naive; "
                 "linear step axis)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
