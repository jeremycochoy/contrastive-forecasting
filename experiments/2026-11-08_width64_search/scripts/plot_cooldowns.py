"""The cooldowns of the 5 width-64 runs alone (the user, 11-12 about 20:25 UTC: 'just put the point before cooldown
and the point after cooldown and different color for each arm matching the existing color and nothing else').
For each run: its point before the cooldowns (the run at 125k, its last score before their 135k start) and its
points after them at 166k: the run with no cooldown (constant lr 1e-3), its exponential cooldown and its 1-sqrt
cooldown. One colour per run (the colours of plot_hp64_scores.py), one marker per schedule. Four panels (GM, short,
medium, long). A point appears when its score file exists.

Usage: python3 plot_cooldowns.py <output png>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from plot_hp64_scores import CD_RUNS, COOLDOWNS, RUNS, SN, mase, scores  # noqa: E402

BEFORE, AFTER = 125, 166
# schedule -> (folder and tag suffix, marker, filled, line style, legend label, x shift inside the run's slot)
SCHEDULES = {
    "none": ("", "o", False, "-", "no cooldown: the run itself (constant lr 1e-3)", -0.04),
    "exponential": (COOLDOWNS["exponential"][0], "s", True, "--",
                    "exponential cooldown: lr 1e-3 x 0.5^((step - 135k) / 5k)", 0.0),
    "1-sqrt": (COOLDOWNS["1-sqrt"][0], "D", True, ":",
               "square-root cooldown: lr 1e-3 x (1 - sqrt((step - 135k) / 31k)), 0 at 166k", 0.04),
}
SLOT = 0.12  # the x distance between the runs at one position: their 3 markers never touch
PANELS = [("gm", "All 97 configs (GM)"), ("short", "Short horizons"), ("medium", "Medium horizons"),
          ("long", "Long horizons")]


def load(naive):
    """{(code, schedule): {'before': scores, 'after': scores}}. 'before' is the run at 125k for each schedule."""
    data = {}
    for code in CD_RUNS:
        before = scores(code, BEFORE, naive)
        for schedule, (suffix, *_) in SCHEDULES.items():
            data[(code, schedule)] = {"before": before, "after": scores(code, AFTER, naive, suffix)}
    return data


def draw_panel(ax, key, title, data):
    for i, code in enumerate(CD_RUNS):
        colour, slot = RUNS[code][2], (i - (len(CD_RUNS) - 1) / 2) * SLOT
        for schedule, (_, marker, filled, ls, _, shift) in SCHEDULES.items():
            before, after = data[(code, schedule)]["before"], data[(code, schedule)]["after"]
            if before and after:
                ax.plot([slot, 1 + slot + shift], [before[key], after[key]], color=colour, ls=ls, lw=1.6, zorder=2)
                # a hollow marker has no face and lies on top: a filled marker at almost the same score shows through it
                ax.plot(1 + slot + shift, after[key], color=colour, marker=marker, ms=9, ls="none",
                        mfc=colour if filled else "none", mew=1.6, zorder=3 if filled else 4)
        if data[(code, "none")]["before"]:
            ax.plot(slot, data[(code, "none")]["before"][key], color=colour, marker="o", ms=9, mfc="white",
                    mew=1.6, zorder=3)
    ax.set_xticks([0, 1], ["before the cooldowns\n(the run at 125k)", "after the cooldowns\n(166k steps)"])
    ax.set_xlim(-0.4, 1.4)
    ax.grid(axis="y", color="#e6e6e6")
    ax.set_title(title, loc="left")
    ax.set_ylabel("GM-Relative MASE (lower is better)")


def legend_handles(data):
    runs = [Line2D([], [], color=RUNS[code][2], lw=2.5, label=code) for code in CD_RUNS]
    marks = []
    for schedule, (_, marker, filled, ls, label, _) in SCHEDULES.items():
        if not any(data[(code, schedule)]["after"] for code in CD_RUNS):
            label += " (no 166k score yet)"
        marks.append(Line2D([], [], color="grey", marker=marker, ms=9, mfc="grey" if filled else "white", ls=ls,
                            lw=1.6, label=label))
    return runs + marks


def main(out):
    data = load(mase(SN))
    fig, axes = plt.subplots(2, 2, figsize=(14, 11.5))
    for ax, (key, title) in zip(axes.flat, PANELS):
        draw_panel(ax, key, title, data)
    fig.legend(handles=legend_handles(data), loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle("The cooldowns of the width-64 runs: each run before its cooldowns and 166k steps after its 135k "
                 "save", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
