#!/usr/bin/env python3
"""The headline: each configuration at 11.4M against its 1.1M twin.

WHY THIS FIGURE EXISTS. The card asks one question. Does the contrastive
backbone improve when it has the capacity of a small foundation model? One
number for each configuration at each size answers it, and a bar for each size
side by side is how a reader sees the sign of the difference in one look.

WHAT IT DRAWS. One row for each arm that holds an 11.4M score. The 11.4M score
takes the series color. The 1.1M score its parent published takes the muted
reference ink, because it is not a run of this card. The seed band of #409,
0.0471, rides on the 11.4M bar as an error bar: a difference inside it is not
a rank.

Lower is better on this metric, so a bar that reaches further right is worse.

Usage:  plot_scores.py [--stop 40000] [--out plots/scores.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import plot_style as S  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
HEAD_STEPS = 30000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default=str(STUDY / "results" / "scores.csv"))
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--stop", type=int, default=S.STOP)
    p.add_argument("--out", default=str(STUDY / "plots" / "scores.png"))
    args = p.parse_args()

    arms = S.read_arms(args.arms)
    scored = S.read_scores(args.scores, stop=args.stop)
    rows = [r for r in arms if r["arm"] in scored]
    if not rows:
        raise SystemExit(f"no arm holds a score at {args.stop} steps")
    rows.sort(key=lambda r: scored[r["arm"]])

    fig, ax = plt.subplots(figsize=(9.0, 0.9 * len(rows) + 2.2))
    fig.patch.set_facecolor(S.SURFACE)
    ax.set_facecolor(S.SURFACE)
    h = 0.34

    unmatched = False
    for n, row in enumerate(rows):
        arm = row["arm"]
        y = len(rows) - 1 - n
        value = scored[arm]
        ax.barh(y + h / 2 + 0.03, value, height=h, color=S.SERIES,
                edgecolor=S.SURFACE, linewidth=2.0, zorder=3)
        ax.errorbar(value, y + h / 2 + 0.03, xerr=S.BAND / 2, fmt="none",
                    ecolor=S.SURFACE, elinewidth=1.6, capsize=3, zorder=4)
        ax.annotate(f"{value:.4f}", (value, y + h / 2 + 0.03), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=S.INK)
        ref = S.reference(arm, args.stop)
        if ref:
            # A hatched bar is a reference whose head budget is not this
            # card's, so the pair is not head-matched.
            matched = ref[1] == HEAD_STEPS
            unmatched = unmatched or not matched
            ax.barh(y - h / 2 - 0.03, ref[0], height=h, color=S.REFERENCE,
                    edgecolor=S.SURFACE, linewidth=2.0,
                    alpha=0.55 if matched else 0.30,
                    hatch=None if matched else "///", zorder=3)
            note = "" if matched else f"  ({ref[1]:,}-step head)"
            ax.annotate(f"{ref[0]:.4f}{note}", (ref[0], y - h / 2 - 0.03),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=S.MUTED)
        else:
            ax.annotate("no 1.1M twin — the parent never ran this cell",
                        (0.0, y - h / 2 - 0.03), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color=S.MUTED, style="italic")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([S.arm_label(r) for r in reversed(rows)], fontsize=9,
                       color=S.INK)
    ax.set_xlabel("GM-Relative MASE over 97 GIFT-Eval configs — lower is better")
    ax.set_xlim(left=0.0)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_title(f"11.4M against 1.1M at {args.stop:,} backbone steps",
                 color=S.INK, fontsize=12, loc="left", pad=12)
    S.tidy(ax)
    ax.grid(axis="y", visible=False)

    handles = [Patch(color=S.SERIES,
                     label=f"11.4M, this card, {HEAD_STEPS:,}-step head"),
               Patch(color=S.REFERENCE, alpha=0.55,
                     label="1.1M, the parent study")]
    if unmatched:
        handles.append(Patch(facecolor=S.REFERENCE, alpha=0.30, hatch="///",
                             label="a 1.1M score at another head budget"))
    handles.append(Line2D([], [], color=S.MUTED, linewidth=1.6,
                          label=f"the {S.BAND:.4f} seed band of #409"))
    # Below the figure, never on a bar and never on the axis label.
    fig.legend(handles=handles, frameon=False, fontsize=8, labelcolor=S.INK,
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"wrote {args.out} ({len(rows)} arm(s))")


if __name__ == "__main__":
    main()
