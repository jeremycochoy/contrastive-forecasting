"""The forecast scores of the width-384 runs of 11-18 against MPE (our Moirai small, width 384), at each scored stop.
green-384 and grey-384 are green and grey at width 384, with 1 encoder and 5 forecaster layers and bf16 (the user:
'scale the grey to 1.5 and 380 something in latent and the green the same'). The score is the GM-Relative MASE over
the 97 GIFT-Eval configs, then by horizon (the geometric mean of MASE[0.5] / the MASE[0.5] of Seasonal Naive over
the configs of that horizon).

Usage: python3 plot_w384_scores.py <output png>   (prints one markdown table per stop, then draws 4 panels)
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import plot_hp64_scores as p  # noqa: E402

W384 = "g166b_w384e1f5_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere"
# label -> (name suffix, legend label, colour, marker, x offset factor)
RUNS = {
    "green-384": ("_align10", "green-384: green (10-L-ALIGN) at width 384, 1 + 5 layers", "#2ca02c", "D", 0.985),
    "grey-384": ("_align10harm", "grey-384: grey (1/d run) at width 384, 1 + 5 layers", "#7f7f7f", "h", 1.015),
}
PANELS = [("gm", "All 97 configs (GM)"), ("short", "Short horizons"), ("medium", "Medium horizons"),
          ("long", "Long horizons")]


def scores(suffix, stop, naive):
    """{'gm', 'short', 'medium', 'long'} of one width-384 run at one stop, or None before its score file."""
    name = f"{W384}{suffix}_rollfix_bb{stop}k_h30k_control"
    score_file = p.FF / "results" / f"score_{name}.txt"
    if not score_file.is_file() or not score_file.read_text().strip():
        return None
    ratios = {c: v / naive[c] for c, v in p.mase(p.FF / "heads/eval" / name / "gift/all_results.csv").items()}
    out = {"gm": float(score_file.read_text().split()[0])}
    for h in p.HORIZONS:
        out[h] = p.geo_mean([r for c, r in ratios.items() if c.endswith("/" + h)])
    assert len(ratios) == 97 and abs(p.geo_mean(ratios.values()) - out["gm"]) < 1e-4, (name, out)
    return out


def draw_panel(ax, key, table, mpe):
    """One panel: MPE, then each width-384 run, on a log step axis."""
    xs = sorted(mpe)
    ax.plot(xs, [mpe[s][key] for s in xs], color="#000080", marker="*", ms=11, ls="--", lw=1.6, label=p.MPE_LABEL)
    for code, (_, label, colour, marker, offset) in RUNS.items():
        points = sorted((stop * offset, s[key]) for (c, stop), s in table.items() if c == code)
        if points:
            ax.plot(*zip(*points), color=colour, marker=marker, ms=8, lw=2, mfc="none", mew=1.8, label=label)
    ax.axhline(1.0, color="grey", ls=":", lw=1.2, label="Seasonal Naive (1.0)")
    ax.set_xscale("log")
    ax.set_xticks(p.MPE_STOPS, [str(s) if s in p.LABELLED else "" for s in p.MPE_STOPS])  # 150 and 166 overlap
    ax.minorticks_off()
    ax.set_xlabel("Training step at batch 256 (thousands, log scale)")
    ax.set_ylabel("GM-Relative MASE (lower is better)")
    ax.grid(color="#e6e6e6")


def main(out):
    naive = p.mase(p.SN)
    table = {}
    for stop in p.MPE_STOPS:
        rows = {code: s for code, (suffix, *_) in RUNS.items() if (s := scores(suffix, stop, naive))}
        table.update({(code, stop): s for code, s in rows.items()})
        if rows:
            p.print_table(stop, rows)
    mpe = p.reference_lines(naive)[0][0]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, (key, title) in zip(axes.flat, PANELS):
        draw_panel(ax, key, table, mpe)
        ax.set_title(title, loc="left")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle("Forecast GM-Relative MASE of the width-384 runs against MPE, at each scored stop", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(out, dpi=110)
    print("\nwrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
