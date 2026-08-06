"""Per-domain GM-Relative MASE radars for the EMA-schedule ladder.

An entry is one run plus one head. The radial axis is log2(ratio), so equal
multiplicative steps are equal distances.

Left  — the 5 lowest entries, each at its last evaluated stop.
Right — the entries that reached the highest backbone step (200k).
"""
from pathlib import Path
import csv, math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
EXP = HERE.parent.parent.parent / "experiments" / "2026-08-04_ema_sched_ladder"
SN_REF = HERE.parent.parent / "2026-07-21_split_pred_rep_small" / "results" / "seasonal_naive_all_results.csv"

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})

# Five entries per panel, and two entries of one run can both be drawn, so
# colour is keyed to the ENTRY position, not to the run: keying it to the run
# put two blues and two golds in the left panel.
ENTRY_COLORS = ["#2a78d6", "#e06c00", "#007d4f", "#b5279b", "#6a4a2f"]
HEAD_STYLE = {"student": {"ls": "-", "marker": "o", "ms": 7},
              "teacher": {"ls": "--", "marker": "s", "ms": 6}}


def _mase(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                out[row["dataset"]] = float(row["eval_metrics/MASE[0.5]"])
            except (KeyError, ValueError, TypeError):
                continue
    return out


def _domains(path):
    with open(path) as f:
        return {r["dataset"]: r["domain"] for r in csv.DictReader(f) if r.get("domain")}


SN = _mase(SN_REF)


def per_domain(cell, stop, head):
    p = EXP / "results" / "eval" / cell / "eval" / f"bb{stop // 1000}k_{head}" / "gift" / "all_results.csv"
    ours, doms = _mase(p), _domains(p)
    logs, counts = {}, {}
    for ds, m in ours.items():
        ref, d = SN.get(ds), doms.get(ds)
        if not ref or ref <= 0 or not d or m <= 0:
            continue
        logs.setdefault(d, []).append(math.log(m / ref))
        counts[d] = counts.get(d, 0) + 1
    return {d: math.exp(sum(v) / len(v)) for d, v in logs.items()}, counts


rows = []
with open(EXP / "results" / "ladder_all.csv") as f:
    for r in csv.DictReader(f):
        rows.append((r["cell"], int(r["stop"]), r["head"], float(r["gm_rel_mase"])))

last = {}
for cell, stop, head, v in rows:
    k = (cell, head)
    if k not in last or stop > last[k][0]:
        last[k] = (stop, v)
lowest = sorted(((v, c, s, h) for (c, h), (s, v) in last.items()))[:5]

top_stop = max(s for _c, s, _h, _v in rows)
deepest = sorted((v, c, s, h) for c, s, h, v in rows if s == top_stop)

allv = []
for _v, c, s, h in lowest + deepest:
    d, _ = per_domain(c, s, h)
    allv += list(d.values())
LO, HI = min(allv), max(allv)
_, ref_counts = per_domain("arm6_v2_combab_alignS", 100000, "student")
DOMAINS = sorted(ref_counts, key=lambda d: -ref_counts[d])
N = len(DOMAINS)
ANG = [n / N * 2 * math.pi for n in range(N)] + [0.0]
TICKS = [t for t in (0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.8, 4.0) if LO / 1.08 <= t <= HI * 1.08]
if TICKS[0] > LO:
    TICKS.insert(0, round(LO, 2))
if TICKS[-1] < HI:
    TICKS.append(round(HI, 2))
r = math.log2
RMIN, RMAX = r(TICKS[0]) - 0.03, r(TICKS[-1]) + 0.03


def draw(ax, entries, title):
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(RMIN, RMAX)
    ax.set_xticks(ANG[:-1])
    ax.set_xticklabels([f"{d}\n({ref_counts[d]} cfg)" for d in DOMAINS], fontsize=9)
    ax.set_yticks([r(t) for t in TICKS])
    ax.set_yticklabels([f"{t:g}" for t in TICKS], fontsize=7.5, color=MUTED)
    ax.grid(color=GRID, alpha=0.9)
    ax.set_rlabel_position(180 / N)
    ax.plot(ANG, [r(1.0)] * (N + 1), color="#c04040", lw=1.8, ls="--", zorder=3)
    ax.fill(ANG, [r(1.0)] * (N + 1), color="#c04040", alpha=0.06, zorder=0)

    handles = []
    for idx, (val, cell, stop, head) in enumerate(entries):
        d, _ = per_domain(cell, stop, head)
        vals = [r(d[k]) for k in DOMAINS]
        vals += vals[:1]
        st = HEAD_STYLE[head]
        colour = ENTRY_COLORS[idx % len(ENTRY_COLORS)]
        ln, = ax.plot(ANG, vals, color=colour, lw=2.0, ls=st["ls"], marker=st["marker"],
                      markersize=st["ms"], markeredgewidth=0, zorder=4,
                      label=f"{cell} {head} @ bb{stop // 1000}k — all-config {val:.3f}")
        handles.append(ln)
        ax.fill(ANG, vals, color=colour, alpha=0.05, zorder=1)
        for ang, k in zip(ANG[:-1], DOMAINS):
            if d[k] >= 2.0:
                ax.annotate(f"{d[k]:.2f}", xy=(ang, r(d[k])), fontsize=7.5, color=colour,
                            ha="left", va="bottom", zorder=5)
    ax.set_title(title, fontsize=10.5, pad=18)
    handles.append(Line2D([], [], color="#c04040", lw=1.8, ls="--",
                          label="parity ring: ratio 1.0, seasonal naive"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              fontsize=8.5, frameon=False, ncol=1)


fig, axes = plt.subplots(1, 2, figsize=(14.5, 9.4), subplot_kw={"projection": "polar"})
draw(axes[0], lowest, "5 lowest entries, each at its last evaluated stop")
draw(axes[1], deepest, f"Entries that reached the deepest backbone step (bb{top_stop // 1000}k)")
fig.suptitle("GM-Relative MASE by dataset domain (radial axis: log2 ratio)",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0.10, 1, 0.94])
# The numbers behind both panels, so every per-domain claim traces to a file.
scores_out = EXP / "results" / "domain_scores.csv"
with open(scores_out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["cell", "head", "stop", "panel", "domain", "n_configs",
                "gm_rel_mase"])
    for panel, entries in (("lowest5", lowest), ("deepest", deepest)):
        for _v, c, s_, h in entries:
            d, counts = per_domain(c, s_, h)
            for dom in DOMAINS:
                w.writerow([c, h, s_, panel, dom, counts[dom],
                            f"{d[dom]:.4f}"])
print("wrote", scores_out)

out = HERE / "domain_radar.png"
fig.savefig(out)
print("wrote", out)
print("left :", [(c, s, h, round(v, 4)) for v, c, s, h in lowest])
print("right:", [(c, s, h, round(v, 4)) for v, c, s, h in deepest])
