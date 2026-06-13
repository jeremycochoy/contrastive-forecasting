#!/usr/bin/env python3
"""#327 — plots + scoreboard for the allt·0.8% arm across the contrastive-batch ladder
256 -> 1024 -> 2048 (all negatives pooled), per q-head (2L / 6L).

Bootstrap + GM logic lifted verbatim from #322's plots.py so the CIs are computed
identically (clean paired comparison). Single arm here, so the 5-arm radar/pooled views
of #322 collapse to: a batch-ladder bar chart, the paired Δ per doubling, a per-domain
b1024-vs-b2048 radar, and the training-stability trace.

Outputs (into ../plots/ and ../results/):
  gm_ladder.png   Full-97 GM-Relative MASE at b256 / b1024 / b2048, per head. Whisker =
                  bootstrap 90% CI over the 97 configs. v11c + seasonal-naive as lines.
  batch_delta.png Paired-bootstrap Δ over the 97 shared configs, per head, for each
                  doubling: (b1024-b256) and (b2048-b1024). <0 = the larger pool helps.
  perdomain.png   Per-domain GM, b1024 vs b2048 (each at its better head).
  stability.png   Floor-subtracted contrastive loss + forecast-vs-future gap + cross-batch
                  cosine through training — the collapse check at batch 2048.
  gm_table.csv    The scoreboard (also printed).
Runs incrementally: missing summaries are skipped (printed as NA)."""
import csv
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
RES = {
    "b256":  (f"{ROOT}/2026-05-26_forked_6Lf/results",          "xshh_allt_forked2_6Lf_50k"),
    "b1024": (f"{ROOT}/2026-05-29_forked_6Lf_b1024/results",    "xshh_allt_forked2_qk_aon_b1024"),
    "b2048": (f"{ROOT}/2026-06-03_forked_allt08_b2048/results", "forked2_qk_aon"),
}
WT = ("/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/"
      "forked-allt08-b2048/experiments/2026-06-03_forked_allt08_b2048")
PLOTS = f"{WT}/plots"
os.makedirs(PLOTS, exist_ok=True)
HEADS = ["2L", "6L"]
V11C = 1.292
NAIVE = 1.0
LOSSCSV = (f"{ROOT}/2026-06-03_forked_allt08_b2048/runs/"
           "bb_xshh_allt_forked2_qk_aon_6Lf_b2048_losses.csv")
LOSSCSV_B1024 = (f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/"
                 "bb_xshh_allt_forked2_qk_aon_6Lf_b1024_losses.csv")


# ---- methodology, verbatim from #322 plots.py ----------------------------------
def relatives_dict(sum_txt):
    out = {}
    if not os.path.exists(sum_txt):
        return out
    with open(sum_txt) as f:
        for line in f:
            p = line.split()
            if len(p) == 4 and "/" in p[0]:
                try:
                    out[p[0]] = float(p[3])
                except ValueError:
                    pass
    return out


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def gm_ci(xs, n=2000, seed=0):
    xs = [x for x in xs if x and x > 0]
    if len(xs) < 2:
        return (None, None)
    rng = random.Random(seed)
    gms = []
    for _ in range(n):
        s = [xs[rng.randrange(len(xs))] for _ in xs]
        gms.append(gm(s))
    gms.sort()
    return (gms[int(0.05 * n)], gms[int(0.95 * n)])


def paired_delta_ci(da, db, n=2000, seed=0):
    """Δ = gm(db) − gm(da), paired bootstrap over shared configs."""
    common = sorted(set(da) & set(db))
    if len(common) < 2:
        return (None, None, None, 0)
    a = [da[c] for c in common]
    b = [db[c] for c in common]
    delta = gm(b) - gm(a)
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        idx = [rng.randrange(len(common)) for _ in common]
        ds.append(gm([b[i] for i in idx]) - gm([a[i] for i in idx]))
    ds.sort()
    return (delta, ds[int(0.05 * n)], ds[int(0.95 * n)], len(common))


def rels(batch, head, sset="full"):
    res, tag = RES[batch]
    return relatives_dict(f"{res}/gift_eval_{sset}_{tag}_{head}/summary.txt")


def _domain_map():
    for batch in ("b1024", "b2048", "b256"):
        res, tag = RES[batch]
        for head in HEADS:
            path = f"{res}/gift_eval_full_{tag}_{head}/all_results.csv"
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                out = {row["dataset"]: row.get("domain", "?")
                       for row in csv.DictReader(fh)}
            if out:
                return out
    return {}


def f(x, p=4):
    return "NA" if x is None else f"{x:.{p}f}"


# ---- scoreboard ----------------------------------------------------------------
def collect():
    rows = []
    for head in HEADS:
        r = {b: rels(b, head, "full") for b in RES}
        t = {b: rels(b, head, "triage") for b in RES}
        d10, lo10, hi10, n10 = paired_delta_ci(r["b256"], r["b1024"])
        d21, lo21, hi21, n21 = paired_delta_ci(r["b1024"], r["b2048"])
        rows.append(dict(
            arm="allt·0.8%", head=head,
            b256=gm(list(r["b256"].values())),
            b1024=gm(list(r["b1024"].values())),
            b2048=gm(list(r["b2048"].values())),
            d_1024_256=d10, ci10_lo=lo10, ci10_hi=hi10,
            d_2048_1024=d21, ci21_lo=lo21, ci21_hi=hi21, n=n21,
            b256_tri=gm(list(t["b256"].values())),
            b1024_tri=gm(list(t["b1024"].values())),
            b2048_tri=gm(list(t["b2048"].values())),
        ))
    return rows


def print_table(rows):
    print(f"\n{'head':4} {'b256':>7} {'b1024':>7} {'b2048':>7} "
          f"{'Δ(2048-1024)':>13} {'90% CI':>20}")
    for r in rows:
        ci = (f"({f(r['ci21_lo'],3)}, {f(r['ci21_hi'],3)})"
              if r['ci21_lo'] is not None else "—")
        print(f"{r['head']:4} {f(r['b256']):>7} {f(r['b1024']):>7} {f(r['b2048']):>7} "
              f"{f(r['d_2048_1024'],3):>13} {ci:>20}")
    out = f"{RES['b2048'][0]}/gm_table.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # mirror into the committed report dir
    with open(f"{WT}/results/gm_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} (+ report copy)")


C = {"b256": "#c6dbef", "b1024": "#6baed6", "b2048": "#08519c"}  # blues, darker = bigger
C2 = {"2L": "#1f77b4", "6L": "#d94801"}
C_V11C = "#9467bd"


def plot_gm_ladder(rows):
    by = {r['head']: r for r in rows}
    batches = ["b256", "b1024", "b2048"]
    x = np.arange(len(HEADS)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, b in enumerate(batches):
        vals = [by[h].get(b) for h in HEADS]
        cis = [gm_ci(list(rels(b, h, "full").values())) for h in HEADS]
        err = [[(v - lo) if (v and lo) else 0 for v, (lo, hi) in zip(vals, cis)],
               [(hi - v) if (v and hi) else 0 for v, (lo, hi) in zip(vals, cis)]]
        ax.bar(x + (i - 1) * w, [v or 0 for v in vals], w, label=b, color=C[b],
               yerr=err, capsize=3, ecolor="#555")
        for xi, v in zip(x + (i - 1) * w, vals):
            if v:
                ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(V11C, color=C_V11C, ls="--", lw=1.4, label=f"v11c ({V11C})")
    ax.axhline(NAIVE, color="k", ls=":", lw=1.2, label="seasonal-naive (1.0)")
    ax.set_xticks(x); ax.set_xticklabels([f"{h} q-head" for h in HEADS])
    ax.set_ylabel("Full-97 GM-Relative MASE (lower better)")
    ax.set_title("allt·0.8%: forecast error vs contrastive batch (256 → 1024 → 2048)")
    ax.legend(ncol=2, fontsize=8); ax.set_ylim(0.9, None)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/gm_ladder.png", dpi=120); plt.close(fig)
    print("wrote gm_ladder.png")


def plot_batch_delta(rows):
    labels, deltas, los, his, cols = [], [], [], [], []
    for r in rows:
        for tag, d, lo, hi in (("1024−256", r['d_1024_256'], r['ci10_lo'], r['ci10_hi']),
                               ("2048−1024", r['d_2048_1024'], r['ci21_lo'], r['ci21_hi'])):
            if d is None:
                continue
            labels.append(f"{r['head']}: Δ({tag})")
            deltas.append(d); los.append(lo); his.append(hi)
            cols.append("#2ca02c" if hi < 0 else "#d62728" if lo > 0 else "#7f7f7f")
    if not deltas:
        print("batch_delta: no cells yet"); return
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.7 * len(labels))))
    ax.barh(y, deltas, color=cols,
            xerr=[np.array(deltas) - np.array(los), np.array(his) - np.array(deltas)],
            capsize=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("Δ GM-Relative MASE;  < 0 = the larger pooled batch helps")
    ax.set_title("allt·0.8%: change in forecast error per contrastive-batch doubling")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/batch_delta.png", dpi=120); plt.close(fig)
    print("wrote batch_delta.png")


def _best_profile(batch, dom_map):
    cand = []
    for h in HEADS:
        r = rels(batch, h, "full")
        if r:
            cand.append((gm(list(r.values())), h, r))
    if not cand:
        return None
    agg, hd, r = min(cand)
    byd = {}
    for cfg, v in r.items():
        byd.setdefault(dom_map.get(cfg, "?"), []).append(v)
    return hd, {d: gm(vs) for d, vs in byd.items()}


def plot_perdomain():
    dom_map = _domain_map()
    p1024 = _best_profile("b1024", dom_map)
    p2048 = _best_profile("b2048", dom_map)
    if not (dom_map and p1024 and p2048):
        print("perdomain: incomplete — skip"); return
    doms = sorted(set(p1024[1]) | set(p2048[1]))
    ang = np.linspace(0, 2 * np.pi, len(doms), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    fig, ax = plt.subplots(figsize=(8.5, 7.2), subplot_kw=dict(polar=True))
    ax.plot(np.linspace(0, 2 * np.pi, 200), [NAIVE] * 200, "--", color="k", lw=1.2,
            alpha=0.7, label="seasonal-naive (1.0)")
    ax.plot(np.linspace(0, 2 * np.pi, 200), [V11C] * 200, "--", color=C_V11C, lw=1.2,
            alpha=0.8, label=f"v11c ({V11C})")
    for (hd, m), col, ls, lw in ((p1024, C["b1024"], "--", 1.6), (p2048, C["b2048"], "-", 2.1)):
        v = [m.get(d, np.nan) for d in doms]
        tag = "b1024" if col == C["b1024"] else "b2048"
        ax.plot(ang_c, v + v[:1], ls, color=col, lw=lw, label=f"{tag} ({hd})")
    ax.set_rscale("log"); ax.set_rlim(0.6, 3.2)
    ax.set_xticks(ang); ax.set_xticklabels(doms, fontsize=9); ax.set_rlabel_position(95)
    ax.set_title("allt·0.8%: forecast error by data domain, batch 1024 vs 2048", pad=18)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.28, 1.10))
    fig.tight_layout(); fig.savefig(f"{PLOTS}/perdomain.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote perdomain.png")


def _loss_series(path, key, from_step=100):
    if not os.path.exists(path):
        return [], []
    s, v = [], []
    for r in csv.DictReader(open(path)):
        st = int(r["step"]); x = r.get(key)
        if st >= from_step and x not in (None, "", "nan"):
            try: v.append(float(x)); s.append(st)
            except ValueError: pass
    return s, v


# The forecastable-latent training curves (step >= 100): loss, the subtraction gap
# (ff - fp) and the quotient gap (1-ff)/(1-fp) -> 0, R2_random and R2_naive (forecaster
# predicting the next encoder latent vs random / persistence), cross-batch cosine.
def plot_train_curves():
    panels = [("loss", "floor-subtracted contrastive loss", True),
              ("gap", "subtraction gap  ff - fp", False),
              ("gap_ratio", "quotient gap  (1-ff)/(1-fp)  -> 0", True),
              ("r2_random", "R2_random  (vs random latent)", False),
              ("r2_naive", "R2_naive  (vs persistence)", False),
              ("cross_batch", "cross-batch cosine (collapse -> up)", True)]
    if not os.path.exists(LOSSCSV):
        print("train_curves: no b2048 loss CSV — skip"); return
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (key, title, logy) in zip(axes.flat, panels):
        for F, c, lab in ((LOSSCSV_B1024, C["b1024"], "batch 1024"),
                          (LOSSCSV, C["b2048"], "batch 2048")):
            s, v = _loss_series(F, key)
            if s: ax.plot(s, v, color=c, lw=1.4, label=lab)
        ax.set_xscale("log")
        if logy: ax.set_yscale("log")
        if key in ("gap", "r2_random", "r2_naive"): ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_title(title + ("  [log-log]" if logy else "  [log-x]"))
        ax.set_xlabel("step (log, from 100)"); ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("allt·0.8% training curves, step>=100 — batch 1024 vs 2048", fontsize=14)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/train_curves_loglog.png", dpi=120); plt.close(fig)
    print("wrote train_curves_loglog.png")


HORIZONS = ["short", "medium", "long"]


def _horizon(cfg):
    for h in HORIZONS:
        if cfg.endswith("/" + h): return h
    return "other"


def by_horizon(batch, head):
    r = rels(batch, head, "full")
    out = {}
    if not r: return out
    for h in HORIZONS:
        vs = [v for k, v in r.items() if _horizon(k) == h]
        if vs: out[h] = gm(vs)
    return out


# Headline: forecast error split by horizon. b2048 matches b1024 on short horizons and
# collapses on medium/long (multi-step rollout compounds its weaker per-step forecaster).
def plot_horizon(head="6L"):
    batches = ["b256", "b1024", "b2048"]
    prof = {b: by_horizon(b, head) for b in batches}
    if not prof["b2048"]:
        print("horizon: no b2048 — skip"); return
    x = np.arange(len(HORIZONS)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, b in enumerate(batches):
        vals = [prof[b].get(h) for h in HORIZONS]
        ax.bar(x + (i - 1) * w, [v or 0 for v in vals], w, label=b, color=C[b])
        for xi, v in zip(x + (i - 1) * w, vals):
            if v: ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(NAIVE, color="k", ls=":", lw=1.2, label="seasonal-naive (1.0)")
    ax.set_xticks(x); ax.set_xticklabels([h + " horizon" for h in HORIZONS])
    ax.set_ylabel("GM-Relative MASE (lower better)")
    ax.set_title(f"allt·0.8% ({head} head): forecast error by horizon, batch 256/1024/2048")
    ax.legend(); ax.set_ylim(0.9, None)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/horizon_split_{head}.png", dpi=120); plt.close(fig)
    print(f"wrote horizon_split_{head}.png")


def plateau_table():
    """Plateau test: mid-plateau (step ~2500) vs fully-trained (step 6250) head, per depth."""
    res = RES["b2048"][0]
    g = lambda tag, head, sset: gm(list(relatives_dict(
        f"{res}/gift_eval_{sset}_{tag}_{head}/summary.txt").values()) or [None])
    print("\nPlateau test (b2048 full-97 GM):")
    print(f"{'head':4} {'plateau(2500)':>14} {'final(6250)':>12}")
    for head in HEADS:
        p = g("platpeak", head, "full"); fi = g("forked2_qk_aon", head, "full")
        print(f"{head:4} {f(p,3):>14} {f(fi,3):>12}")


if __name__ == "__main__":
    # Emits exactly the three figures RESULTS.md embeds (no orphan images):
    #   horizon_split_6L.png (Fig 1), gm_ladder.png (Fig 2), train_curves_loglog.png (Fig 3).
    # plot_batch_delta / plot_perdomain / plot_horizon("2L") are kept as functions for ad-hoc
    # use but are not part of the report, so they are not generated here.
    rows = collect()
    print_table(rows)
    plateau_table()
    plot_horizon("6L")
    plot_gm_ladder(rows)
    plot_train_curves()
