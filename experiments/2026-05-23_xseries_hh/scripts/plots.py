#!/usr/bin/env python3
"""#318 plots — generates every figure for RESULTS.md from the eval
summaries + training losses CSVs. Each figure auto-skips if its inputs are
missing, so this is safe to run incrementally as cells land.

Figures:
  gm_summary.png      final full-97 & triage GM-Relative MASE: xshh {2L,6L}
                      vs β {2L,6L}, with v11c + seasonal-naive references.
  gm_vs_step.png      full-97 GM vs backbone training step (2L head),
                      xshh vs β — does the "more training stops helping"
                      decoupling shrink?
  training_curves.png contrastive training dynamics (loss / gap / AUC / Top1)
                      vs step, xshh vs β.
  perdomain.png       per-domain GM-Relative MASE (2L), xshh vs β vs v11c —
                      flags collateral damage on strongly-seasonal domains.
"""
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# For the InfoNCE floor (re-bases the contrastive loss so arms with different
# negative counts N are comparable). Training config: τ=0.10, B=256, C=1, and
# latent T=64 — the HF streamer crops to T_RAW=1024 (it ignores --t-raw 4096),
# and 1024//W(=16) = 64. (Verified by a model forward.)
sys.path.insert(0, "/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/cross-series-hh")
from src.loss import infonce_floor, _effective_negative_count

TAU_TRAIN, B_TRAIN, T_TRAIN, C_TRAIN = 0.10, 256, 64, 1
_SHAPE = {"xshh": "cosine_similarity_batch_full_hh_negs_xshh",
          "allt": "cosine_similarity_batch_full_hh_negs_xshh_allt",
          "beta": "cosine_similarity_batch_full_hh_negs"}
FLOOR = {k: infonce_floor(TAU_TRAIN, _effective_negative_count(s, B_TRAIN, T_TRAIN, C_TRAIN))
         for k, s in _SHAPE.items()}

OUT = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-23_xseries_hh"
RES = f"{OUT}/results"
PLOTS = "/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/cross-series-hh/experiments/2026-05-23_xseries_hh/plots"
os.makedirs(PLOTS, exist_ok=True)
BETA_DIR = "/home/jupyter/contrastive-forecasting/experiments/2026-05-20_bottleneck_beta2_confound"
V11C = "/home/jupyter/contrastive-forecasting/experiments/2026-05-11_exp_encoder_forecaster/results/gift_eval_full_v11c"

XSHH_CSV = f"{OUT}/runs/bb_xshh_50k_losses.csv"
ALLT_CSV = f"{OUT}/runs/bb_xshh_allt_50k_losses.csv"
BETA_CSV = f"{BETA_DIR}/runs/bb_beta_50k_losses.csv"
FORK_CSV  = f"{OUT}/runs/bb_xshh_allt_forked_50k_losses.csv"    # forked, all-time loss, 50% synth
FORK2_CSV = f"{OUT}/runs/bb_xshh_allt_forked2_50k_losses.csv"   # forked, all-time loss, 2 samples/batch
BFORK_CSV = f"{OUT}/runs/bb_beta_forked2_50k_losses.csv"        # forked, β loss, 2 samples/batch
FORK10_CSV  = f"{OUT}/runs/bb_xshh_allt_forked10pct_50k_losses.csv"  # forked, all-time loss, 10%
BFORK10_CSV = f"{OUT}/runs/bb_beta_forked10pct_50k_losses.csv"       # forked, β loss, 10%

C_XSHH, C_ALLT, C_BETA, C_V11C = "#1f77b4", "#2ca02c", "#d62728", "#9467bd"
C_FORK, C_FORK2, C_BFORK = "#ff7f0e", "#8c564b", "#e377c2"   # data-side forked: allt·50% / allt·2 / β·2
C_FORK10, C_BFORK10 = "#bcbd22", "#17becf"   # forked 10%: all-time / β


def agg_gm(sum_txt):
    """Aggregate GM-Relative MASE printed in a summary.txt (or None)."""
    if not os.path.exists(sum_txt):
        return None
    with open(sum_txt) as f:
        for line in f:
            if "Aggregate GM-Relative MASE" in line:
                for t in reversed(line.replace(":", " ").split()):
                    try:
                        return float(t)
                    except ValueError:
                        continue
    return None


def per_config_relative(sum_txt):
    """{config: relative_mase} parsed from a summary.txt body (or {})."""
    out = {}
    if not os.path.exists(sum_txt):
        return out
    with open(sum_txt) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and "/" in p[0]:
                try:
                    out[p[0]] = float(p[-1])
                except ValueError:
                    pass
    return out


def config_domain(csv_path):
    """{config: domain} from an all_results.csv (or {})."""
    out = {}
    if not os.path.exists(csv_path):
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["dataset"]] = row.get("domain", "?")
    return out


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


# ---------------------------------------------------------------- summary bars
def plot_gm_summary():
    # label -> summary.txt. β 2L is the #309 backbone-of-record eval.
    # `s2` (optional 5th item) = (full, triage) of a SECOND seed (20260521) for
    # that exact cell; drawn as a whisker so the seed spread is visible.
    fk = lambda tag, h, s: f"{RES}/gift_eval_{s}_{tag}_{h}/summary.txt"
    cells = [
        ("xshh same-step · 2L", fk("xshh_50k", "2L", "full"), fk("xshh_50k", "2L", "triage"), C_XSHH),
        ("xshh same-step · 6L", fk("xshh_50k", "6L", "full"), fk("xshh_50k", "6L", "triage"), C_XSHH),
        ("xshh all-time · 2L", fk("xshh_allt_50k", "2L", "full"), fk("xshh_allt_50k", "2L", "triage"), C_ALLT),
        ("xshh all-time · 6L", fk("xshh_allt_50k", "6L", "full"), fk("xshh_allt_50k", "6L", "triage"), C_ALLT),
        ("forked allt·50% · 2L", fk("xshh_allt_forked_50k", "2L", "full"), fk("xshh_allt_forked_50k", "2L", "triage"), C_FORK),
        ("forked allt·50% · 6L", fk("xshh_allt_forked_50k", "6L", "full"), fk("xshh_allt_forked_50k", "6L", "triage"), C_FORK),
        ("forked allt·0.8% · 2L", fk("xshh_allt_forked2_50k", "2L", "full"), fk("xshh_allt_forked2_50k", "2L", "triage"), C_FORK2),
        ("forked allt·0.8% · 6L", fk("xshh_allt_forked2_50k", "6L", "full"), fk("xshh_allt_forked2_50k", "6L", "triage"), C_FORK2),
        ("forked β·0.8% · 2L", fk("beta_forked2_50k", "2L", "full"), fk("beta_forked2_50k", "2L", "triage"), C_BFORK),
        ("forked β·0.8% · 6L", fk("beta_forked2_50k", "6L", "full"), fk("beta_forked2_50k", "6L", "triage"), C_BFORK),
        ("forked allt·10% · 2L", fk("xshh_allt_forked10pct_50k", "2L", "full"), fk("xshh_allt_forked10pct_50k", "2L", "triage"), C_FORK10),
        ("forked allt·10% · 6L", fk("xshh_allt_forked10pct_50k", "6L", "full"), fk("xshh_allt_forked10pct_50k", "6L", "triage"), C_FORK10),
        ("forked β·10% · 2L", fk("beta_forked10pct_50k", "2L", "full"), fk("beta_forked10pct_50k", "2L", "triage"), C_BFORK10,
         (fk("beta_forked10pct_s2_50k", "2L", "full"), fk("beta_forked10pct_s2_50k", "2L", "triage"))),
        ("forked β·10% · 6L", fk("beta_forked10pct_50k", "6L", "full"), fk("beta_forked10pct_50k", "6L", "triage"), C_BFORK10,
         (fk("beta_forked10pct_s2_50k", "6L", "full"), fk("beta_forked10pct_s2_50k", "6L", "triage"))),
        ("β · 2L", f"{BETA_DIR}/results/gift_eval_full_bb_beta_50k/summary.txt",
         f"{BETA_DIR}/results/gift_eval_triage_bb_beta_50k/summary.txt", C_BETA,
         (fk("beta_s2_50k", "2L", "full"), fk("beta_s2_50k", "2L", "triage"))),
        ("β · 6L", fk("beta_50k", "6L", "full"), fk("beta_50k", "6L", "triage"), C_BETA,
         (fk("beta_s2_50k", "6L", "full"), fk("beta_s2_50k", "6L", "triage"))),
    ]
    # rows: (lab, full_gm, triage_gm, colour, (full_gm_s2, triage_gm_s2) or None)
    rows = []
    for c in cells:
        s2 = (agg_gm(c[4][0]), agg_gm(c[4][1])) if len(c) > 4 else None
        rows.append((c[0], agg_gm(c[1]), agg_gm(c[2]), c[3], s2))
    rows = [r for r in rows if r[1] or r[2]]   # drop arms with no data yet (e.g. forked 2/batch pre-landing)
    if not rows:
        print("gm_summary: no data yet — skip")
        return
    v = agg_gm(f"{V11C}/summary.txt") or 1.292
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, idx, title in ((axes[0], 1, "full-97"), (axes[1], 2, "triage-11")):
        seed_lbl_done = False
        labs = [r[0] for r in rows]
        vals = [r[idx] for r in rows]
        cols = [r[3] for r in rows]
        y = range(len(rows))
        ax.barh(list(y), [x if x else 0 for x in vals], color=cols, alpha=0.85)
        # seed-1 value text: anchor past the whisker end so the marker doesn't
        # land on the number (triage spreads are tiny -> caret sits on the label).
        span = (max(good_all) - min(good_all)) if (good_all := [x for x in vals if x]) else 1.0
        for i, (val, r) in enumerate(zip(vals, rows)):
            if not val:
                continue
            s2 = r[4]
            anchor = val
            if s2 and s2[idx - 1]:
                anchor = max(val, s2[idx - 1]) + 0.022 * span   # clear the caret
            ax.text(anchor + 0.004, i, f"{val:.4f}", va="center", fontsize=9)
        # second-seed whiskers: a thin line seed-1 -> seed-2 with a caret at seed-2.
        for i, r in enumerate(rows):
            v1, s2 = r[idx], r[4]
            if not (s2 and v1 and s2[idx - 1]):
                continue
            v2 = s2[idx - 1]
            lbl = None if seed_lbl_done else "2nd seed (20260521)"
            ax.plot([v1, v2], [i, i], color="k", lw=1.1, alpha=0.85, zorder=5)
            ax.plot([v2], [i], marker=("<" if v2 < v1 else ">"), color="k",
                    ms=6, alpha=0.9, zorder=6, label=lbl)
            ax.plot([v1], [i], marker="|", color="k", ms=7, alpha=0.9, zorder=6)
            seed_lbl_done = True
        ax.set_yticks(list(y)); ax.set_yticklabels(labs, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(f"{title} GM-Relative MASE (lower = better)")
        good = [x for x in vals if x]
        s2_good = [s2[idx - 1] for *_, s2 in rows if s2 and s2[idx - 1]]
        if good:
            ax.set_xlim(min(good + s2_good + [v]) * 0.98, max(good + s2_good) * 1.06)
        if idx == 1:
            ax.axvline(v, color=C_V11C, ls="--", lw=1.4, label=f"v11c = {v:.3f}")
        ax.axvline(1.0, color="k", ls=":", lw=1.0, alpha=0.5)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("#318 — frozen-backbone GM-Relative MASE: shortcut-denial arms (loss-side h↔h · data-side forked) vs β", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/gm_summary.png", dpi=130, bbox_inches="tight")
    print("gm_summary.png written")


# --------------------------------------------------------------- GM vs step
def plot_gm_vs_step():
    steps = [20000, 35000, 50000]
    xshh = {20000: f"{RES}/gift_eval_full_xshh_20k_2L/summary.txt",
            35000: f"{RES}/gift_eval_full_xshh_35k_2L/summary.txt",
            50000: f"{RES}/gift_eval_full_xshh_50k_2L/summary.txt"}
    beta = {20000: f"{RES}/gift_eval_full_beta_20k_2L/summary.txt",
            35000: f"{RES}/gift_eval_full_beta_35k_2L/summary.txt",
            50000: f"{BETA_DIR}/results/gift_eval_full_bb_beta_50k/summary.txt"}
    xs = [(s, agg_gm(xshh[s])) for s in steps if agg_gm(xshh[s])]
    bs = [(s, agg_gm(beta[s])) for s in steps if agg_gm(beta[s])]
    if not xs and not bs:
        print("gm_vs_step: no data yet — skip")
        return
    v = agg_gm(f"{V11C}/summary.txt") or 1.292
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if xs:
        ax.plot([s/1000 for s, _ in xs], [g for _, g in xs], "o-", color=C_XSHH, lw=2, label="xshh (this card)")
    if bs:
        ax.plot([s/1000 for s, _ in bs], [g for _, g in bs], "s-", color=C_BETA, lw=2, label="β (#309)")
    ax.axhline(v, color=C_V11C, ls="--", lw=1.4, label=f"v11c = {v:.3f}")
    ax.set_xlabel("backbone training step (k)")
    ax.set_ylabel("full-97 GM-Relative MASE (2L head, lower = better)")
    ax.set_title("#318 — does more contrastive training stop helping?")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/gm_vs_step.png", dpi=130, bbox_inches="tight")
    print("gm_vs_step.png written")


# --------------------------------------------------------- training curves
def _load_curve(csv_path):
    if not os.path.exists(csv_path):
        return None
    cols = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for k, val in row.items():
                try:
                    cols.setdefault(k, []).append(float(val))
                except (ValueError, TypeError):
                    pass
    return cols


def _smooth(y, w=200):
    if len(y) < w:
        return y
    out, acc = [], 0.0
    from collections import deque
    q = deque()
    for v in y:
        q.append(v); acc += v
        if len(q) > w:
            acc -= q.popleft()
        out.append(acc/len(q))
    return out


# Skip the warm-up: on a log-x the first ~1000 steps are a transient that
# squashes the informative tail. Crop (step, y) pairs to step >= S0.
S0_PLOT = 1000
def _crop(steps, ys, s0=S0_PLOT):
    z = [(s, y) for s, y in zip(steps, ys) if s >= s0]
    return ([s for s, _ in z], [y for _, y in z]) if z else (steps, ys)


def plot_training_curves():
    # (label, curve, colour, floor-key)
    arms = [("same-step",       _load_curve(XSHH_CSV), C_XSHH, "xshh"),
            ("all-time",        _load_curve(ALLT_CSV), C_ALLT, "allt"),
            ("forked allt·50%", _load_curve(FORK_CSV), C_FORK, "allt"),
            ("forked allt·0.8%", _load_curve(FORK2_CSV), C_FORK2, "allt"),
            ("forked β·0.8%",    _load_curve(BFORK_CSV), C_BFORK, "beta"),
            ("forked allt·10%", _load_curve(FORK10_CSV), C_FORK10, "allt"),
            ("forked β·10%",    _load_curve(BFORK10_CSV), C_BFORK10, "beta"),
            ("β",               _load_curve(BETA_CSV), C_BETA, "beta")]
    arms = [a for a in arms if a[1]]   # pending arms (no CSV yet) drop out
    if not arms:
        print("training_curves: no data yet — skip")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # All panels: log–log, warm-up cropped (step ≥ 1000).
    # Panel 1: contrastive loss MINUS each arm's InfoNCE floor. The floor
    # log(1+N·e^(−1/τ)) grows ~52× with the negative count N, so RAW losses
    # aren't comparable — subtracting each arm's own floor is. Floor-excess stays
    # strictly positive (≈ +0.5–0.6 at 50k), so log–log is well-defined.
    ax = axes[0, 0]
    for lab, cols, col, fk in arms:
        if "loss" in cols and "step" in cols:
            xs, ys = _crop(cols["step"], _smooth([v - FLOOR[fk] for v in cols["loss"]], 100))
            ax.plot(xs, ys, color=col, lw=1.4, label=f"{lab}  (floor {FLOOR[fk]:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("contrastive loss − InfoNCE floor   (log–log)")
    ax.set_xlabel("step"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)

    # Panel 2: gap-ratio = (1−ff)/(1−fp) (ff = forecast↔future, fp = forecast↔present
    # cosine). In (0,1) for a skilled forecaster; 0 = perfect. Direct log–log (the
    # 2026-05-10_exp_transformer_encoder recipe). Lower = better.
    ax = axes[0, 1]
    for lab, cols, col, _ in arms:
        if "gap_ratio" in cols and "step" in cols:
            xs, ys = _crop(cols["step"], _smooth([max(v, 1e-4) for v in cols["gap_ratio"]], 100))
            ax.plot(xs, ys, color=col, lw=1.4, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("gap-ratio (1−ff)/(1−fp)   (log–log; lower = better, 0 = perfect)")
    ax.set_xlabel("step"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)

    # Panels 3–4: dimension usage U = 1/(d·mean_{i≠j} cos²) ∈ (0,1] (src/metrics.py).
    # 1 = latents span all d dims; →0 = dimensional collapse. Higher = better.
    # u_temporal (spread across time) and u_batch (across series).
    for ax, key, title in ((axes[1, 0], "u_temporal", "dimension usage — temporal"),
                           (axes[1, 1], "u_batch", "dimension usage — batch")):
        for lab, cols, col, _ in arms:
            if key in cols and "step" in cols:
                xs, ys = _crop(cols["step"], _smooth([max(v, 1e-4) for v in cols[key]], 100))
                ax.plot(xs, ys, color=col, lw=1.4, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{title}   (log–log; higher = better)")
        ax.set_xlabel("step"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)

    fig.suptitle("#318 — contrastive training dynamics (100-step MA, step ≥ 1000; all log–log)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/training_curves.png", dpi=175, bbox_inches="tight")
    print("training_curves.png written")


# ------------------------------------------------------------- per-domain
def plot_perdomain():
    dom_map = (config_domain(f"{BETA_DIR}/results/gift_eval_full_bb_beta_50k/all_results.csv")
               or config_domain(f"{RES}/gift_eval_full_xshh_50k_2L/all_results.csv"))
    if not dom_map:
        print("perdomain: no domain map — skip")
        return
    import numpy as np

    # One radar: per arm, keep the q-head with the LOWER full-97 aggregate (its
    # best head) and plot that head's per-domain profile; the head is shown in
    # the legend. We never compute an eval here — just plot what the pipeline has.
    # ALL arms are shown (not a curated 5). Two arms carry a 2nd seed (20260521):
    # forked β·10% and β. For those we plot a shaded band spanning the two seeds'
    # per-domain values (each seed at its OWN best head); single-seed arms = lines.
    # `seeds` is a list of {head: summary} dicts (one per seed).
    arms = [
        ("same-step", C_XSHH, [{h: f"{RES}/gift_eval_full_xshh_50k_{h}/summary.txt" for h in ("2L", "6L")}]),
        ("all-time", C_ALLT, [{h: f"{RES}/gift_eval_full_xshh_allt_50k_{h}/summary.txt" for h in ("2L", "6L")}]),
        ("forked allt·50%", C_FORK, [{h: f"{RES}/gift_eval_full_xshh_allt_forked_50k_{h}/summary.txt" for h in ("2L", "6L")}]),
        ("forked allt·0.8%", C_FORK2, [{h: f"{RES}/gift_eval_full_xshh_allt_forked2_50k_{h}/summary.txt" for h in ("2L", "6L")}]),
        ("forked allt·10%", C_FORK10, [{h: f"{RES}/gift_eval_full_xshh_allt_forked10pct_50k_{h}/summary.txt" for h in ("2L", "6L")}]),
        ("forked β·0.8%", C_BFORK, [{h: f"{RES}/gift_eval_full_beta_forked2_50k_{h}/summary.txt" for h in ("2L", "6L")}]),
        ("forked β·10%", C_BFORK10, [
            {h: f"{RES}/gift_eval_full_beta_forked10pct_50k_{h}/summary.txt" for h in ("2L", "6L")},
            {h: f"{RES}/gift_eval_full_beta_forked10pct_s2_50k_{h}/summary.txt" for h in ("2L", "6L")}]),
        ("β", C_BETA, [
            {"2L": f"{BETA_DIR}/results/gift_eval_full_bb_beta_50k/summary.txt",
             "6L": f"{RES}/gift_eval_full_beta_50k_6L/summary.txt"},
            {h: f"{RES}/gift_eval_full_beta_s2_50k_{h}/summary.txt" for h in ("2L", "6L")}]),
        ("v11c", C_V11C, [{"2L": f"{V11C}/summary.txt"}]),
    ]

    def best_profile(heads):
        """(best_head, {domain: gm relative}) for the lowest-full-97 head, or None."""
        cand = [(g, h, st) for h, st in heads.items() if (g := agg_gm(st))]
        if not cand:
            return None
        _, best_h, st = min(cand)
        byd = {}
        for cfg, r in per_config_relative(st).items():
            byd.setdefault(dom_map.get(cfg, "?"), []).append(r)
        return best_h, {d: gm(v) for d, v in byd.items()}

    # series: (tag, colour, list[(head,{dom:val})]) — 1 entry=line, 2 entries=band.
    series = []
    for lab, c, seeds in arms:
        profs = [p for p in (best_profile(s) for s in seeds) if p]
        if not profs:
            continue
        if lab == "v11c":
            tag = lab
        elif len(profs) == 1:
            tag = f"{lab} ({profs[0][0]})"
        else:                                            # banded multi-seed arm
            tag = f"{lab} ({'/'.join(p[0] for p in profs)}, 2 seeds)"
        series.append((tag, c, profs))
    if not series:
        print("perdomain: no data yet — skip")
        return
    doms = sorted({d for _, _, profs in series for _, m in profs for d in m})
    ang = np.linspace(0, 2 * np.pi, len(doms), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    fig, ax = plt.subplots(figsize=(9.5, 8), subplot_kw=dict(polar=True))
    rmax = max(v for _, _, profs in series for _, m in profs for v in m.values())
    # Draw single-seed "context" arms first (thin, lightly muted), then the two
    # banded multi-seed arms on top so the headline seed comparison stays readable
    # in the crowded centre. Thin lines, no markers — 9 arms + markers is a mess.
    singles = [s for s in series if len(s[2]) == 1]
    banded = [s for s in series if len(s[2]) == 2]
    for tag, c, profs in singles:
        lw = 1.8 if tag == "v11c" else 1.2
        _, m = profs[0]
        v = [m.get(d, np.nan) for d in doms]
        ax.plot(ang_c, v + v[:1], "-", color=c, lw=lw, alpha=0.8, label=tag, zorder=3)
    for tag, c, profs in banded:                         # seed band, drawn on top
        lo = np.array([min(profs[0][1].get(d, np.nan), profs[1][1].get(d, np.nan)) for d in doms])
        hi = np.array([max(profs[0][1].get(d, np.nan), profs[1][1].get(d, np.nan)) for d in doms])
        ax.fill_between(ang_c, np.append(lo, lo[0]), np.append(hi, hi[0]),
                        color=c, alpha=0.28, lw=0, zorder=4)
        for _, m in profs:                               # each seed faint
            v = [m.get(d, np.nan) for d in doms]
            ax.plot(ang_c, v + v[:1], "-", color=c, lw=0.9, alpha=0.5, zorder=5)
        mid = (lo + hi) / 2.0                             # bold mid line carries label
        ax.plot(ang_c, np.append(mid, mid[0]), "-", color=c, lw=2.4, alpha=0.98, label=tag, zorder=6)
    ax.plot(np.linspace(0, 2 * np.pi, 200), [1.0] * 200, "--", color="k", lw=1.0, alpha=0.6)
    ax.set_rscale("log")
    ax.set_rlim(0.75, rmax * 1.12)
    ax.set_xticks(ang); ax.set_xticklabels(doms, fontsize=9)
    ax.set_rlabel_position(95)
    ax.set_title("#318 — per-domain GM-Relative MASE (full-97, best q-head per arm · log radial)\n"
                 "lower = better; dashed ring = seasonal-naive (1.0); shaded = seed-1↔seed-2 spread", fontsize=10.5, pad=28)
    ax.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.36, 1.13))
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/perdomain.png", dpi=130, bbox_inches="tight")
    print("perdomain.png written (radar, all arms, best head per arm; 2 banded seeds)")
    plt.close(fig)


if __name__ == "__main__":
    plot_gm_summary()
    # plot_gm_vs_step() — deferred: only the 50k endpoints exist, so the curve
    # would be two lone points. Re-enable once a step-resolved sweep is run.
    plot_training_curves()
    plot_perdomain()
