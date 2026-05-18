#!/usr/bin/env python3
"""#303 crossed-loss ablation figures.

(1) Per-domain star/radar of held-out relative MASE — distance from
    centre = GM relative MASE per GIFT-Eval domain (lower better; unit
    circle = seasonal naive). Four arms: (A) full_fh_negs [the
    backbone-of-record from #296], (B) full_hh_negs, (C) full_ff_negs,
    (A)+(B) full_fh_hh_negs. Style/method per
    ../2026-05-18_qhead150k_on_150kbb/scripts/plot_qhead150k.py.

(2) Training curves, all log–log, four arms overlaid:
    loss · dimension usage (u_temporal, u_batch) · reference loss
    (loss_tau_ref) · 1−AUC.

Robust to missing arms: an arm with no losses CSV / no full-eval
summary is silently skipped, so the script can be run for previews
while runs are still in flight.
"""
import csv, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A17 = "/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-17_bottleneck_fullfh_ddp"
ART = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation"
OUT = "/home/jupyter/cf-wt-crossed-loss/experiments/2026-05-19_crossed_loss_ablation/plots"
os.makedirs(OUT, exist_ok=True)

A_NAME = "enc_fcst_bneck128_dk07_fullfh_norminfonce_1L_fp16_ddp128_50k"
# label, colour, losses_csv, full_eval_dir
ARMS = [
    ("(A) full_fh_negs", "#7f7f7f",
     f"{A17}/runs/{A_NAME}_losses.csv",
     f"{A17}/results/gift_eval_full_{A_NAME}"),
    ("(B) full_hh_negs", "#1f77b4",
     f"{ART}/runs/cl_hh_50k_losses.csv",
     f"{ART}/results/gift_eval_full_cl_hh_50k"),
    ("(C) full_ff_negs", "#2ca02c",
     f"{ART}/runs/cl_ff_50k_losses.csv",
     f"{ART}/results/gift_eval_full_cl_ff_50k"),
    ("(A)+(B) full_fh_hh_negs", "#d62728",
     f"{ART}/runs/cl_fhhh_50k_losses.csv",
     f"{ART}/results/gift_eval_full_cl_fhhh_50k"),
]


# ---------- (1) per-domain radar ----------
def dom_map(ar_csv):
    m = {}
    if not os.path.exists(ar_csv):
        return m
    with open(ar_csv) as f:
        for r in csv.DictReader(f):
            m[r["dataset"]] = r.get("domain", "?")
    return m


def rel_by_domain(sum_txt, dmap):
    """summary.txt rows: Config MASE SN_MASE Relative — GM(Relative)/domain."""
    if not os.path.exists(sum_txt) or not dmap:
        return {}
    acc = {}
    with open(sum_txt) as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            cfg = p[0]
            try:
                rel = float(p[-1])
            except ValueError:
                continue
            if cfg not in dmap or rel <= 0:
                continue
            acc.setdefault(dmap[cfg], []).append(math.log(rel))
    return {d: math.exp(sum(v) / len(v)) for d, v in acc.items()}


def agg_gm(sum_txt):
    if not os.path.exists(sum_txt):
        return None
    with open(sum_txt) as f:
        for line in f:
            if "Aggregate GM-Relative MASE" in line:
                tok = line.replace(":", " ").split()
                for t in reversed(tok):
                    try:
                        return float(t)
                    except ValueError:
                        continue
    return None


radar = []  # (label, colour, {domain: gm})
for lab, col, _, edir in ARMS:
    g = rel_by_domain(f"{edir}/summary.txt", dom_map(f"{edir}/all_results.csv"))
    if g:
        radar.append((lab, col, g, agg_gm(f"{edir}/summary.txt")))

if radar:
    doms = sorted(set().union(*[set(g) for _, _, g, _ in radar]))
    ang = np.linspace(0, 2 * np.pi, len(doms), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for lab, col, g, gm in radar:
        v = [g.get(d, np.nan) for d in doms] + [g.get(doms[0], np.nan)]
        tag = f"{lab}" + (f"  (full GM {gm:.3f})" if gm else "")
        ax.plot(ang, v, lw=1.7, label=tag, color=col)
        ax.fill(ang, v, alpha=.06, color=col)
    ax.plot(ang, [1.0] * len(ang), lw=1.0, ls="--", color="green",
            label="seasonal naive (=1.0)")
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(doms, fontsize=8)
    ax.set_title("#303 — held-out relative MASE per GIFT-Eval domain\n"
                 "(distance from centre = GM rel-MASE; lower better; "
                 "dashed = seasonal naive)", fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/perdomain_star.png", dpi=130)
    print("radar arms:", [(l, round(gm, 4) if gm else None)
                          for l, _, _, gm in radar])
else:
    print("radar SKIPPED — no full-eval summaries yet")


# ---------- (2) training curves (log–log) ----------
def load(csv_path):
    cols = {}
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        rd = csv.DictReader(f)
        for k in rd.fieldnames:
            cols[k] = []
        for r in rd:
            for k in rd.fieldnames:
                cols[k].append(r[k])
    out = {}
    for k, vs in cols.items():
        arr = []
        for x in vs:
            try:
                arr.append(float(x))
            except (ValueError, TypeError):
                arr.append(math.nan)
        out[k] = np.array(arr)
    return out


def posmask(step, y):
    m = np.isfinite(step) & np.isfinite(y) & (step > 0) & (y > 0)
    return step[m], y[m]


series = [(lab, col, load(lc)) for lab, col, lc, _ in ARMS]
series = [(l, c, d) for l, c, d in series if d is not None and len(d.get("step", [])) > 1]
if series:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        ("loss", [("loss", None)], "contrastive loss (--pos-in-denominator)"),
        ("dim", [("u_temporal", "-"), ("u_batch", "--")],
         "latent dimension usage"),
        ("tauref", [("loss_tau_ref", None)],
         "reference loss  loss_tau_ref (fixed-τ diagnostic)"),
        ("auc", [("__one_minus_auc__", None)], "1 − AUC"),
    ]
    for ax, (key, fields, title) in zip(axes.flat, panels):
        for lab, col, d in series:
            st = d["step"]
            for fld, ls in fields:
                if fld == "__one_minus_auc__":
                    y = 1.0 - d["auc"]
                    lbl = lab
                elif fld in d:
                    y = d[fld]
                    lbl = lab if ls in (None, "-") else None
                else:
                    continue
                x, yy = posmask(st, y)
                if len(x) == 0:
                    continue
                ax.plot(x, yy, lw=.8, color=col, ls=(ls or "-"),
                        label=(lbl if lbl else None))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("step (log)"); ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=.4)
        ax.legend(fontsize=7)
    axes.flat[1].set_ylabel("u_temporal (solid) / u_batch (dashed)")
    fig.suptitle("#303 crossed-loss ablation — training curves (log–log)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/training_curves.png", dpi=130)
    print("curves arms:", [l for l, _, _ in series])
else:
    print("curves SKIPPED — no losses CSVs yet")
