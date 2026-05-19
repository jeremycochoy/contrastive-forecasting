#!/usr/bin/env python3
"""#307 cross-branch ablation figures (continuation of #303).

(1) perdomain_star.png — per-domain held-out relative MASE radar. The 3
    NEW arms (hhff=B+C, fhhhff=A+B+C, hhxbf=B-xbranch-free) plus the kept
    #303 arms (A full_fh, B full_hh, C full_ff, A+B full_fh_hh) + the
    same v11c / best-ever reference rings as #303.
(2) training_curves.png — loss · u_temporal · u_batch · loss_tau_ref,
    all log–log, NEW arms overlaid with the kept #303 arms. FIRST 100
    TRAIN STEPS SKIPPED (readability). NO 1−AUC panel (not informative
    here — issue #307).

Robust to missing arms (skips silently) so it previews mid-run.
"""
import csv, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART303 = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation"
A17 = "/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-17_bottleneck_fullfh_ddp"
SY = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_xbranch_ablation"
OUT = "/home/jupyter/cf-wt-crossed-loss/experiments/2026-05-19_crossed_loss_xbranch_ablation/plots"
os.makedirs(OUT, exist_ok=True)
A_NAME = "enc_fcst_bneck128_dk07_fullfh_norminfonce_1L_fp16_ddp128_50k"

# label, colour, losses_csv, full_eval_dir, linestyle, lw, is_new
ARMS = [
    ("(A) full_fh_negs #296", "#7f7f7f",
     f"{ART303}/results/A_ref_losses_clean.csv",
     f"{A17}/results/gift_eval_full_{A_NAME}", "-", 1.4, False),
    ("(B) full_hh_negs", "#1f77b4", f"{ART303}/runs/cl_hh_50k_losses.csv",
     f"{ART303}/results/gift_eval_full_cl_hh_50k", "-", 1.4, False),
    ("(C) full_ff_negs", "#2ca02c", f"{ART303}/runs/cl_ff_50k_losses.csv",
     f"{ART303}/results/gift_eval_full_cl_ff_50k", "-", 1.4, False),
    ("(A)+(B) full_fh_hh", "#ff7f0e", f"{ART303}/runs/cl_fhhh_50k_losses.csv",
     f"{ART303}/results/gift_eval_full_cl_fhhh_50k", "-", 1.4, False),
    ("(B)+(C) full_hh_ff [#307]", "#d62728",
     f"{SY}/sync_hhff/runs/cl_hhff_50k_losses.csv",
     f"{SY}/sync_hhff/results/gift_eval_full_cl_hhff_50k", "-", 2.4, True),
    ("(A)+(B)+(C) full_fh_hh_ff [#307]", "#9467bd",
     f"{SY}/sync_fhhhff/runs/cl_fhhhff_50k_losses.csv",
     f"{SY}/sync_fhhhff/results/gift_eval_full_cl_fhhhff_50k", "-", 2.4, True),
    ("(B) xbranch-free [#307]", "#17becf",
     f"{SY}/sync_hhxbf/runs/cl_hhxbf_50k_losses.csv",
     f"{SY}/sync_hhxbf/results/gift_eval_full_cl_hhxbf_50k", "-", 2.4, True),
]
REFS = [
    ("best-ever · xfmr-q 12L (#127)", "#b8860b", "--", 2.2,
     "/home/jupyter/contrastive-forecasting/experiments/"
     "2026-05-05_exp_qhead_improvements/results/"
     "R9_E13_xfmr12L_quant_moirai_cosine_e_then_f_60k_full"),
    ("v11c · best enc-fcst", "#8c564b", (0, (5, 2)), 1.4,
     "/home/jupyter/contrastive-forecasting/experiments/"
     "2026-05-11_exp_encoder_forecaster/results/gift_eval_full_v11c"),
]


def dom_map(p):
    m = {}
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            m[r["dataset"]] = r.get("domain", "?")
    return m


def rel_by_domain(sum_txt, dmap):
    if not os.path.exists(sum_txt) or not dmap:
        return {}
    acc = {}
    for line in open(sum_txt):
        p = line.split()
        if len(p) < 4:
            continue
        try:
            rel = float(p[-1])
        except ValueError:
            continue
        if p[0] in dmap and rel > 0:
            acc.setdefault(dmap[p[0]], []).append(math.log(rel))
    return {d: math.exp(sum(v) / len(v)) for d, v in acc.items()}


def agg_gm(sum_txt):
    if not os.path.exists(sum_txt):
        return None
    for line in open(sum_txt):
        if "Aggregate GM-Relative MASE" in line:
            for t in reversed(line.replace(":", " ").split()):
                try:
                    return float(t)
                except ValueError:
                    pass
    return None


radar = []
for lab, col, _, ed, ls, lw, new in ARMS:
    g = rel_by_domain(f"{ed}/summary.txt", dom_map(f"{ed}/all_results.csv"))
    if g:
        radar.append((lab, col, g, agg_gm(f"{ed}/summary.txt"), ls,
                       lw + (0.6 if new else 0), new))
for lab, col, ls, lw, d in REFS:
    g = rel_by_domain(f"{d}/summary.txt", dom_map(f"{d}/all_results.csv"))
    if g:
        radar.append((lab, col, g, agg_gm(f"{d}/summary.txt"), ls, lw, False))

if radar:
    doms = sorted(set().union(*[set(g) for _, _, g, _, _, _, _ in radar]))
    ang = np.linspace(0, 2 * np.pi, len(doms), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    for lab, col, g, gm, ls, lw, new in radar:
        v = [g.get(d, np.nan) for d in doms] + [g.get(doms[0], np.nan)]
        ax.plot(ang, v, lw=lw, ls=ls, color=col,
                label=lab + (f"  — full GM {gm:.3f}" if gm else ""))
        if new:
            ax.fill(ang, v, alpha=.05, color=col)
    ax.plot(ang, [1.0] * len(ang), lw=1.0, ls=":", color="green",
            label="seasonal naive (=1.0)")
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(doms, fontsize=10)
    ax.set_title("#307 cross-branch ablation — held-out rel-MASE per "
                 "GIFT-Eval domain\n(thick = #307 new arms; thin = #303 "
                 "kept; lower=better; dotted=seasonal naive)", fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.11), fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/perdomain_star.png", dpi=140)
    print("radar:", [(l, round(gm, 4) if gm else None)
                     for l, _, _, gm, _, _, _ in radar])
else:
    print("radar SKIPPED (no full-eval summaries yet)")


def load(p):
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    out = {}
    for k in rows[0]:
        a = []
        for r in rows:
            try:
                a.append(float(r[k]))
            except (ValueError, TypeError):
                a.append(math.nan)
        out[k] = np.array(a)
    return out


def msk(s, y):
    m = np.isfinite(s) & np.isfinite(y) & (s > 100) & (y > 0)  # skip first 100
    return s[m], y[m]


def smooth(y):
    y = np.asarray(y, float); n = len(y)
    if n < 8:
        return y
    w = max(21, n // 400) | 1
    w = min(w, n if n % 2 else n - 1)
    return np.nanmedian(
        np.lib.stride_tricks.sliding_window_view(np.pad(y, w // 2, "edge"), w),
        axis=1)


series = [(l, c, ls, lw, load(lc)) for l, c, lc, _, ls, lw, _ in ARMS]
series = [(l, c, ls, lw, d) for l, c, ls, lw, d in series
          if d is not None and len(d.get("step", [])) > 1]
if series:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panels = [("loss", "contrastive loss (--pos-in-denominator)"),
              ("u_temporal", "latent dim usage — u_temporal"),
              ("u_batch", "latent dim usage — u_batch"),
              ("loss_tau_ref", "reference loss loss_tau_ref (fixed-τ)")]
    for ax, (key, title) in zip(axes.flat, panels):
        for lab, col, ls, lw, d in series:
            if key not in d:
                continue
            x, yy = msk(d["step"], d[key])
            if len(x) == 0:
                continue
            ax.plot(x, yy, lw=.5, color=col, ls=ls, alpha=.15)
            xs, ys = msk(d["step"], smooth(d[key]))
            ax.plot(xs, ys, lw=lw, color=col, ls=ls, alpha=.95, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("step (log, first 100 skipped)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=.4)
        ax.legend(fontsize=7)
    fig.suptitle("#307 cross-branch ablation — training curves "
                 "(log–log; first 100 steps skipped; no 1−AUC)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/training_curves.png", dpi=130)
    print("curves:", [l for l, _, _, _, _ in series])
else:
    print("curves SKIPPED (no losses CSVs yet)")
