#!/usr/bin/env python3
"""#309 — per-domain radars + training curves for the converged arms.

Arms (see RESULTS.md): (B) = bneck/β2.95 baseline; α = no-bneck/β2.98;
β = bneck/β2.98; γ = no-bneck/β2.95. Bottleneck arms ((B), β) train in
fp16; no-bneck arms (α, γ) diverge under fp16 (an fp16-acceleration
failure) and are trained in fp32.

τ=0.1 radar/curves show the 4 converged arms + v11c. τ=0.8 radar/curves
show only the converged τ=0.8 arms — α and γ (fp32) + v11c — because the
fp16 bottleneck arms ((B), β) COLLAPSE at τ=0.8 (top1/gap/auc fall off
after ~step 500; their best_loss checkpoint is a collapse-onset snapshot
@ step ~324 / ~500, not a converged model). That collapse is shown in a
separate tau08_bneck_collapse.png. fp16_divergence.png illustrates the
no-bneck fp16 divergence (τ=0.1, beginning).

Robust to missing arms: skipped if no eval summary / losses CSV yet.
"""
import csv, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAIN = "/home/jupyter/contrastive-forecasting/experiments/2026-05-20_bottleneck_beta2_confound"
CL_ABL = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation"
V11C = "/home/jupyter/contrastive-forecasting/experiments/2026-05-11_exp_encoder_forecaster/results/gift_eval_full_v11c"
OUT = f"{MAIN}/plots"
os.makedirs(OUT, exist_ok=True)

# consistent per-arm colours across τ
C_B, C_A, C_BETA, C_G, C_V = "#7f7f7f", "#d62728", "#1f77b4", "#2ca02c", "#9467bd"
V11C_ARM = ("v11c (ref)", C_V, None, V11C, (0, (4, 2)), 1.4, 0.0)

# tuple = (label, colour, losses_csv, full_eval_dir, ls, lw, fill)
# τ=0.1 — 4 arms, properly trained (bneck fp16; no-bneck fp32).
TAU01_ARMS = [
    ("(B) bneck β2.95", C_B,
     f"{CL_ABL}/runs/cl_hh_50k_losses.csv",
     f"{CL_ABL}/results/gift_eval_full_cl_hh_50k", "-", 2.0, 0.0),
    ("α no-bneck β2.98 (fp32)", C_A,
     f"{MAIN}/runs/bb_alpha_tau01_fp32_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_alpha_tau01_fp32_50k", "-", 2.0, 0.0),
    ("β bneck β2.98", C_BETA,
     f"{MAIN}/runs/bb_beta_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_beta_50k", "-", 2.0, 0.0),
    ("γ no-bneck β2.95 (fp32)", C_G,
     f"{MAIN}/runs/bb_gamma_tau01_fp32_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_gamma_tau01_fp32_50k", "-", 2.0, 0.0),
    V11C_ARM,
]

# τ=0.8 — the CONVERGED τ=0.8 arms only. The fp16 bottleneck arms at
# τ=0.8 ((B), β) collapse after ~step 500 (best_loss = collapse-onset
# snapshot @ step ~324 / ~500, not a converged model) and are excluded
# from the converged radar/curves; their collapse is shown separately in
# tau08_bneck_collapse.png.
TAU08_ARMS = [
    ("α no-bneck β2.98 (fp32)", C_A,
     f"{MAIN}/runs/bb_alpha_tau08_fp32_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_alpha_tau08_fp32_50k", "-", 2.0, 0.0),
    ("γ no-bneck β2.95 (fp32)", C_G,
     f"{MAIN}/runs/bb_gamma_tau08_fp32_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_gamma_tau08_fp32_50k", "-", 2.0, 0.0),
    V11C_ARM,
]

# τ=0.8 fp16 BOTTLENECK collapse illustration: (B) and β at τ=0.8 train
# fine for a few hundred steps then collapse (top1/gap/auc fall off),
# so their best_loss checkpoint is an early snapshot, NOT a converged
# model — the sub-1.29 GIFT-Eval numbers they produce are under-training
# artifacts. These are full-trajectory CSVs (47.7k / 50k steps).
TAU08_COLLAPSE_ARMS = [
    ("(B) bneck fp16 τ0.8 (collapses)", C_B,
     f"{MAIN}/runs/bb_bbase_tau08_50k_losses.csv", None, "-", 2.0, 0.0),
    ("β bneck fp16 τ0.8 (collapses)", C_BETA,
     f"{MAIN}/runs/bb_beta_tau08_50k_losses.csv", None, "-", 2.0, 0.0),
]

# fp16 divergence illustration (τ=0.1, "the beginning"): bneck arms are
# fp16-stable; no-bneck arms diverge under fp16 (→ trained in fp32 above).
DIVERGENCE_ARMS = [
    ("(B) bneck fp16 (stable)", C_B,
     f"{CL_ABL}/runs/cl_hh_50k_losses.csv", None, "-", 2.0, 0.0),
    ("β bneck fp16 (stable)", C_BETA,
     f"{MAIN}/runs/bb_beta_50k_losses.csv", None, "-", 2.0, 0.0),
    ("α no-bneck fp16 (diverges)", C_A,
     f"{MAIN}/runs/bb_alpha_50k_losses.csv", None, "-", 2.0, 0.0),
    ("γ no-bneck fp16 (diverges)", C_G,
     f"{MAIN}/runs/bb_gamma_50k_losses.csv", None, "-", 2.0, 0.0),
]


def dom_map(ar_csv):
    m = {}
    if not os.path.exists(ar_csv): return m
    with open(ar_csv) as f:
        for r in csv.DictReader(f):
            m[r["dataset"]] = r.get("domain", "?")
    return m


def rel_by_domain(sum_txt, dmap):
    if not os.path.exists(sum_txt) or not dmap: return {}
    acc = {}
    with open(sum_txt) as f:
        for line in f:
            p = line.split()
            if len(p) < 4: continue
            cfg = p[0]
            try: rel = float(p[-1])
            except ValueError: continue
            if cfg not in dmap or rel <= 0: continue
            acc.setdefault(dmap[cfg], []).append(math.log(rel))
    return {d: math.exp(sum(v)/len(v)) for d, v in acc.items()}


def agg_gm(sum_txt):
    if not os.path.exists(sum_txt): return None
    with open(sum_txt) as f:
        for line in f:
            if "Aggregate GM-Relative MASE" in line:
                for t in reversed(line.replace(":", " ").split()):
                    try: return float(t)
                    except ValueError: continue
    return None


def load_csv(path):
    if not path or not os.path.exists(path): return None
    with open(path) as f:
        return list(csv.DictReader(f))


def draw_radar(arms, out_path, title):
    radar = []
    for lab, col, _, edir, ls, lw, _fill in arms:
        g = rel_by_domain(f"{edir}/summary.txt", dom_map(f"{edir}/all_results.csv"))
        if g:
            radar.append((lab, col, g, agg_gm(f"{edir}/summary.txt"), ls, lw))
    if not radar:
        print(f"radar {out_path}: no arm eval yet — skipping"); return
    domains = sorted({d for _, _, g, *_ in radar for d in g})
    N = len(domains)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])
    vals = [v for _, _, g, *_ in radar for v in g.values()]
    lo, hi = max(0.5, min(vals) * 0.92), max(vals) * 1.06
    fig, ax = plt.subplots(figsize=(9.5, 9.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(theta); ax.set_xticklabels(domains, fontsize=10)
    ax.set_rscale("log"); ax.set_ylim(lo, hi)
    rticks = [t for t in (0.8, 1.0, 1.5, 2.0, 2.5, 3.0) if lo < t < hi]
    ax.set_yticks(rticks); ax.set_yticklabels([f"{t:g}" for t in rticks], fontsize=8, color="0.4")
    ax.set_rlabel_position(90)
    ax.plot(theta_closed, [1.0]*len(theta_closed), color="k", ls=(0, (2, 2)), lw=1.0, alpha=0.6, zorder=1)
    for lab, col, g, gm, ls, lw in radar:
        v = np.array([g.get(d, np.nan) for d in domains] + [g.get(domains[0], np.nan)])
        ax.plot(theta_closed, v, color=col, ls=ls, lw=lw, zorder=3,
                label=f"{lab}   GM={gm:.3f}" if gm else lab, marker="o", markersize=3)
    ax.set_title(title + "\n(radial = per-domain GM rel-MASE, log; dashed ring = seasonal naive 1.0; lower = better)",
                 fontsize=11, pad=24)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote {out_path} — arms={len(radar)} domains={N} rlim=({lo:.2f},{hi:.2f})")


def draw_curves(arms, out_path, suptitle):
    curves = []
    for lab, col, csv_path, _edir, ls, lw, _fill in arms:
        rows = load_csv(csv_path)
        if rows is not None:
            curves.append((lab, col, rows, ls, lw))
    if not curves:
        print(f"curves {out_path}: no losses CSV yet — skipping"); return
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    panels = [("loss", "loss"), ("loss_tau_ref (norm InfoNCE)", "loss_tau_ref"),
              ("1 − AUC", "auc"), ("gap", "gap")]
    for ax, (title, col_key) in zip(axs.flat, panels):
        for lab, c, rows, ls, lw in curves:
            xs, ys = [], []
            for r in rows:
                try:
                    s = int(r["step"]); y = float(r[col_key])
                except (KeyError, ValueError):
                    continue
                if title == "1 − AUC": y = 1.0 - y
                if y <= 0: continue
                xs.append(s); ys.append(y)
            if not xs: continue
            if len(xs) > 800:
                idx = np.linspace(0, len(xs)-1, 800).astype(int)
                xs = [xs[i] for i in idx]; ys = [ys[i] for i in idx]
            ax.plot(xs, ys, color=c, ls=ls, lw=lw, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title); ax.set_xlabel("step"); ax.grid(True, alpha=0.3)
    axs[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote {out_path} — arms={len(curves)}")


draw_radar(TAU01_ARMS, f"{OUT}/perdomain_star_tau01.png",
           "#309 — full GIFT-Eval (97 cfg) per domain · τ=0.1 · 4 converged arms")
draw_radar(TAU08_ARMS, f"{OUT}/perdomain_star_tau08.png",
           "#309 — full GIFT-Eval (97 cfg) per domain · τ=0.8 · converged arms (α, γ fp32)")
draw_curves(TAU01_ARMS, f"{OUT}/training_curves_tau01.png",
            "#309 training curves · τ=0.1 · 4 arms (B, α, β, γ — converged)")
draw_curves(TAU08_ARMS, f"{OUT}/training_curves_tau08.png",
            "#309 training curves · τ=0.8 · converged arms (α, γ fp32)")
draw_curves(DIVERGENCE_ARMS, f"{OUT}/fp16_divergence.png",
            "#309 fp16 acceleration failure (τ=0.1): no-bneck arms diverge → trained in fp32")
draw_curves(TAU08_COLLAPSE_ARMS, f"{OUT}/tau08_bneck_collapse.png",
            "τ=0.8 collapses the fp16 bottleneck arms (top1/gap collapse after ~step 500 "
            "→ best_loss is an early snapshot, not a converged model)")
