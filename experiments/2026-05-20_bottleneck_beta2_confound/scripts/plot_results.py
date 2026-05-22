#!/usr/bin/env python3
"""#309 bottleneck × β2 × τ: per-domain radars + training curves.

Two per-domain radars (τ=0.1 and τ=0.8), each with the v11c reference
ring (dashed purple) and the seasonal-naive ring. Training curves
(log/log) on the τ=0.1 set show fp16-divergence vs fp32-stability.

Robust to missing arms: an arm with no losses CSV / no full-eval
summary is silently skipped so the script can preview while runs are
still in flight.
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

V11C_ARM = ("v11c (ref, all-fp32 no-bneck)", "#9467bd", None, V11C, (0, (4, 2)), 1.4, 0.0)

# τ=0.1: snapshot (fp16, pre-divergence) vs converged (fp32 50k) vs (B) + v11c.
# tuple = (label, colour, losses_csv, full_eval_dir, ls, lw, fill)
TAU01_ARMS = [
    ("(B) bneck fp16 50k", "#1f77b4",
     f"{CL_ABL}/runs/cl_hh_50k_losses.csv",
     f"{CL_ABL}/results/gift_eval_full_cl_hh_50k", "-", 2.0, 0.0),
    ("α no-bneck fp16 snap", "#2ca02c",
     f"{MAIN}/runs/bb_alpha_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_alpha_50k", "-", 2.0, 0.0),
    ("γ no-bneck fp16 snap", "#17becf",
     f"{MAIN}/runs/bb_gamma_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_gamma_50k", "-", 2.0, 0.0),
    ("γ no-bneck fp32 50k (best converged)", "#ff7f0e",
     f"{MAIN}/runs/bb_gamma_tau01_fp32_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_gamma_tau01_fp32_50k", "-", 2.0, 0.0),
    ("α no-bneck fp32 50k (worst)", "#d62728",
     f"{MAIN}/runs/bb_alpha_tau01_fp32_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_alpha_tau01_fp32_50k", "-", 2.0, 0.0),
    V11C_ARM,
]

# τ=0.8: β (bneck, fp16, converged — matches v11c) vs α/γ (no-bneck fp32 50k) + v11c.
TAU08_ARMS = [
    ("β bneck fp16 50k (matches v11c)", "#1f77b4",
     f"{MAIN}/runs/bb_beta_tau08_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_beta_tau08_50k", "-", 2.0, 0.0),
    ("α no-bneck fp32 50k", "#d62728",
     f"{MAIN}/runs/bb_alpha_tau08_fp32_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_alpha_tau08_fp32_50k", "-", 2.0, 0.0),
    ("γ no-bneck fp32 50k", "#ff7f0e",
     f"{MAIN}/runs/bb_gamma_tau08_fp32_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_gamma_tau08_fp32_50k", "-", 2.0, 0.0),
    V11C_ARM,
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


draw_radar(TAU01_ARMS, f"{OUT}/perdomain_star_tau01.png",
           "#309 — full GIFT-Eval (97 cfg) per domain · τ=0.1")
draw_radar(TAU08_ARMS, f"{OUT}/perdomain_star_tau08.png",
           "#309 — full GIFT-Eval (97 cfg) per domain · τ=0.8")


# ---------- training curves ----------
def load_csv(path):
    if not path or not os.path.exists(path): return None
    with open(path) as f:
        return list(csv.DictReader(f))


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


# τ=0.8 curve set: the killed fp16 no-bneck α (diverges ~step 3.5k) +
# the bneck fp16 β + the two fresh-fp32 no-bneck arms (all stable). γ
# τ=0.8 fp16 was never run (went straight to fresh fp32), so only α
# illustrates the τ=0.8 fp16 divergence.
TAU08_CURVE_ARMS = [
    ("α no-bneck fp16 (diverges)", "#2ca02c",
     f"{MAIN}/runs/bb_alpha_tau08_50k_losses.csv", None, "-", 2.0, 0.0),
    ("β bneck fp16 50k", "#1f77b4",
     f"{MAIN}/runs/bb_beta_tau08_50k_losses.csv", None, "-", 2.0, 0.0),
    ("α no-bneck fp32 50k", "#d62728",
     f"{MAIN}/runs/bb_alpha_tau08_fp32_50k_losses.csv", None, "-", 2.0, 0.0),
    ("γ no-bneck fp32 50k", "#ff7f0e",
     f"{MAIN}/runs/bb_gamma_tau08_fp32_50k_losses.csv", None, "-", 2.0, 0.0),
]

draw_curves(TAU01_ARMS, f"{OUT}/training_curves_tau01.png",
            "#309 training curves · τ=0.1  (fp16 no-bneck diverges; fp32 stable)")
draw_curves(TAU08_CURVE_ARMS, f"{OUT}/training_curves_tau08.png",
            "#309 training curves · τ=0.8  (fp16 no-bneck diverges; fp32 + bneck-fp16 stable)")
