#!/usr/bin/env python3
"""#309 bottleneck × β2 confound: figures.

Per-domain radar (full GIFT-Eval): four arms (B, α, β, γ) plus v11c
reference. Training curves (log/log): loss, loss_tau_ref, gap, 1−AUC.

Robust to missing arms: an arm with no losses CSV / no full-eval
summary is silently skipped so the script can preview while runs are
still in flight.
"""
import csv, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paths — all data lives in MAIN checkout (CLAUDE.md rule).
MAIN = "/home/jupyter/contrastive-forecasting/experiments/2026-05-20_bottleneck_beta2_confound"
CL_ABL = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation"
V11C = "/home/jupyter/contrastive-forecasting/experiments/2026-05-11_exp_encoder_forecaster/results/gift_eval_full_v11c"
OUT = f"{MAIN}/plots"
os.makedirs(OUT, exist_ok=True)

# label, colour, losses_csv, full_eval_dir, ls, lw, fill
ARMS = [
    ("(B) bneck+β2=0.95", "#1f77b4",
     f"{CL_ABL}/runs/cl_hh_50k_losses.csv",
     f"{CL_ABL}/results/gift_eval_full_cl_hh_50k",
     "-", 2.0, 0.10),
    ("α no-bneck+β2=0.98", "#d62728",
     f"{MAIN}/runs/bb_alpha_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_alpha_50k",
     "-", 2.0, 0.10),
    ("β bneck+β2=0.98", "#2ca02c",
     f"{MAIN}/runs/bb_beta_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_beta_50k",
     "-", 2.0, 0.10),
    ("γ no-bneck+β2=0.95", "#ff7f0e",
     f"{MAIN}/runs/bb_gamma_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_gamma_50k",
     "-", 2.0, 0.10),
    ("α fp32-cont (no-bneck, 50k)", "#8c564b",
     f"{MAIN}/runs/bb_alpha_fp32cont_50k_losses.csv",
     f"{MAIN}/results/gift_eval_full_bb_alpha_fp32cont_50k",
     "-", 2.0, 0.10),
    ("v11c (ref, all-fp32 no-bneck)", "#9467bd",
     None,
     V11C,
     (0, (4, 2)), 1.2, 0.0),  # thin purple dashed, no fill
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


# ---------- per-domain star ----------
radar = []
for lab, col, _, edir, ls, lw, fill in ARMS:
    g = rel_by_domain(f"{edir}/summary.txt", dom_map(f"{edir}/all_results.csv"))
    if not g: continue
    radar.append((lab, col, g, agg_gm(f"{edir}/summary.txt"), ls, lw, fill))

if radar:
    domains = sorted({d for _, _, g, *_ in radar for d in g})
    N = len(domains)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_xticks(theta[:-1]); ax.set_xticklabels(domains, fontsize=9)
    ax.set_rscale("log")
    # Unit circle (seasonal naive) reference
    ax.plot(theta, [1.0]*len(theta), color="k", ls="--", lw=0.8, alpha=0.5)
    ax.text(0, 1.0, "  SN=1.0", fontsize=8, color="k")
    for lab, col, g, gm, ls, lw, fill in radar:
        v = [g.get(d, np.nan) for d in domains] + [g.get(domains[0], np.nan)]
        ax.plot(theta, v, color=col, ls=ls, lw=lw, label=f"{lab}  GM={gm:.3f}" if gm else lab)
        if fill > 0: ax.fill(theta, v, color=col, alpha=fill)
    ax.set_ylim(0.8, 3.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    ax.set_title("#309 — full GIFT-Eval (97 cfg), GM-Relative MASE per domain", pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUT}/perdomain_star.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT}/perdomain_star.png — arms={len(radar)} domains={N}")
else:
    print("radar: no arm has full-eval summary yet — skipping")


# ---------- training curves ----------
def load_csv(path):
    if not path or not os.path.exists(path): return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


curves = []
for lab, col, csv_path, _edir, ls, lw, fill in ARMS:
    rows = load_csv(csv_path)
    if rows is None: continue
    curves.append((lab, col, rows, ls, lw))

if not curves:
    print("curves: no losses CSV yet — skipping")
else:
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ("loss", "loss"),
        ("loss_tau_ref (norm InfoNCE)", "loss_tau_ref"),
        ("1 − AUC", "auc"),
        ("gap", "gap"),
    ]
    for ax, (title, col_key) in zip(axs.flat, panels):
        for lab, c, rows, ls, lw in curves:
            xs, ys = [], []
            for r in rows:
                try:
                    s = int(r["step"]); y = float(r[col_key])
                except (KeyError, ValueError):
                    continue
                if title == "1 − AUC":
                    y = 1.0 - y
                if y <= 0:
                    continue
                xs.append(s); ys.append(y)
            if not xs: continue
            # downsample for legibility
            if len(xs) > 800:
                idx = np.linspace(0, len(xs)-1, 800).astype(int)
                xs = [xs[i] for i in idx]; ys = [ys[i] for i in idx]
            ax.plot(xs, ys, color=c, ls=ls, lw=lw, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title); ax.set_xlabel("step"); ax.grid(True, alpha=0.3)
    axs[0, 0].legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/training_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT}/training_curves.png — arms={len(curves)}")
