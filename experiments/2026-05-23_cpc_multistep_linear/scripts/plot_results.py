#!/usr/bin/env python3
"""#316 — CPC multi-step linear forecast on β: figures.

Produces (robust to missing inputs — each figure is skipped if its data is
not yet on disk, so the script is safe to run incrementally as evals land):

  gm_summary.png        full-97 GM-MASE bars: CPC arms vs β / v11c / (B).
  perdomain_radar.png   per-domain GM-MASE: CPC (best arm) vs β vs v11c.
  training_curves.png   loss / loss_tau_ref / gap / R²_naive vs step (both seeds).
  gmase_vs_steps.png    full-97 GM-MASE at 10k..50k for CPC vs flat β/v11c.

Reference numbers (full-97 GM-Relative MASE) are read live from the merged
experiment trees so the figures stay grounded:
  v11c = 1.292 (encoder-forecaster champion), β = 1.3272, (B) = 1.3572.
"""
import csv, math, os, re, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/jupyter/contrastive-forecasting/experiments"
MAIN = f"{ROOT}/2026-05-23_cpc_multistep_linear"
RES = f"{MAIN}/results"
RUNS = f"{MAIN}/runs"
OUT = f"{MAIN}/plots"
os.makedirs(OUT, exist_ok=True)

V11C = f"{ROOT}/2026-05-11_exp_encoder_forecaster/results/gift_eval_full_v11c"
BETA = f"{ROOT}/2026-05-20_bottleneck_beta2_confound/results/gift_eval_full_bb_beta_50k"
BB   = f"{ROOT}/2026-05-19_crossed_loss_ablation/results/gift_eval_full_cl_hh_50k"

C_CPC, C_CPC2, C_BETA, C_V, C_BB = "#1f77b4", "#17becf", "#ff7f0e", "#9467bd", "#7f7f7f"

# Seeds present in this experiment (filled at runtime from run logs / runs/).
SEED_A, SEED_B = "20260520", "20260523"


def agg_gm(sum_txt):
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


def dom_map(ar_csv):
    m = {}
    if not os.path.exists(ar_csv):
        return m
    with open(ar_csv) as f:
        for r in csv.DictReader(f):
            m[r["dataset"]] = r.get("domain", "?")
    return m


def rel_by_domain(sum_txt, dmap):
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


def load_csv(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def eval_dir(tag):
    return f"{RES}/gift_eval_full_{tag}"


# Backbone basename for a seed's FINAL checkpoint + head depth → eval tag.
def final_tag(seed, hl, prec="fp16"):
    return f"bb_cpc_k12_s{seed}_{prec}_50k_FINAL_h{hl}L"


def step_tag(seed, step_k, hl, prec="fp16"):
    return f"bb_cpc_k12_s{seed}_{prec}_50k_{step_k}k_h{hl}L"


# ----------------------------------------------------------------------------
def fig_gm_summary():
    v11c, beta, bb = agg_gm(f"{V11C}/summary.txt"), agg_gm(f"{BETA}/summary.txt"), agg_gm(f"{BB}/summary.txt")
    rows = []  # (label, gm, color)
    for seed, cseed in ((SEED_A, C_CPC), (SEED_B, C_CPC2)):
        for hl, hlab in ((2, "small 2L head"), (6, "6L head")):
            gm = agg_gm(f"{eval_dir(final_tag(seed, hl))}/summary.txt")
            if gm is not None:
                rows.append((f"CPC k=12 · s{seed[-2:]} · {hlab}", gm, cseed))
    if not rows:
        print("gm_summary: no CPC evals yet — skipping")
        return
    rows.sort(key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cols = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 2.4))
    y = list(range(len(rows)))
    ax.barh(y, vals, color=cols, alpha=0.88)
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("full-97 GM-Relative MASE (lower = better)")
    refs = [r for r in (v11c, beta, bb) if r]
    lo = (min(vals + refs) if refs else min(vals)) * 0.985
    ax.set_xlim(lo, max(vals + refs) * 1.04)
    if beta:
        ax.axvline(beta, color=C_BETA, ls="--", lw=1.6, label=f"β (bneck, β2=0.98) = {beta:.3f}")
    if v11c:
        ax.axvline(v11c, color=C_V, ls="--", lw=1.6, label=f"v11c (champion) = {v11c:.3f}")
    if bb:
        ax.axvline(bb, color=C_BB, ls=":", lw=1.2, label=f"(B) = {bb:.3f}")
    ax.axvline(1.0, color="k", ls=":", lw=1.0, alpha=0.4, label="seasonal naive = 1.0")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.set_title("#316 — full GIFT-Eval (97 cfg): CPC multi-step linear forecast vs β / v11c", fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(f"{OUT}/gm_summary.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote {OUT}/gm_summary.png — {len(rows)} CPC arms; β={beta} v11c={v11c}")


def fig_radar():
    # CPC best available arm (prefer seed A small head FINAL) vs β vs v11c.
    arms = []
    for seed in (SEED_A, SEED_B):
        for hl in (2, 6):
            ed = eval_dir(final_tag(seed, hl))
            g = rel_by_domain(f"{ed}/summary.txt", dom_map(f"{ed}/all_results.csv"))
            if g:
                arms.append((f"CPC k=12 · s{seed[-2:]} · h{hl}L", C_CPC if hl == 2 else C_CPC2, g,
                             agg_gm(f"{ed}/summary.txt"), "-", 2.0))
                break
        if arms:
            break
    for lab, col, base in (("β (bneck β2.98)", C_BETA, BETA), ("v11c (champion)", C_V, V11C)):
        g = rel_by_domain(f"{base}/summary.txt", dom_map(f"{base}/all_results.csv"))
        if g:
            arms.append((lab, col, g, agg_gm(f"{base}/summary.txt"), (0, (4, 2)), 1.6))
    if len(arms) < 2:
        print("radar: need CPC + a reference — skipping")
        return
    domains = sorted({d for _, _, g, *_ in arms for d in g})
    N = len(domains); theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    tc = np.concatenate([theta, theta[:1]])
    vals = [v for _, _, g, *_ in arms for v in g.values()]
    lo, hi = max(0.5, min(vals) * 0.92), max(vals) * 1.06
    fig, ax = plt.subplots(figsize=(9.5, 9.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(theta); ax.set_xticklabels(domains, fontsize=10)
    ax.set_rscale("log"); ax.set_ylim(lo, hi)
    rticks = [t for t in (0.8, 1.0, 1.5, 2.0, 2.5, 3.0) if lo < t < hi]
    ax.set_yticks(rticks); ax.set_yticklabels([f"{t:g}" for t in rticks], fontsize=8, color="0.4")
    ax.set_rlabel_position(90)
    ax.plot(tc, [1.0] * len(tc), color="k", ls=(0, (2, 2)), lw=1.0, alpha=0.6, zorder=1)
    for lab, col, g, gm, ls, lw in arms:
        v = np.array([g.get(d, np.nan) for d in domains] + [g.get(domains[0], np.nan)])
        ax.plot(tc, v, color=col, ls=ls, lw=lw, zorder=3, marker="o", markersize=3,
                label=f"{lab}   GM={gm:.3f}" if gm else lab)
    ax.set_title("#316 — full GIFT-Eval (97 cfg) per domain: CPC vs β vs v11c\n"
                 "(radial = per-domain GM rel-MASE, log; dashed ring = seasonal naive 1.0; lower = better)",
                 fontsize=11, pad=24)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/perdomain_radar.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote {OUT}/perdomain_radar.png — arms={len(arms)} domains={N}")


def fig_curves():
    curves = []
    for seed, col in ((SEED_A, C_CPC), (SEED_B, C_CPC2)):
        csvp = f"{RUNS}/bb_cpc_k12_s{seed}_fp16_50k_losses.csv"
        rows = load_csv(csvp)
        if rows:
            curves.append((f"CPC s{seed[-2:]}", col, rows))
    if not curves:
        print("curves: no losses CSV yet — skipping")
        return
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    panels = [("loss", "loss", False), ("loss_tau_ref (norm InfoNCE)", "loss_tau_ref", False),
              ("gap = ff − fp (k=1 head)", "gap", "signed"), ("R²_naive (k=1 head)", "r2_naive", "signed")]
    for ax, (title, key, mode) in zip(axs.flat, panels):
        for lab, c, rows in curves:
            xs, ys = [], []
            for r in rows:
                try:
                    s = int(r["step"]); y = float(r[key])
                except (KeyError, ValueError):
                    continue
                if mode is False and y <= 0:
                    continue
                xs.append(s); ys.append(y)
            if not xs:
                continue
            if len(xs) > 800:
                idx = np.linspace(0, len(xs) - 1, 800).astype(int)
                xs = [xs[i] for i in idx]; ys = [ys[i] for i in idx]
            ax.plot(xs, ys, color=c, lw=2.0, label=lab)
        ax.set_xscale("log")
        if mode is False:
            ax.set_yscale("log")
        else:
            ax.axhline(0.0, color="k", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(title); ax.set_xlabel("step"); ax.grid(True, alpha=0.3)
    axs[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("#316 training curves — CPC multi-step linear forecaster (k=12), fp16, τ=0.10, β2=0.98", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/training_curves.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote {OUT}/training_curves.png — seeds={len(curves)}")


def fig_gmase_vs_steps():
    beta, v11c = agg_gm(f"{BETA}/summary.txt"), agg_gm(f"{V11C}/summary.txt")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plotted = False
    for seed, col in ((SEED_A, C_CPC), (SEED_B, C_CPC2)):
        pts = []
        for sk in (10, 20, 30, 40, 50):
            tag = step_tag(seed, sk, 2) if sk != 50 else final_tag(seed, 2)
            gm = agg_gm(f"{eval_dir(tag)}/summary.txt")
            if gm is not None:
                pts.append((sk, gm))
        if pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, "o-", color=col, lw=2.0, label=f"CPC k=12 · s{seed[-2:]} (small head)")
            plotted = True
    if not plotted:
        print("gmase_vs_steps: no per-step evals yet — skipping")
        plt.close(); return
    if beta:
        ax.axhline(beta, color=C_BETA, ls="--", lw=1.6, label=f"β (final 50k) = {beta:.3f}")
    if v11c:
        ax.axhline(v11c, color=C_V, ls="--", lw=1.6, label=f"v11c = {v11c:.3f}")
    ax.set_xlabel("backbone training step (k)"); ax.set_ylabel("full-97 GM-Relative MASE (lower = better)")
    ax.set_title("#316 — does GM-MASE keep improving with training?\n"
                 "CPC backbone GIFT-Eval transfer vs training step", fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/gmase_vs_steps.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"wrote {OUT}/gmase_vs_steps.png")


if __name__ == "__main__":
    fig_gm_summary()
    fig_radar()
    fig_curves()
    fig_gmase_vs_steps()
