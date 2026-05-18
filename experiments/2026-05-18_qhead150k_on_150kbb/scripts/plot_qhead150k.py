#!/usr/bin/env python3
"""Two figures for the #300 matched-head experiment:
 (1) q-head training loss vs step (the head trained 150k, not 30k);
 (2) per-domain star/radar of held-out relative MASE — distance from
     centre = GM relative MASE per GIFT-Eval domain (lower = better;
     unit circle = seasonal naive), 30k-head vs 150k-head overlaid.
"""
import csv, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

E17 = "/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-17_bottleneck_fullfh_ddp"
OUT = "/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-18_qhead150k_on_150kbb/plots"
LOSS = f"{E17}/runs/bneck_1L_fp16_bb150k_qh150k_qhead_xfmr2L_quant_150k_losses.csv"
# all_results.csv has per-config MASE + domain; SN reference per config is
# in summary.txt (Config / MASE / SN_MASE / Relative). Join on dataset.
AR_150 = f"{E17}/results/gift_eval_full_bneck_1L_fp16_bb150k_qh150k/all_results.csv"
SUM_150 = f"{E17}/results/gift_eval_full_bneck_1L_fp16_bb150k_qh150k/summary.txt"
AR_30 = f"{E17}/results/gift_eval_full_bneck_1L_fp16_150k/all_results.csv"
SUM_30 = f"{E17}/results/gift_eval_full_bneck_1L_fp16_150k/summary.txt"

os.makedirs(OUT, exist_ok=True)

# ---- (1) q-head loss ----
st, ls = [], []
with open(LOSS) as f:
    for r in csv.DictReader(f):
        try:
            s, l = int(float(r["step"])), float(r["loss"])
        except (ValueError, KeyError):
            continue
        if s > 0 and math.isfinite(l):
            st.append(s); ls.append(l)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(st, ls, lw=.8, color="#1f77b4")
ax.set_xscale("log")
ax.set_xlabel("q-head step (log)"); ax.set_ylabel("q-head training loss")
ax.set_title("Matched-head: q-head trained 150k on the 150k backbone (#300)")
ax.grid(True, which="both", ls=":", alpha=.4)
ax.axvline(30000, color="grey", ls="--", lw=.8, label="30k (the #296 head length)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/qhead150k_loss.png", dpi=130)
print(f"loss: {len(st)} pts {ls[0]:.3f}->{ls[-1]:.3f}")

# ---- (2) per-domain radar ----
def dom_map(ar_csv):
    m = {}
    if not os.path.exists(ar_csv):
        return m
    with open(ar_csv) as f:
        for r in csv.DictReader(f):
            m[r["dataset"]] = r.get("domain", "?")
    return m

def rel_by_domain(sum_txt, dmap):
    """summary.txt rows: Config  MASE  SN_MASE  Relative — GM(Relative) per domain."""
    if not os.path.exists(sum_txt):
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

g150 = rel_by_domain(SUM_150, dom_map(AR_150))
g30 = rel_by_domain(SUM_30, dom_map(AR_30))
doms = sorted(set(g150) | set(g30))
if doms:
    ang = np.linspace(0, 2 * np.pi, len(doms), endpoint=False).tolist()
    ang += ang[:1]
    def closed(g):
        v = [g.get(d, np.nan) for d in doms]; return v + v[:1]
    fig = plt.figure(figsize=(7.5, 7.5)); ax = plt.subplot(111, polar=True)
    for g, lab, c in [(g30, "30k head (#296)", "#7f7f7f"),
                      (g150, "150k head (#300)", "#d62728")]:
        ax.plot(ang, closed(g), lw=1.6, label=lab, color=c)
        ax.fill(ang, closed(g), alpha=.08, color=c)
    ax.plot(ang, [1.0] * len(ang), lw=1.0, ls="--", color="green",
            label="seasonal naive (=1.0)")
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(doms, fontsize=8)
    ax.set_title("Held-out relative MASE per GIFT-Eval domain\n"
                 "(distance from centre = GM rel-MASE; lower better; "
                 "dashed = seasonal naive)", fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/perdomain_star.png", dpi=130)
    print("domains:", {d: round(g150.get(d, float("nan")), 3) for d in doms})
else:
    print("radar SKIPPED — full-eval summary not ready yet")
