"""GM-MASE on span=512 synth: was this run's 0.886 '+cosine_similarity_batch'
result a regression, or a confound? Two groupings from the same committed
eval table ../../2026-04-27__aggregate/results/synth_eval.csv, keyed on `arm`
(gm_mase + sn_gm_mase):

  CONFOUNDED (this run vs the span-sweep baseline; selector + continuity differ):
    "fe+mu @ 30k span=512"                          -> no_time_neg, best_gap,  clean         (0.848)
    "fe+mu @ 30k span=512 +cosine_similarity_batch" -> csb,         best_loss, multi-resume  (0.886)
  CLEAN A/B (follow-up 2026-04-28_exp_csb_pair_span512; matched best_loss, single-shot):
    "pair span=512 ntn (clean, best_loss)"          -> no_time_neg                            (0.924)
    "pair span=512 csb (clean, best_loss)"          -> csb                                    (0.883)

In the confounded comparison csb looks +4.4% WORSE; in the clean A/B csb is 4.5% BETTER.
Seasonal-naive (0.497) = sn_gm_mase. Single seed; 1024 held-out synth samples. Lower = better.
"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "2026-04-27__aggregate", "results", "synth_eval.csv")
rows = {r["arm"]: r for r in csv.DictReader(open(CSV))}
g = lambda arm: float(rows[arm]["gm_mase"])
sn = float(rows["fe+mu @ 30k span=512"]["sn_gm_mase"])

conf_ntn = g("fe+mu @ 30k span=512")
conf_csb = g("fe+mu @ 30k span=512 +cosine_similarity_batch")
clean_ntn = g("pair span=512 ntn (clean, best_loss)")
clean_csb = g("pair span=512 csb (clean, best_loss)")

NTN, CSB = "#5b8db8", "#c0392b"   # no_time_neg vs +cosine_similarity_batch
xs = [0, 1, 2.6, 3.6]
vals = [conf_ntn, conf_csb, clean_ntn, clean_csb]
cols = [NTN, CSB, NTN, CSB]

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.bar(xs, vals, width=0.82, color=cols, zorder=3)
for x, v in zip(xs, vals):
    ax.text(x, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
ax.axhline(sn, color="#2c3e50", ls="--", lw=1.4, zorder=2, label=f"seasonal-naive = {sn:.3f}")

conf_pct = (conf_csb - conf_ntn) / conf_ntn * 100      # +4.4  (csb looks worse)
clean_pct = (clean_csb - clean_ntn) / clean_ntn * 100  # -4.5  (csb is better)
ax.text(0.5, max(conf_ntn, conf_csb) + 0.055, f"+csb {conf_pct:+.1f}%",
        ha="center", fontsize=10, color=CSB, fontweight="bold")
ax.text(3.1, max(clean_ntn, clean_csb) + 0.055, f"+csb {clean_pct:+.1f}%",
        ha="center", fontsize=10, color="#1e7a34", fontweight="bold")

ax.set_xticks(xs)
ax.set_xticklabels(["no_time_neg\nbest_gap, clean", "+csb\nbest_loss, multiresume",
                    "no_time_neg\nbest_loss, clean", "+csb\nbest_loss, clean"], fontsize=8.5)
ax.set_ylabel("GM-MASE  (lower = better)")
ax.set_ylim(0, max(vals) * 1.24)
ax.set_title("Cross-time negatives on span=512 synth — confounded comparison vs clean A/B\n"
             "left pair: this run vs span-sweep baseline (selector + continuity differ → csb looks +4.4% worse)\n"
             "right pair: clean follow-up, matched best_loss → csb is 4.5% BETTER.  Single seed; 1024 synth samples",
             fontsize=8.6)
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
OUT = os.path.join(HERE, "..", "plots", "csb_eval.png")
plt.savefig(OUT, dpi=130)
print("wrote", os.path.relpath(OUT, HERE))
for arm in ["fe+mu @ 30k span=512", "fe+mu @ 30k span=512 +cosine_similarity_batch",
            "pair span=512 ntn (clean, best_loss)", "pair span=512 csb (clean, best_loss)"]:
    print(f"  {arm:50s} gm_mase={g(arm):.5f}")
print(f"  confounded {conf_pct:+.2f}%   clean {clean_pct:+.2f}%   sn={sn:.5f}")
