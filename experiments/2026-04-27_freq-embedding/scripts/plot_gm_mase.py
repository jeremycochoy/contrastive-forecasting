"""GM-MASE over all 97 GIFT-Eval configs, per arm. Reads results/<arm>/all_results.csv."""
import csv, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.join(HERE, "..", "results")
ARMS = [("R1v3c_ctrl","control"),("R1v3c_mix","synth-mix"),("R1v3c_mix_90k","synth-mix\n90k bb"),
        ("R1_freqemb_mix","freq-emb"),("R1_freqemb_mixup_mix","freq-emb\n+ mixup"),("R1_femu_90k","fe+mu\nhead 90k")]
def gm(d):
    rows=list(csv.DictReader(open(os.path.join(BASE,d,"all_results.csv"))))
    v=[float(r["eval_metrics/MASE[0.5]"]) for r in rows if r["eval_metrics/MASE[0.5]"] not in ("","nan")]
    return math.exp(sum(map(math.log,v))/len(v))
labels=[l for _,l in ARMS]; vals=[gm(d) for d,_ in ARMS]
colors=["#555","#999","#999","#999","#27ae60","#999"]
fig,ax=plt.subplots(figsize=(8,4.4))
b=ax.bar(labels,vals,color=colors)
ax.axhline(vals[0],color="#555",ls=":",lw=1)
ax.set_ylim(1.60,1.72); ax.set_ylabel("GM-MASE, 97 configs (lower = better)")
ax.set_title("Frequency-label embedding: only the mixup arm beats control, by ~2%")
for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2, v+0.0015, f"{v:.3f}", ha="center", fontsize=9)
ax.text(0.99,0.04,"y-axis zoomed; full spread ≈2%, single seed", transform=ax.transAxes, ha="right", fontsize=8, color="#777")
plt.tight_layout(); plt.savefig(os.path.join(HERE,"..","plots","gm_mase_per_arm.png"),dpi=130); print("freq plot saved")
