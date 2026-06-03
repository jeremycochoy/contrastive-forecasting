"""GM-MASE per arm, benchmark-wide (97 configs) and on the periodic-focus subset (6 configs).
Reads results/<arm>/all_results.csv."""
import csv, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.join(HERE, "..", "results")
ARMS = [("R1v3c_ctrl","control\n(no emb)"), ("R1_freqemb_mix","freq-emb"), ("R1_freqemb_mixup_mix","freq-emb\n+ mixup")]
FOCUS = {"m4_hourly/H/short","solar/10T/long","solar/10T/medium","solar/H/short","ett1/15T/short","ett2/W/short"}
def load(a):
    return {r["dataset"]: float(r["eval_metrics/MASE[0.5]"]) for r in csv.DictReader(open(os.path.join(BASE,a,"all_results.csv")))}
def gm(xs): return math.exp(sum(map(math.log,xs))/len(xs))
data=[load(a) for a,_ in ARMS]
overall=[gm(list(d.values())) for d in data]
periodic=[gm([d[k] for k in FOCUS if k in d]) for d in data]
labels=[l for _,l in ARMS]
colors=["#555","#3b7dd8","#27ae60"]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,4.3))
for ax,vals,title,note in [(ax1,overall,"All 97 GIFT-Eval configs","embedding is a pass-through on the\nnon-periodic majority (real rows = 'unknown')"),
                           (ax2,periodic,"6 periodic-focus configs","where the frequency hint applies")]:
    b=ax.bar(labels,vals,color=colors)
    for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=11); ax.text(0.5,-0.30,note,transform=ax.transAxes,ha="center",fontsize=7.5,color="#777")
ax1.set_ylim(1.60,1.72); ax2.set_ylim(2.3,2.7)
ax1.set_ylabel("GM-MASE (lower = better)")
ax1.axhline(overall[0],color="#555",ls=":",lw=1); ax2.axhline(periodic[0],color="#555",ls=":",lw=1)
fig.suptitle("Frequency embedding: flat benchmark-wide, ~5% better on periodic series (single seed)", fontsize=11.5)
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.savefig(os.path.join(HERE,"..","plots","gm_mase_per_arm.png"),dpi=130); print("saved", overall, periodic)
