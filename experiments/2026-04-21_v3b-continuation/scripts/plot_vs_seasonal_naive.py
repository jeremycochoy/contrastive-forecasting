"""v3b MASE / seasonal-naive MASE for all 97 GIFT-Eval configs, sorted. Reads results/R1v3b/summary.txt."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
rows=[]
for line in open(os.path.join(HERE,"..","results","R1v3b","summary.txt")):
    p=line.split()
    if len(p)>=4 and "/" in p[0]:
        try: ours,sn,ratio=float(p[-3]),float(p[-2]),float(p[-1])
        except ValueError: continue
        rows.append((p[0],ratio))
rows.sort(key=lambda x:x[1])
labels=[r[0] for r in rows]; ratios=[r[1] for r in rows]
colors=["#c0392b" if x>1.5 else "#e67e22" if x>1.0 else "#2980b9" for x in ratios]
fig,ax=plt.subplots(figsize=(7.5,12))
ax.barh(range(len(rows)),ratios,color=colors)
ax.axvline(1.0,color="k",ls="--",lw=1.2,label="seasonal-naive (1.0)")
ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels,fontsize=5)
ax.set_ylim(-1,len(rows)); ax.set_xlabel("v3b MASE ÷ seasonal-naive MASE  (1.0 = seasonal-naive; lower better)")
ax.set_title("v3b vs seasonal-naive, all 97 GIFT-Eval configs\nred = >1.5x worse (periodic / high-frequency)")
ax.legend(loc="lower right"); plt.tight_layout()
plt.savefig(os.path.join(HERE,"..","plots","vs_seasonal_naive.png"),dpi=130); print(f"v3b plot saved, {len(rows)} configs")
