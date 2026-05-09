"""Generate the (batch × time) square diagram for the loss extensions design doc."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis("off")

# ── vertex positions (all in [0,10]×[0,9] space) ──────────────────────────────
X_F, X_H = 2.5, 7.5        # f_t column, h_{t+1} column
Y_B, Y_BP = 6.5, 2.5       # row b (top), row b' (bottom)
POS = dict(f_b=(X_F, Y_B), h_b=(X_H, Y_B), f_bp=(X_F, Y_BP), h_bp=(X_H, Y_BP))

def edge(src, dst, color, lw, arrowhead=False, ls="-"):
    x0, y0 = POS[src]; x1, y1 = POS[dst]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->" if arrowhead else "-",
                                color=color, lw=lw, linestyle=ls,
                                connectionstyle="arc3,rad=0.0"),
                zorder=3)

# ── edges ──────────────────────────────────────────────────────────────────────
edge("f_b",  "h_b",  "#2a9d2a", 2.5, arrowhead=True)           # positive top
edge("f_bp", "h_bp", "#2a9d2a", 2.5, arrowhead=True)           # positive bottom
edge("f_b",  "f_bp", "#1a6ec2", 2.2)                           # left NEW
edge("h_b",  "h_bp", "#c22020", 2.2)                           # right NEW
edge("f_b",  "h_bp", "#d07000", 1.8, arrowhead=True, ls=(0,(6,3)))  # diagonal kept

# ── vertices ───────────────────────────────────────────────────────────────────
VLABELS = dict(f_b=r"$f_{b,t}$", h_b=r"$h_{b,t+1}$",
               f_bp=r"$f_{b',t}$", h_bp=r"$h_{b',t+1}$")
for k, (x, y) in POS.items():
    ax.scatter(x, y, s=260, zorder=5, color="white", edgecolors="#222", linewidths=1.8)
    dy, va = (0.4, "bottom") if y == Y_B else (-0.4, "top")
    ax.text(x, y+dy, VLABELS[k], ha="center", va=va, fontsize=13, fontweight="bold")

# ── inline edge labels ─────────────────────────────────────────────────────────
MX, MY = (X_F+X_H)/2, (Y_B+Y_BP)/2
# positive top
ax.text(MX, Y_B+0.9, r"positive: $f_{b,t}\sim h_{b,t+1}$",
        ha="center", fontsize=9.5, color="#2a9d2a",
        bbox=dict(fc="white", ec="none", pad=1))
# positive bottom
ax.text(MX, Y_BP-0.9, r"positive: $f_{b',t}\sim h_{b',t+1}$",
        ha="center", fontsize=9.5, color="#2a9d2a",
        bbox=dict(fc="white", ec="none", pad=1))
# left edge
ax.text(X_F-1.55, MY,
        "neg_cross_batch\n_forecast  (NEW)\n"+r"$f_{b,t}\leftrightarrow f_{b',t}$",
        ha="center", va="center", fontsize=9, color="#1a6ec2",
        bbox=dict(fc="white", ec="none", pad=1))
# right edge
ax.text(X_H+1.55, MY,
        "neg_cross_batch\n_embedding  (NEW)\n"+r"$h_{b,t+1}\leftrightarrow h_{b',t+1}$",
        ha="center", va="center", fontsize=9, color="#c22020",
        bbox=dict(fc="white", ec="none", pad=1))
# diagonal
ax.text(MX+0.25, MY-0.15,
        "neg_cross_batch\n_forecast_embedding\n"+r"$f_{b,t}\leftrightarrow h_{b',t+1}$",
        ha="center", va="center", fontsize=8.8, color="#d07000",
        bbox=dict(fc="white", ec="#e8c880", boxstyle="round,pad=0.3", alpha=0.9))

# ── axis / column labels ───────────────────────────────────────────────────────
ax.text(X_F, Y_B+1.55, r"$f_t$ column", ha="center", fontsize=10, color="#555", style="italic")
ax.text(X_H, Y_B+1.55, r"$h_{t+1}$ column", ha="center", fontsize=10, color="#555", style="italic")
# batch-axis arrow + labels
ax.annotate("", xy=(0.5, Y_BP+0.15), xytext=(0.5, Y_B-0.15),
            arrowprops=dict(arrowstyle="<->", color="#aaa", lw=1.3))
ax.text(0.5, Y_B,  "b",  ha="center", va="center", fontsize=13, fontweight="bold", color="#333")
ax.text(0.5, Y_BP, "b'", ha="center", va="center", fontsize=13, fontweight="bold", color="#333")
ax.text(0.05, MY, "batch\naxis", ha="center", va="center", fontsize=8, color="#aaa",
        style="italic", rotation=90)

# ── footnote ───────────────────────────────────────────────────────────────────
ax.text(5, 0.55,
        r"$neg\_zy$: $f_{b,t}\leftrightarrow f_{b,t+1}$ (temporal, same batch) — already present",
        ha="center", va="center", fontsize=8.8, color="#555")

# ── title ──────────────────────────────────────────────────────────────────────
ax.set_title("Square loss: (batch × time) negative structure\n"
             r"$\mathtt{cosine\_similarity\_batch\_square}$",
             fontsize=12, pad=10)

# ── legend (inside axes, top-centre) ──────────────────────────────────────────
items = [
    mpatches.Patch(color="#2a9d2a", label="positive pair (existing)"),
    mpatches.Patch(color="#d07000", label="neg_cross_batch_forecast_embedding — diagonal (kept)"),
    mpatches.Patch(color="#1a6ec2", label="neg_cross_batch_forecast — left edge (NEW)"),
    mpatches.Patch(color="#c22020", label="neg_cross_batch_embedding — right edge (NEW)"),
]
ax.legend(handles=items, loc="upper center", fontsize=8.5,
          framealpha=0.95, edgecolor="#ccc", ncol=2,
          bbox_to_anchor=(0.5, 0.14))   # fraction of axes: centre, near bottom

out = "experiments/2026-05-09_exp_loss_extensions/plots/square_diagram.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved → {out}")
