"""Loss-of-record (A) `cosine_similarity_batch_full_fh_negs` — the
fₜ↔hₗ all-time crossed negative, as a horizontal time ladder.

Deliberately NOT exhaustive (cf. the square-loss diagram): one anchor
fₜ, its single positive, and its crossed-negative fan over l≠t+1. The
base same-time cross-channel / cross-batch negatives are noted, not
drawn. Siblings (B) hₜ↔hₗ and (C) fₜ↔fₗ reuse the SAME ladder with the
anchor/target row swapped — stated in the caption, defined in the annex.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 5.0))
ax.set_xlim(0.2, 5.8)
ax.set_ylim(0.05, 3.25)
ax.axis("off")

TIMES = ["t-2", "t-1", "t", "t+1", "t+2"]
X = {tt: i + 1 for i, tt in enumerate(TIMES)}
Y_F, Y_H = 2.7, 0.8                       # forecaster row (top), encoder row (bottom)
ANCHOR = "t"                              # the anchor forecast f_t
POS = "t+1"                               # its positive target h_{t+1}

C_POS = "#2a9d2a"      # green  — positive  f_t → h_{t+1}
C_NEG = "#c22020"      # red    — fₜ↔hₗ crossed negative (the (A) term)
F_FACE, H_FACE = "#d0e8ff", "#ffd0d0"
F_EDGE_A = "#d07000"   # gold edge — mark the anchor f_t

def node(x, y, face, edge="#333", lw=1.6, s=560):
    ax.scatter(x, y, s=s, zorder=5, color=face, edgecolors=edge, linewidths=lw)

# crossed negatives: f_t ↔ h_l  for every l ≠ t+1 (the (A) term)
for tt in TIMES:
    if tt == POS:
        continue
    ax.annotate("", xy=(X[tt], Y_H + 0.18), xytext=(X[ANCHOR], Y_F - 0.18),
                arrowprops=dict(arrowstyle="-", color=C_NEG, lw=1.8,
                                alpha=0.85,
                                connectionstyle="arc3,rad=0.06"), zorder=2)
# positive: f_t → h_{t+1}
ax.annotate("", xy=(X[POS], Y_H + 0.18), xytext=(X[ANCHOR], Y_F - 0.18),
            arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=3.0,
                            connectionstyle="arc3,rad=0.06"), zorder=4)

# nodes + labels
for tt in TIMES:
    is_anchor = tt == ANCHOR
    node(X[tt], Y_F, F_FACE,
         edge=F_EDGE_A if is_anchor else "#333",
         lw=3.0 if is_anchor else 1.6)
    is_pos = tt == POS
    node(X[tt], Y_H, H_FACE,
         edge=C_POS if is_pos else "#333",
         lw=3.0 if is_pos else 1.6)
    ax.text(X[tt], Y_F + 0.34, rf"$f_{{{tt}}}$", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(X[tt], Y_H - 0.34, rf"$h_{{{tt}}}$", ha="center", va="top",
            fontsize=12, fontweight="bold")

ax.text(0.35, Y_F, "forecaster", ha="left", va="center", fontsize=9,
        color="#666", style="italic")
ax.text(0.35, Y_H, "encoder", ha="left", va="center", fontsize=9,
        color="#666", style="italic")
ax.annotate("", xy=(5.55, 1.75), xytext=(0.75, 1.75),
            arrowprops=dict(arrowstyle="->", color="#d8d8d8", lw=1.1))
ax.text(5.55, 1.92, "time", ha="right", fontsize=8.5, color="#bbb",
        style="italic")

ax.set_title(
    "Loss-of-record (A) — the fₜ↔hₗ all-time crossed negative\n"
    r"anchor $f_t$: positive $h_{t+1}$; negatives $h_\ell\ \forall\,\ell\neq t{+}1$",
    fontsize=11.5, pad=6)

leg = [
    mpatches.Patch(color=C_POS, label=r"positive  $f_t \rightarrow h_{t+1}$"),
    mpatches.Patch(color=C_NEG,
                   label=r"(A) crossed negative  $f_t \leftrightarrow h_\ell$,"
                         r" $\ell\neq t{+}1$"),
]
ax.legend(handles=leg, loc="lower center", ncol=2, fontsize=9,
          framealpha=0.95, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.16))
ax.text(0.5, -0.30,
        "Base same-time cross-channel & cross-batch negatives omitted for "
        "clarity (annex).  Siblings reuse this ladder: (B) swaps the anchor "
        r"row → $h_t\!\leftrightarrow\!h_\ell$;  (C) → $f_t\!\leftrightarrow\!f_\ell$.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.2,
        color="#777")

out = ("/home/jupyter/cf-wt-crossed-loss/experiments/"
       "2026-05-19_crossed_loss_ablation/plots/loss_diagram.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved → {out}")
