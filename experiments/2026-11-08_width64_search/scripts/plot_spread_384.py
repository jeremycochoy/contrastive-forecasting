"""The spread of the latent dimensions of green-384 and grey-384 over training, beside their L_align parts (the user,
11-20: "What is causing Grey to diverge? The l align to be too strong in late time? Or the l align to be too strong
against sig reg? I think at 5k sig reg should force the same spread of dimensions").

From the losses CSV of each run (one row per step): SIGReg on e and on h (the values in the loss, weight 1), the
dimension usage U of e and of h over batch x time (U = 1 / (d x mean cos^2 of two samples): 1 = isotropic, 1/d = all
samples on one line; the pooling of SIGReg), and L_align x 10 at 1 patch ahead and at the deeper depths (the rest of
the total, as in plot_loss_stack_384.py). Each curve is a mean over WINDOW steps, on a log step axis.

Usage: python3 plot_spread_384.py <output png>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import plot_loss_stack_384 as stack  # noqa: E402

WINDOW = 500
PANELS = [("sigreg_e", "SIGReg on e (lower = closer to an isotropic Gaussian)"),
          ("u_batchtime_e", "Dimension usage U of e, batch x time (1 = isotropic)"),
          ("align0", "L_align x 10, 1 patch ahead"),
          ("sigreg_h", "SIGReg on h (lower = closer to an isotropic Gaussian)"),
          ("u_batchtime", "Dimension usage U of h, batch x time (1 = isotropic)"),
          ("align_deep", "L_align x 10, the deeper depths (weighted, summed)")]
MARKS = (5, 10, 25, 50, 75)


def columns(name):
    """{key: values} of one run: the CSV columns of PANELS and the L_align parts of plot_loss_stack_384."""
    steps, parts = stack.parts(name)
    rows = [r for r in stack.csv.DictReader(open(f"{stack.HP}/{name}/ckpt/{name}_losses.csv")) if r.get("loss")]
    out = {k: np.array([float(r[k]) for r in rows]) for k in ("sigreg_e", "sigreg_h", "u_batchtime", "u_batchtime_e")}
    return steps, {**out, **parts}


def smooth(x):
    return np.convolve(x, np.ones(WINDOW) / WINDOW, mode="valid")


def main(png):
    data = {label: columns(name) for label, (name, _) in stack.RUNS.items()}
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for ax, (key, title) in zip(axes.flat, PANELS):
        for label, (steps, values) in data.items():
            ax.plot(smooth(steps), smooth(values[key]), color=stack.RUNS[label][1], lw=1.6, label=label)
        for s in MARKS:
            ax.axvline(s, color="#bbbbbb", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5, 10, 25, 50, 100])
        ax.set_xticklabels(["0.5", "1", "2", "5", "10", "25", "50", "100"])
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel("Training step (thousands, log scale; dotted: 5, 10, 25, 50, 75k)")
        ax.grid(color="#eeeeee")
    axes[0, 0].legend()
    fig.suptitle(f"green-384 and grey-384: the spread of e and h, and the L_align parts (means over {WINDOW} steps)",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(png, dpi=100)
    print("wrote", png)
    for label, (steps, values) in data.items():
        for lo, hi in ((0.5, 1), (2, 3), (4.5, 5.5), (9.5, 10.5), (24.5, 25.5), (49.5, 50.5), (74.5, 75.5), (94, 96)):
            sel = (steps >= lo) & (steps < hi)
            if sel.any():
                print(label, f"{lo}-{hi}k", {k: round(float(values[k][sel].mean()), 4) for k, _ in PANELS})


if __name__ == "__main__":
    main(sys.argv[1])
