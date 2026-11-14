"""Where the gradient of step 13,755 grows along the 32-call rollout chain (Nov 14).

Reads the records of rerun_gain.py: for each forecaster call j and each row, the gradient at f before the sphere
(g_raw) and at the input tokens of the first forecaster layer (g_in). Call j + 1 reads the output of call j, so the
backward pass goes g_raw(j + 1) -> g_in(j + 1) -> g_raw(j). Two gains per call, on the L2 norm of the gradient:
  layer gain  G(j) = |g_in(j)| / |g_raw(j)|        the forecaster layers of call j
  link gain   L(j) = |g_raw(j)| / |g_in(j + 1)|    the sphere of call j, the feedback, and the loss terms of depth j
Usage: python3 plot_call_gain.py <variant records.pt> <red records.pt> <out.png>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

BURST_ROW = 17  # the row with most of the variant's burst (64 % at call 0, rerun_cpu_train_s0_dump)
LOSS_DEPTHS = (0, 1, 3, 7, 15, 31)  # the depths j = d - 1 of the distances 1, 2, 4, 8, 16, 32


def row_norms(per_token):
    """[rows]: the L2 norm of each row's gradient, from the norms of its tokens."""
    return per_token.pow(2).sum(1).sqrt()


def gradients(path):
    """[calls, rows] twice (|g_raw|, |g_in|), and the parameter gradient norm of the step."""
    rec = torch.load(path, map_location="cpu", weights_only=False)
    raw = torch.stack([row_norms(c["grad_norm"]) for c in rec["calls"]])
    inp = torch.stack([row_norms(c["grad_in"]) for c in rec["inputs"]])
    return raw, inp, rec["grad_total"]


def batch(t):
    """[calls]: the L2 norm over the rows."""
    return t.pow(2).sum(1).sqrt()


def report(name, raw, inp, total):
    layer, link = batch(inp) / batch(raw), batch(raw[:-1]) / batch(inp[1:])
    row_layer, row_link = inp[:, BURST_ROW] / raw[:, BURST_ROW], raw[:-1, BURST_ROW] / inp[1:, BURST_ROW]
    print(f"{name}: parameter gradient norm {total:.6g}")
    print(f"  call  |g_raw|     G (batch)  L (batch)   G (row {BURST_ROW})  L (row {BURST_ROW})")
    for j in range(raw.shape[0]):
        tail = (f"{float(link[j]):9.3f}   {float(row_layer[j]):9.3f}  {float(row_link[j]):9.3f}"
                if j < len(link) else f"{'-':>9}   {float(row_layer[j]):9.3f}  {'-':>9}")
        print(f"  {j:4d}  {float(batch(raw)[j]):10.4g}  {float(layer[j]):9.3f}  {tail}")
    return layer, link


def main(variant_path, red_path, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for name, path, colour in (("red without SIGReg-F, step 13,755", variant_path, "#e377c2"),
                               ("red at 35k, the same batch", red_path, "#d62728")):
        raw, inp, total = gradients(path)
        layer, link = report(name, raw, inp, total)
        axes[0].semilogy(batch(raw), color=colour, marker="o", ms=3,
                         label=f"{name}\n(parameter gradient norm {total:.3g})")
        axes[1].plot(layer, color=colour, marker="o", ms=3, label=name)
        axes[2].plot(link, color=colour, marker="o", ms=3, label=name)
    for ax in axes:
        for j in LOSS_DEPTHS:
            ax.axvline(j, color="#bbbbbb", lw=0.6, zorder=0)
        ax.set_xlabel("forecaster call j (0: the forward pass; 1 to 31: the rollout)\n"
                      "grey lines: the calls with an L_align term (distances 1, 2, 4, 8, 16, 32)")
        ax.legend(fontsize=8)
    for ax in axes[1:]:
        ax.axhline(1.0, color="#7f7f7f", lw=0.8, ls=":")
        ax.set_ylabel("gain")
    axes[0].set_ylabel("gradient norm")
    axes[0].set_title("gradient at f before the sphere, per call (L2 norm, whole batch)")
    axes[1].set_title("layer gain G(j) = |g_in(j)| / |g_raw(j)|\n(the forecaster layers of call j)")
    axes[2].set_title("link gain L(j) = |g_raw(j)| / |g_in(j + 1)|\n(the sphere of call j and the feedback)")
    fig.suptitle("Step 13,755 of the run without SIGReg-F, again on the CPU in fp32: "
                 "where the backward pass grows along the 32-call chain", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
