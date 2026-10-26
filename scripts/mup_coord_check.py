"""Coordinate check of muP (src/mup.py), as in Tensor Programs V (Appendix D.1).

A stack of the trainer's transformer layers (DecoderOnlyTransformerLayer with
QK-norm, the attention-output norm and the depthwise conv) between an input
layer and an output layer trains a few AdamW steps on random data at several
widths, once in the standard parametrization (SP) and once in muP with the
smallest width as the base. The figure shows, for each layer, the mean
absolute change of its output on a fixed batch after the last step. Under muP
the change keeps its size as the width grows. Under SP it grows with the width.

Usage: python3 scripts/mup_coord_check.py --out coord_check.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.blocks import DecoderOnlyTransformerLayer  # noqa: E402
from src.mup import mup_attention_, mup_init_, mup_param_groups, parameter_rules  # noqa: E402

IN_DIM, OUT_DIM, T, BATCH, HEADS, LAYERS = 22, 16, 32, 16, 8, 3


class Stack(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.inp = nn.Linear(IN_DIM, d)
        self.layers = nn.ModuleList(
            DecoderOnlyTransformerLayer(d, HEADS, dim_feedforward=4 * d, dropout=0.0,
                                        batch_first=True, bias=False, qk_norm=True,
                                        attn_out_norm=True)
            for _ in range(LAYERS))
        self.out = nn.Linear(d, OUT_DIM)

    def forward(self, x):
        """The output of the input layer, of each transformer layer and of the
        output layer, in order."""
        h = self.inp(x)
        outs = [h]
        for layer in self.layers:
            h = layer(h)
            outs.append(h)
        outs.append(self.out(h))
        return outs


def optimizer(model, d, base_d, mup, lr):
    if not mup:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    with torch.device("meta"):
        base = Stack(base_d)
    rules = parameter_rules(model, base)
    mup_init_(model, rules)
    mup_attention_(model, base)
    return torch.optim.AdamW(mup_param_groups(model, rules, lr, 0.0))


def changes(d, base_d, mup, lr, steps, seed):
    """Mean |output after `steps` steps - output at init| of each layer."""
    torch.manual_seed(seed)
    model = Stack(d)
    opt = optimizer(model, d, base_d, mup, lr)
    gen = torch.Generator().manual_seed(1234)
    x = torch.randn(BATCH, T, IN_DIM, generator=gen)
    y = torch.randn(BATCH, T, OUT_DIM, generator=gen)
    with torch.no_grad():
        start = model(x)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(model(x)[-1], y).backward()
        opt.step()
    with torch.no_grad():
        return [(a - b).abs().mean().item() for a, b in zip(model(x), start)]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="the PNG to write")
    p.add_argument("--widths", default="64,128,256,512")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    widths = [int(w) for w in args.widths.split(",")]
    names = ["input"] + [f"layer {i + 1}" for i in range(LAYERS)] + ["output"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, mup in zip(axes, (False, True)):
        runs = [[changes(d, widths[0], mup, args.lr, args.steps, s) for s in range(args.seeds)]
                for d in widths]
        for i, name in enumerate(names):
            mean = [sum(r[i] for r in rs) / len(rs) for rs in runs]
            ax.plot(widths, mean, marker="o", label=name)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(widths, [str(w) for w in widths])
        ax.set_xlabel("width (d_model)")
        ax.set_title("muP, base width %d" % widths[0] if mup else "SP (PyTorch default)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"mean |change of the output| after {args.steps} AdamW steps")
    axes[1].legend(fontsize=9)
    fig.suptitle(f"Coordinate check: {LAYERS} transformer layers, lr {args.lr:g}, "
                 f"mean of {args.seeds} seeds")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)


if __name__ == "__main__":
    main()
