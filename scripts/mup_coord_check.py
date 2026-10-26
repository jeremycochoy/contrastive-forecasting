"""Coordinate check of muP (src/mup.py), as in Tensor Programs V (Appendix D.1).

A stack of the trainer's transformer layers (DecoderOnlyTransformerLayer with
QK-norm, the attention-output norm and the depthwise conv) between an input
layer and an output layer trains a few AdamW steps on random data, once in the
standard parametrization (SP) and once in muP. The figure shows the mean
absolute change of the output of each layer on a fixed batch after the last
step.

--vary width (default): 3 layers at several widths, with the smallest width as
the muP base. Under muP the change keeps its size as the width grows. Under SP
it grows with the width.

--vary depth: one width (default 128) at several depths, with the muP base
width 64. muP sets the rules of the width only: the figure shows what the
depth does in each parametrization.

Usage: python3 scripts/mup_coord_check.py --out coord_check.png [--vary depth]
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

IN_DIM, OUT_DIM, T, BATCH, HEADS = 22, 16, 32, 16, 8


class Stack(nn.Module):
    def __init__(self, d, n_layers):
        super().__init__()
        self.inp = nn.Linear(IN_DIM, d)
        self.layers = nn.ModuleList(
            DecoderOnlyTransformerLayer(d, HEADS, dim_feedforward=4 * d, dropout=0.0,
                                        batch_first=True, bias=False, qk_norm=True,
                                        attn_out_norm=True)
            for _ in range(n_layers))
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


def optimizer(model, n_layers, base_d, lr):
    """AdamW of SP (``base_d`` None), or of muP against the base width."""
    if base_d is None:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    with torch.device("meta"):
        base, delta = Stack(base_d, n_layers), Stack(2 * base_d, n_layers)
    rules = parameter_rules(model, base, delta)
    mup_init_(model, rules)
    mup_attention_(model, base)
    return torch.optim.AdamW(mup_param_groups(model, rules, lr, 0.0))


def changes(d, n_layers, base_d, lr, steps, seed):
    """Mean |output after `steps` steps - output at init| of each layer."""
    torch.manual_seed(seed)
    model = Stack(d, n_layers)
    opt = optimizer(model, n_layers, base_d, lr)
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


def mean_changes(d, n_layers, base_d, args):
    """``changes`` averaged over the seeds."""
    runs = [changes(d, n_layers, base_d, args.lr, args.steps, s) for s in range(args.seeds)]
    return [sum(r[i] for r in runs) / len(runs) for i in range(len(runs[0]))]


PICKS = {"input": lambda n: 0, "layer 1": lambda n: 1,
         "middle layer": lambda n: (n + 1) // 2, "last layer": lambda n: n,
         "output": lambda n: n + 1}


def width_panel(ax, args, mup):
    """Each layer up to 3 layers. Above 3 layers, the layers of PICKS."""
    widths = [int(w) for w in args.widths.split(",")]
    base = args.base_width or widths[0]
    rows = [mean_changes(d, args.layers, base if mup else None, args) for d in widths]
    n = args.layers
    names = (dict(PICKS) if n > 3 else
             {"input": lambda n: 0, **{f"layer {i}": (lambda n, i=i: i) for i in range(1, n + 1)},
              "output": lambda n: n + 1})
    for name, index in names.items():
        ax.plot(widths, [r[index(n)] for r in rows], marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths, [str(w) for w in widths])
    ax.set_xlabel("width (d_model)")
    return f"muP, base width {base}"


def depth_panel(ax, args, mup):
    depths = [int(n) for n in args.depths.split(",")]
    base = args.base_width or 64
    rows = [mean_changes(args.width, n, base if mup else None, args) for n in depths]
    for name, index in PICKS.items():
        ax.plot(depths, [r[index(n)] for r, n in zip(rows, depths)], marker="o", label=name)
    ax.set_xticks(depths, [str(n) for n in depths])
    ax.set_xlabel(f"number of transformer layers (width {args.width})")
    return f"muP, base width {base}"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="the PNG to write")
    p.add_argument("--vary", choices=("width", "depth"), default="width")
    p.add_argument("--widths", default="64,128,256,512", help="--vary width")
    p.add_argument("--layers", type=int, default=3, help="--vary width: the depth")
    p.add_argument("--depths", default="3,4,5,6,7,8,9", help="--vary depth")
    p.add_argument("--width", type=int, default=128, help="--vary depth: the width")
    p.add_argument("--base-width", type=int, default=None,
                   help="the muP base. Default: the smallest width, or 64 with --vary depth")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    panel = width_panel if args.vary == "width" else depth_panel
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, mup in zip(axes, (False, True)):
        mup_title = panel(ax, args, mup)
        ax.set_yscale("log")
        ax.set_title(mup_title if mup else "SP (PyTorch default)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"mean |change of the output| after {args.steps} AdamW steps")
    axes[1].legend(fontsize=9)
    shape = (f"{args.layers} transformer layers" if args.vary == "width"
             else f"width {args.width}, {args.depths.replace(',', ', ')} layers")
    fig.suptitle(f"Coordinate check: {shape}, lr {args.lr:g}, mean of {args.seeds} seeds")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)


if __name__ == "__main__":
    main()
