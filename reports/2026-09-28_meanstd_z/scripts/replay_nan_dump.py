#!/usr/bin/env python3
"""Replay the step of a CF_NAN_DEBUG dump on the CPU in fp32 (#421).

The dump holds the weights before the step and the batch after all
transforms. This script rebuilds the #421z model, runs the value-space
objective of each patch-size group in eval mode (no dropout, no DropKey),
and writes:

    replay_groups.tsv  per group and rollout depth: the loss and the total
                       gradient norm, as trained and with --gru-input-bound
    replay_rows.tsv    the rows and patches with the largest input gradient
    gru_gain.tsv       per patch size and input amplitude, the largest
                       backward gain of the GRU from its input values

It trains nothing and changes no file of the run.

Usage (CPU only):
    CUDA_VISIBLE_DEVICES= python3 replay_nan_dump.py DUMP --out results/nan_15392
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

import src.forecasting_head as fh  # noqa: E402
from src.models import ConfigurableModel  # noqa: E402

SIZES = (8, 16, 32, 64, 128)
AMPLITUDES = (5, 10, 20, 30, 40, 60, 100)


def build_model(weights, bound=0.0):
    """The #421z model with the weights of the dump."""
    model = ConfigurableModel(
        C=1, H=384, W=16, encoder_type="gru", num_layers=3, nhead=8,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        freq_emb_dim=3, num_freqs=15, seasonality_emb_dim=3,
        rev_norm_kind="meanstd", rev_norm_skip_leading_zeros=True,
        num_encoder_layers=3, encoder_dropkey=0.7,
        encoder_dropkey_share_heads=True, encoder_dropkey_share_layers=True,
        qk_norm=True, attn_out_norm=True, value_head_quantiles=9,
        multi_patch_sizes=SIZES, gru_input_bound=bound,
        enc_transformer_use_grad_checkpoint=False)
    # The dump holds named_parameters: no buffer, and the encoder bank once.
    missing, unexpected = model.load_state_dict(weights, strict=False)
    assert not unexpected and all(k.startswith((
        "transformer.input_to_latent.", "rev_norm.", "channel_mixing_module.",
        "encoder.encoders.")) for k in missing), missing
    return model.eval()


def group_inputs(model, batch, size):
    """The kept rows of one patch size, with the mixed label embeddings."""
    kept, T = batch["kept"], batch["x_norm"].shape[1]
    a, partner = batch["mixup"]["weight"], batch["mixup"]["partner"]
    with torch.no_grad():
        fe = model.freq_embedding(batch["freq_ids"])
        se = model.seasonality_embedding(batch["seasonality_ids"])
        fe, se = a * fe + (1 - a) * fe[partner], a * se + (1 - a) * se[partner]
    target = torch.arange(T).view(1, -1, 1) >= batch["split"].view(-1, 1, 1)
    rows = torch.arange(len(kept))[kept]
    rows = rows[batch["patch_size"][kept] == size]
    return dict(rows=rows, x_norm=batch["x_norm"][rows],
                pad=batch["padding"][rows], target=target[rows],
                freq_embs=fe[rows], seas_embs=se[rows],
                share=len(rows) / int(kept.sum()))


def replay(model, g, size, depth):
    """The group's loss, its per-depth inputs, and the gradients."""
    inputs, real = [], fh.value_space_forward

    def keep_input(m, x, **kw):
        if x.requires_grad:
            x.retain_grad()
        inputs.append(x)
        return real(m, x, **kw)

    fh.value_space_forward = keep_input
    try:
        loss = fh.value_space_objective(
            model, g["x_norm"], depth=depth, patch_size=size,
            freq_embs=g["freq_embs"], seasonality_embs=g["seas_embs"],
            pad_mask=g["pad"], target_mask=g["target"])[0]
    finally:
        fh.value_space_forward = real
    (g["share"] * loss).backward()
    return loss.item(), inputs


def grad_norm(model):
    grads = [p.grad.double() for p in model.parameters() if p.grad is not None]
    return torch.sqrt(sum((g ** 2).sum() for g in grads)).item()


def group_table(weights, batch, bound):
    """``[(size, depth, loss, total grad norm)]`` for one bound."""
    out = []
    for size in SIZES:
        for depth in range(4):
            model = build_model(weights, bound)
            loss, _ = replay(model, group_inputs(model, batch, size), size,
                             depth)
            out.append((size, depth, loss, grad_norm(model)))
    return out


def top_rows(weights, batch, size, n=8):
    """The (row, patch) pairs with the largest input gradient at depth 2."""
    model = build_model(weights)
    g = group_inputs(model, batch, size)
    _, inputs = replay(model, g, size, 3)
    T, out = 1024 // size, []
    grad = inputs[2].grad.abs().view(len(g["rows"]), T, size).amax(dim=2)
    value = inputs[2].detach().abs().view(len(g["rows"]), T, size).amax(dim=2)
    for k in grad.flatten().argsort(descending=True)[:n].tolist():
        r, t = divmod(k, T)
        row = int(g["rows"][r])
        out.append((row, batch["row_kind"][row], t, grad[r, t].item(),
                    value[r, t].item(), int(batch["split"][row]) // size,
                    batch["scale"][row].item(), batch["max_z"][row].item()))
    return out


def gru_gain(gru, amplitude, size, n=64):
    """The largest |d(v . h_n) / d x| over random walks and alternating
    patches of the given amplitude, for a unit direction v."""
    torch.manual_seed(0)
    walk = torch.randn(n, size).cumsum(1)
    walk = amplitude * walk / walk.abs().amax(1, keepdim=True)
    alternating = amplitude * torch.rand(n, size) * (torch.arange(size) % 2)
    worst = 0.0
    for x in (walk, alternating):
        x = torch.cat([x, 0.1 * torch.randn(n, 6)], dim=1).requires_grad_(True)
        _, h = gru(x.view(n, -1, 1))
        hn = torch.cat([h[-2], h[-1]], dim=-1)
        v = torch.randn(hn.shape[-1])
        (hn @ (v / v.norm())).sum().backward()
        worst = max(worst, x.grad[:, :size].abs().max().item())
    return worst


def write_tsv(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(f"{v:.4g}" if isinstance(v, float) else str(v)
                              for v in row) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dump")
    p.add_argument("--bound", type=float, default=10.0,
                   help="The --gru-input-bound to compare with the run.")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dump = torch.load(args.dump, weights_only=False, map_location="cpu")
    weights, batch = dump["weights_before_step"], dump["batch"]
    rows = [("as trained",) + r for r in group_table(weights, batch, 0.0)]
    rows += [(f"bound {args.bound:g}",) + r
             for r in group_table(weights, batch, args.bound)]
    write_tsv(os.path.join(args.out, "replay_groups.tsv"),
              ["model", "patch_size", "depth", "loss", "grad_norm"], rows)
    write_tsv(os.path.join(args.out, "replay_rows.tsv"),
              ["row", "kind", "patch", "input_grad", "max_value",
               "split_patch", "scale", "max_z"], top_rows(weights, batch, 128))
    model = build_model(weights)
    gains = [(size, a, gru_gain(model.encoder.encoders[str(size)].gru, a,
                                size)) for size in SIZES for a in AMPLITUDES]
    write_tsv(os.path.join(args.out, "gru_gain.tsv"),
              ["patch_size", "amplitude", "max_gain"], gains)
    for name in ("replay_groups.tsv", "replay_rows.tsv", "gru_gain.tsv"):
        print(open(os.path.join(args.out, name)).read())


if __name__ == "__main__":
    main()
