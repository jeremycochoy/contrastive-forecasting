"""The norm of the raw forecast f (the forecaster output before the latent sphere) over training, for green and grey
at widths 64 and 384 (the user, 11-20: 'is there is nothing that prevents the norm of the forecast to explode ...
Could you check the norm of the forecast in the grey and in the green ... compare to the small grey and small green').

The script loads each save with the loader of the scores (train_forecasting_head.backbone_config,
load_frozen_backbone), on the CPU. A forward hook on transformer.fcst_up_proj reads the raw f of each forecaster
pass: depth 0, then each rollout depth (rollout_forecaster_latents with the 'normalised' feedback of the runs: each
copy goes back in at the mean encoder-latent norm). The batch: generate_random_batch(8 windows, 1,024 values, seed
20260722), the same for every save. The value: the mean L2 norm over the tokens, divided by sqrt(H), the radius of
the sphere (1 = on it). The head code comes from $HP64_ROOT/ff/code.

Usage: python3 probe_raw_forecast_norm.py <output png> <output tsv> [--redraw]
  --redraw: no probe. The script draws the png from the tsv of an earlier run.
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

HP = os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64")
FF_CODE = f"{HP}/ff/code"  # the deployed head code, with the loader of the scores
sys.path.insert(0, FF_CODE)
sys.path.insert(0, f"{FF_CODE}/experiments/2026-04-13_gift-eval/scripts")
import torch  # noqa: E402

import train_forecasting_head as tfh  # noqa: E402
from src.forecasting_head import rollout_forecaster_latents, window_latent_norm  # noqa: E402
from src.models import generate_random_batch  # noqa: E402

RUNS = {  # label -> (run name, rollout depth, colour, backbone shape: the flags of its scores)
    "green-384": ("g166b_w384e1f5_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix", 3, "#2ca02c",
                  (384, 5)),
    "grey-384": ("g166b_w384e1f5_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10harm_rollfix", 31, "#7f7f7f",
                 (384, 5)),
    "green-64": ("g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix", 3, "#98df8a", (64, 3)),
    "grey-64": ("g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10harm_rollfix", 31, "#c7c7c7", (64, 3)),
}
STOPS = (1, 10, 25, 50, 75, 100, 125, 150, 166)
PANELS = [(0, "1 patch ahead (depth 0)"), (3, "4 patches ahead (depth 3)"),
          (31, "32 patches ahead (depth 31, grey only)")]


def saves(name):
    """{stop k: path} of the numbered saves of a run (the last branch wins a stop)."""
    out = {}
    for k in STOPS:
        ckpt = f"{HP}/{name}/ckpt/{name}"
        paths = sorted(glob.glob(f"{ckpt}_{k}k.pth") + glob.glob(f"{ckpt}_r*_{k}k.pth"))
        if paths:
            out[k] = paths[-1]
    return out


def load(path, shape):
    """The frozen backbone of a save, with the shape flags of its scores (head_eval_bb.sh: --t-raw 4096
    --n-channels 1)."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    args = tfh.parse_args(["--backbone-path", path, "--t-raw", "4096", "--n-channels", "1", "--d-model", str(shape[0]),
                           "--n-heads", "8", "--num-layers", str(shape[1])])
    config, _ = tfh.backbone_config(args, sd)
    return tfh.load_frozen_backbone(args, config, sd, "cpu")


@torch.no_grad()
def raw_norms(model, x, depth):
    """[mean |raw f| / sqrt(H) of depth 0, 1, ..., depth] on the batch x."""
    seen = []
    hook = model.transformer.fcst_up_proj.register_forward_hook(lambda m, i, o: seen.append(o.float()))
    try:
        B, T_raw, C = x.shape
        xn = model.rev_norm(x, mode="norm") if model.rev_norm is not None else x
        zero = torch.zeros(B, dtype=torch.long)  # class 0 (unknown), as extract_encoder_latents and the drift probe
        ids = {k: zero for k, m in (("freq_ids", "freq_embedding"), ("seasonality_ids", "seasonality_embedding"))
               if getattr(model, m, None) is not None}
        out = model.transformer(model.prepare_encoder_input(xn, **ids))
        T, H = T_raw // model.W, model.H
        f, o = (t.reshape(B, C, T, H).permute(0, 2, 1, 3).float() for t in out[:2])
        rollout_forecaster_latents(model, f, depth, scale=window_latent_norm(o))
    finally:
        hook.remove()
    return [r.norm(dim=-1).mean().item() / H ** 0.5 for r in seen]


def main(png, tsv):
    torch.manual_seed(0)
    x = generate_random_batch(batch_size=8, T_raw=1024, C=1, seed=20260722, dimension=4)
    table = {}
    with open(tsv, "w") as out:
        out.write("run\tstop_k\tdepth\traw_norm_over_radius\n")
        for label, (name, depth, _, shape) in RUNS.items():
            for k, path in saves(name).items():
                norms = raw_norms(load(path, shape), x, depth)
                table[(label, k)] = norms
                out.writelines(f"{label}\t{k}\t{j}\t{v:.4f}\n" for j, v in enumerate(norms))
                print(label, k, "depth 0:", round(norms[0], 3), "depth 3:", round(norms[3], 3),
                      "last depth:", round(norms[-1], 3), flush=True)
    draw(table, png)


def draw_panel(ax, table, j, title):
    """The raw norm at rollout depth j of each run that reaches it, against the stop."""
    for label, (_, depth, colour, _) in RUNS.items():
        pts = sorted((k, v[j]) for (lab, k), v in table.items() if lab == label and j <= depth)
        if pts:
            ax.plot(*zip(*pts), color=colour, marker="o", lw=2, label=label)
    ax.axhline(1.0, color="grey", ls=":", lw=1.2)
    ax.set_yscale("log")  # the first save (about 10) and the late gaps (0.8 to 1.6) on one axis
    ax.set_yticks([0.8, 1, 1.5, 2, 3, 5, 10])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(f"Raw forecast norm, {title}", loc="left")
    ax.set_xlabel("Training step (thousands)")
    ax.grid(color="#eeeeee")


def draw(table, png):
    """One panel per depth of PANELS."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, (j, title) in zip(axes, PANELS):
        draw_panel(ax, table, j, title)
    axes[0].set_ylabel("mean |f before the sphere| / sqrt(H)\n(1 = the sphere radius, the dotted line; log scale)")
    axes[0].legend()
    fig.suptitle("Is the raw forecast growing? Mean norm of f before the latent sphere, on a fixed probe batch, "
                 "at each save", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(png, dpi=100)
    print("wrote", png)


def read_table(tsv):
    """The table of main() from its tsv: {(label, k): [norm of depth 0, 1, ...]}."""
    table = {}
    for row in csv.DictReader(open(tsv), delimiter="\t"):
        table.setdefault((row["run"], int(row["stop_k"])), []).append(float(row["raw_norm_over_radius"]))
    return table


if __name__ == "__main__":
    if "--redraw" in sys.argv:
        draw(read_table(sys.argv[2]), sys.argv[1])
    else:
        main(*sys.argv[1:3])
