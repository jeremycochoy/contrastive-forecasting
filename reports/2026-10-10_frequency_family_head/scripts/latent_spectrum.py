"""Effective rank of the backbone latents on real GIFT-Eval windows.

Usage: latent_spectrum.py OUT_TSV CODE_DIR NUM_LAYERS LABEL=CKPT [LABEL=CKPT ...]

For each checkpoint: the encoder output h and the forecaster output f (the
latent the forecast head reads) at the last 32 patches of the 62 windows of
attention_sharpness.batch(). Per latent: the effective rank exp(entropy of
the normalised eigenvalues of the covariance), the count of directions for
90% of the variance, and the share of the first direction. Rows are appended
to OUT_TSV. One code folder per call: a muP backbone needs a muP blocks.py.
ENCODER_SOURCE=teacher loads the EMA teacher's patch embedding and encoder in
place of the student's: its h is the target of L_align (its f mixes the two).
"""
import importlib.util
import math
import os
import sys

import numpy as np
import torch

LAST = 32
SOURCE = os.environ.get("ENCODER_SOURCE", "student")  # teacher: the EMA encoder gives h
HERE = os.path.dirname(os.path.abspath(__file__))


def load_batch_module(code_dir):
    """attention_sharpness.py with its CODE folder replaced."""
    source = open(f"{HERE}/attention_sharpness.py").read()
    source = source.replace('CODE = "/home/jupyter/cf_runs/qkgain_test/code"', f'CODE = "{code_dir}"')
    spec = importlib.util.spec_from_loader("att", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, "attention_sharpness.py", "exec"), module.__dict__)
    return module


def spectrum_stats(z):
    """(effective rank, directions for 90% variance, share of the first,
    share of the mean in the second moment ||mean||^2 / mean ||z||^2)."""
    mean = z.mean(0, keepdim=True)
    mean_share = (mean.pow(2).sum() / z.pow(2).sum(1).mean()).item()
    z = z - mean
    eig = torch.linalg.eigvalsh(z.T @ z / (len(z) - 1)).clamp_min(0).flip(0).double()
    p = eig / eig.sum()
    entropy = -(p[p > 0] * p[p > 0].log()).sum().item()
    n90 = int((p.cumsum(0) < 0.9).sum().item()) + 1
    return math.exp(entropy), n90, p[0].item(), mean_share


def latents(att, path, num_layers, x, fids, sids):
    job = [a if a != "6" else str(num_layers) for a in att.JOB_ARGS]
    job = [SOURCE if a == "student" else a for a in job]
    args = att.tfh.parse_args(["--backbone-path", path] + job)
    sd = torch.load(path, map_location="cuda", weights_only=True)
    config, _ = att.tfh.backbone_config(args, sd)
    backbone = att.tfh.load_frozen_backbone(args, config, sd, "cuda")
    with torch.no_grad():
        x_norm = backbone.rev_norm(x, mode="norm")
        xr = backbone.prepare_encoder_input(x_norm, freq_ids=fids, seasonality_ids=sids)
        f, h = backbone.transformer(xr)
    return {"h": h[:, -LAST:].float().reshape(-1, h.shape[-1]),
            "f": f[:, -LAST:].float().reshape(-1, f.shape[-1])}


def main():
    out, code_dir, num_layers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    att = load_batch_module(code_dir)
    x, fids, sids = att.batch()
    x, fids, sids = x.cuda(), fids.cuda(), sids.cuda()
    new = not os.path.exists(out)
    with open(out, "a") as fh:
        if new:
            fh.write("label\tlatent\teff_rank\tn90\ttop1\tmean_share\n")
        for item in sys.argv[4:]:
            label, path = item.split("=", 1)
            for name, z in latents(att, path, num_layers, x, fids, sids).items():
                name = name if SOURCE == "student" else f"{name}_{SOURCE}"
                er, n90, top1, ms = spectrum_stats(z)
                fh.write(f"{label}\t{name}\t{er:.2f}\t{n90}\t{top1:.4f}\t{ms:.4f}\n")
                print(f"{label:22s} {name:10s} effective rank {er:6.1f}, n90 {n90:3d}, "
                      f"top1 {top1:.3f}, mean share {ms:.3f}", flush=True)


if __name__ == "__main__":
    main()
