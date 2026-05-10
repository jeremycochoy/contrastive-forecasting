#!/usr/bin/env python3
"""Held-out N=50 eval for the transformer-encoder run.

Mirrors experiments/2026-05-08_exp_tau_sweep/scripts/eval_multisample.py but
configured for the new `transformer` encoder type — same 6 metrics on the
same 50 disjoint held-out batches so the result row is directly mergeable
with the τ-sweep table.

Output:
    experiments/2026-05-10_exp_transformer_encoder/results/transformer_encoder_metrics_multisample_n50.csv

Run:
    PYTHONPATH=. python experiments/2026-05-10_exp_transformer_encoder/scripts/eval_held_out.py
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

import torch

import src.dataloader as dataloader
from src.dataloader import create_hf_dataloader
from src.forecasting_head import (
    extract_encoder_latents,
    extract_forecaster_latents,
)
from src.metrics import (
    dim_usage,
    q_naive_latent,
    q_random,
    retrieval_auc_top1_legacy as retrieval_auc_top1,
)
from src.models import ConfigurableModel


REPO_ROOT = os.environ.get(
    "CFCAST_REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
EXP_DIR = os.path.join(REPO_ROOT, "experiments", "2026-05-10_exp_transformer_encoder")
OUT_CSV = os.path.join(EXP_DIR, "results", "transformer_encoder_metrics_multisample_n50.csv")

BATCH_SIZE = 256
T_RAW = 4096
W = 16
T = T_RAW // W   # 256
HF_REPO = "jeremycochoy/gift-pretrain-full-4096"
HF_PATH = "small_v1"
TOTAL_ROWS = 42_740_000

N_SAMPLES = 50
SKIP_ROWS_LIST = [50_000_000 + i * (TOTAL_ROWS // N_SAMPLES) for i in range(N_SAMPLES)]

METRIC_KEYS = ["r2_random", "r2_naive", "u_temporal", "u_batch", "auc", "top1"]


def pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free = []
    for i in range(torch.cuda.device_count()):
        free_b, _ = torch.cuda.mem_get_info(i)
        free.append((free_b // (1024 * 1024), i))
    free.sort(reverse=True)
    free_mib, idx = free[0]
    if free_mib < 6000:
        return "cpu"
    return f"cuda:{idx}"


def detect_encoder_type(sd: dict) -> str:
    keys = sd.keys()
    if "encoder.linear_up.weight" in sd and any(
            k.startswith("encoder.layers.") for k in keys):
        return "transformer"
    if any(k.startswith("encoder.gru.") for k in keys):
        return "gru"
    if "encoder.linear1.weight" in sd:
        return "mlp"
    raise RuntimeError(f"can't detect encoder type from sd keys: {sorted(keys)[:8]}...")


def build_backbone_from_sd(sd: dict) -> ConfigurableModel:
    """Build a ConfigurableModel that matches the saved state_dict.

    Backbone hyperparams (6 layers, 6 heads, ffn_mult=4, depthwise_conv=3)
    are hardcoded — they match both the τ=0.10 GRU baseline and the
    transformer-encoder run. Encoder-type-specific knobs are detected
    from the sd shapes.
    """
    cfg = dict(C=1, H=384, W=W,
               num_layers=6, nhead=6, ffn_mult=4.0,
               activation="gelu", depthwise_conv=3, dropout=0.1,
               rev_norm_kind="ewma", rev_norm_span=128)
    cfg["encoder_type"] = detect_encoder_type(sd)

    fw = sd.get("freq_embedding.embedding.weight")
    cfg["freq_emb_dim"] = int(fw.shape[1]) if fw is not None else 0
    sw = sd.get("seasonality_embedding.embedding.weight")
    cfg["seasonality_emb_dim"] = int(sw.shape[1]) if sw is not None else 0
    cfg["learnable_tau"] = "log_inv_tau" in sd

    if cfg["encoder_type"] == "transformer":
        # Encoder (transformer) hyperparams derived from state_dict.
        enc_layer_idxs = sorted({int(k.split(".")[2])
                                 for k in sd.keys()
                                 if k.startswith("encoder.layers.")})
        cfg["enc_transformer_num_layers"] = len(enc_layer_idxs)
        if enc_layer_idxs:
            l0_l1 = sd[f"encoder.layers.{enc_layer_idxs[0]}.linear1.weight"]
            cfg["enc_transformer_ffn_mult"] = float(l0_l1.shape[0]) / float(l0_l1.shape[1])
            dw_key = f"encoder.layers.{enc_layer_idxs[0]}.depthwise_conv.conv.weight"
            cfg["enc_transformer_depthwise_conv"] = (
                int(sd[dw_key].shape[2]) if dw_key in sd else 0)
        # nhead is not directly recoverable from sd; default to 6.
        cfg["enc_transformer_nhead"] = 6
        cfg["enc_transformer_dropout"] = 0.0
        # linear_up is Linear(1 -> H); patch_stats not in its shape.
        lu = sd.get("encoder.linear_up.weight")
        if lu is None or lu.shape[1] != 1:
            raise RuntimeError(
                f"encoder.linear_up not Linear(1 -> H); got shape {tuple(lu.shape) if lu is not None else None}")
        cfg["patch_stats_kind"] = "none"
    elif cfg["encoder_type"] == "gru":
        # GRU encoder.skip.weight has in_features = W + patch_stats + freq + seas.
        ref = sd.get("encoder.skip.weight")
        if ref is None:
            raise RuntimeError("GRU encoder missing encoder.skip.weight")
        extra = ref.shape[1] - W - cfg["freq_emb_dim"] - cfg["seasonality_emb_dim"]
        if extra == 0:
            cfg["patch_stats_kind"] = "none"
        elif extra == 2:
            cfg["patch_stats_kind"] = "diff"
        else:
            raise RuntimeError(
                f"unexpected encoder.skip.in_features={ref.shape[1]} (extra={extra})")
    else:
        raise NotImplementedError(f"encoder_type={cfg['encoder_type']!r} not handled")

    bb = ConfigurableModel(**cfg)
    bb.load_state_dict(sd)
    return bb


def load_held_out_batch(skip_rows: int, device: str):
    dataloader.T_RAW = T_RAW
    torch.manual_seed(0)
    loader = create_hf_dataloader(
        repo_id=HF_REPO, batch_size=BATCH_SIZE, C=1,
        path_in_repo=HF_PATH, skip_rows=skip_rows,
    )
    batch = next(iter(loader))
    if isinstance(batch, tuple):
        if len(batch) == 3:
            x, freq_ids, seas_ids = batch
        else:
            x = batch[0]
            freq_ids = None
            seas_ids = None
    else:
        x = batch
        freq_ids = None
        seas_ids = None
    assert x.shape == (BATCH_SIZE, T_RAW, 1), x.shape
    return x.to(device), \
        (freq_ids.to(device) if freq_ids is not None else None), \
        (seas_ids.to(device) if seas_ids is not None else None)


def _forward_in_chunks(bb, x_dev, freq_ids, seas_ids, chunk: int = 32):
    e_parts, f_parts = [], []
    for s in range(0, x_dev.shape[0], chunk):
        x_c = x_dev[s:s + chunk]
        f_c = freq_ids[s:s + chunk] if freq_ids is not None else None
        s_c = seas_ids[s:s + chunk] if seas_ids is not None else None
        e_part, _ = extract_encoder_latents(
            bb, x_c, freq_ids=f_c, seasonality_ids=s_c)
        f_part, _ = extract_forecaster_latents(
            bb, x_c, freq_ids=f_c, seasonality_ids=s_c)
        e_parts.append(e_part)
        f_parts.append(f_part)
    return torch.cat(e_parts, dim=0), torch.cat(f_parts, dim=0)


def eval_one_sample(bb, H, x_dev, freq_ids, seas_ids) -> dict:
    e_bc, f_bc = _forward_in_chunks(bb, x_dev, freq_ids, seas_ids)
    B, C = BATCH_SIZE, 1
    h = e_bc.reshape(B, C, T, H).permute(0, 2, 1, 3).contiguous().float()
    f = f_bc.reshape(B, C, T, H).permute(0, 2, 1, 3).contiguous().float()
    assert h.shape == (B, T, C, H), h.shape

    torch.manual_seed(0)
    q_r = q_random(f[:, :T - 1], h[:, 1:T]).item()
    q_n = q_naive_latent(f[:, :T - 1], h[:, 1:T], h[:, :T - 1]).item()
    u_temp = dim_usage(h, axis=1).item()
    u_batch_v = dim_usage(h, axis=0).item()
    auc, top1 = retrieval_auc_top1(f[:, :T - 1], h[:, :T])
    return dict(
        r2_random=1.0 - q_r,
        r2_naive=1.0 - q_n,
        u_temporal=u_temp,
        u_batch=u_batch_v,
        auc=auc.item(),
        top1=top1.item(),
    )


def aggregate(samples: list[dict]) -> dict:
    out = {}
    n = len(samples)
    for k in METRIC_KEYS:
        vals = [s[k] for s in samples]
        m = statistics.fmean(vals)
        std = statistics.pstdev(vals) if n > 1 else 0.0
        out[f"{k}_mean"] = m
        out[f"{k}_std"] = std
    out["n_samples"] = n
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(
        REPO_ROOT, "sync_transformer_encoder", "checkpoints",
        "transformer_encoder_tau_0_10_50k_FINAL.pth"))
    p.add_argument("--name", default="transformer_encoder_tau_0_10_50k")
    p.add_argument("--tau", type=float, default=0.10)
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"FATAL: ckpt not found: {args.ckpt}", file=sys.stderr)
        sys.exit(2)

    device = pick_device()
    print(f"[eval] device={device}  ckpt={args.ckpt}", file=sys.stderr)

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    bb = build_backbone_from_sd(sd)
    bb.to(device).eval()
    print(f"[eval] params={sum(p.numel() for p in bb.parameters()):,}", file=sys.stderr)

    H = bb.H
    samples: list[dict] = []
    for i, sk in enumerate(SKIP_ROWS_LIST):
        x_dev, freq_ids, seas_ids = load_held_out_batch(sk, device)
        m = eval_one_sample(bb, H, x_dev, freq_ids, seas_ids)
        samples.append(m)
        print(f"  [eval] sample {i+1:>2}/{len(SKIP_ROWS_LIST)} skip={sk} "
              f"AUC={m['auc']:.4f} Top1={m['top1']:.4f}", file=sys.stderr)
        # cleanup between samples
        del x_dev, freq_ids, seas_ids
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    agg = aggregate(samples)
    enc_type = bb.encoder.__class__.__name__.lower().replace("encoder", "") or "unknown"
    # Friendlier name for logging.
    enc_label = ("transformer" if "transformer" in bb.encoder.__class__.__name__.lower()
                 else "gru" if "gru" in bb.encoder.__class__.__name__.lower()
                 else enc_type)

    # 1) Aggregate CSV (one row per arm; appendable across runs).
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    cols = (["name", "tau", "encoder_type"]
            + [f"{k}_mean" for k in METRIC_KEYS]
            + [f"{k}_std" for k in METRIC_KEYS]
            + ["n_samples"])
    write_header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    with open(OUT_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(cols)
        row = [args.name, args.tau, enc_label] \
              + [agg[f"{k}_mean"] for k in METRIC_KEYS] \
              + [agg[f"{k}_std"] for k in METRIC_KEYS] \
              + [agg["n_samples"]]
        w.writerow(row)
    print(f"[eval] wrote {OUT_CSV}")

    # 2) Per-sample CSV (one row per skip_rows; lets downstream plot the
    #    distribution shape, not just mean ± std).
    persample_csv = os.path.join(
        os.path.dirname(OUT_CSV),
        f"{args.name}_metrics_persample_n{len(samples)}.csv")
    with open(persample_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "tau", "encoder_type", "sample_idx", "skip_rows"] + METRIC_KEYS)
        for i, (sk, m) in enumerate(zip(SKIP_ROWS_LIST, samples)):
            w.writerow([args.name, args.tau, enc_label, i, sk]
                       + [m[k] for k in METRIC_KEYS])
    print(f"[eval] wrote {persample_csv}")

    # Also print to stdout in human form.
    print(f"\n=== {args.name} (τ={args.tau}, {enc_label} encoder) — N={agg['n_samples']} ===")
    for k in METRIC_KEYS:
        print(f"  {k:>10s} = {agg[f'{k}_mean']:.4f} ± {agg[f'{k}_std']:.4f}")


if __name__ == "__main__":
    main()
