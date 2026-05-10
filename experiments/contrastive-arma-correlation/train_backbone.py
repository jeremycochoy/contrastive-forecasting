#!/usr/bin/env python3
"""
Backbone contrastive pretraining for the ARMA × correlation experiment.

Uses `ConfigurableModel.forward(x)` so the channel-mixing Kronecker module is
in the loss path (unlike the ARMA-V2 / corrV4 runs, which call
`model.transformer(x)` directly and leave the Kronecker matrices as dead
weight). Loss is `cosine_similarity_batch_no_time_neg` with C=4 — both
cross-channel and cross-batch negatives are now meaningful, and there is no
cross-time term.
"""

import argparse
import json
import os
import sys
import pathlib
import time
from types import SimpleNamespace

import torch
import torch.optim as optim

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data import generate_arma_correlated_batch  # noqa: E402

from src.loss import contrastive_latent_loss
from src.checkpoint import save_training_state, load_training_state, safe_save_path
from src.models import ConfigurableModel, compute_metrics, count_parameters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-type", type=str, default="gru",
                   choices=["mlp", "mlp_wide", "residual_silu", "gru", "conv"])
    p.add_argument("--intermediate-dim", type=int, default=None)
    p.add_argument("--H", type=int, default=1024)
    p.add_argument("--W", type=int, default=32)
    p.add_argument("--C", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--ffn-mult", type=float, default=4.0)
    p.add_argument("--activation", type=str, default="gelu", choices=["gelu", "silu"])
    p.add_argument("--depthwise-conv", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--total-steps", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=7e-5)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--loss-shape", type=str, default="cosine_similarity_batch_no_time_neg")
    p.add_argument("--T-raw", type=int, default=4096)
    p.add_argument("--dimension", type=int, default=4)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--channel-mixing-kind", type=str, default="simple",
                   choices=["simple", "attention"])
    p.add_argument("--channel-mixing-n-heads", type=int, default=8)
    p.add_argument("--save-path", type=str, default="armacorr_backbone.pth")
    p.add_argument("--experiment-id", type=str, default="default")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    model = ConfigurableModel(
        C=args.C, H=args.H, W=args.W,
        encoder_type=args.encoder_type,
        intermediate_dim=args.intermediate_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        activation=args.activation,
        depthwise_conv=args.depthwise_conv,
        channel_mixing_kind=args.channel_mixing_kind,
        channel_mixing_n_heads=args.channel_mixing_n_heads,
    )

    if args.resume:
        args.save_path = safe_save_path(args.save_path, args.resume)
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
    elif args.channel_mixing_kind == 'simple':
        # The channel-mixing module is initialised with `torch.randn(H, H)`
        # in src/blocks.py, which gives entries with σ=1 — way too large for
        # a 1024×1024 matrix. Re-init R as identity and Q as small noise so
        # the initial forward pass is approximately a no-op (h_hat ≈ h),
        # then the loss can develop a gap from a sensible starting point
        # before the Kronecker matrices learn cross-channel mixing.
        with torch.no_grad():
            H = args.H
            model.channel_mixing_module.R.copy_(torch.eye(H))
            model.channel_mixing_module.Q.copy_(torch.randn(H, H) * (0.01 / H ** 0.5))
        print("  -> reinitialised channel_mixing: R=I, Q ~ 0.01·N(0, 1/H)")
    else:
        print(f"  -> channel_mixing_kind={args.channel_mixing_kind} (init handled in module)")

    model = model.to(device)
    n_params = count_parameters(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    best_val_ff_restored = -float("inf")
    best_step_restored = 0
    if args.resume:
        restored = load_training_state(optimizer, args.resume, device=device)
        start_step = restored["step"]
        best_val_ff_restored = restored["best_val_ff"]
        best_step_restored = restored["best_step"]

    x_val, _Cv, _arv, _mav = generate_arma_correlated_batch(
        batch_size=args.batch_size, T_raw=args.T_raw, K=args.C, dimension=args.dimension,
        seed=0, device=device,
    )

    spec = SimpleNamespace(train_configuration={
        "contrastive_divergence_temperature": args.temperature,
        "contrastive_latent_noise": None,
        "loss_shape": args.loss_shape,
        "contrastive_latent_delay": 0,
    })
    cld = spec.train_configuration["contrastive_latent_delay"] + 1

    config = {
        "experiment_id": args.experiment_id,
        "encoder_type": args.encoder_type,
        "H": args.H, "W": args.W, "C": args.C,
        "num_layers": args.num_layers, "nhead": args.nhead,
        "ffn_mult": args.ffn_mult, "activation": args.activation,
        "depthwise_conv": args.depthwise_conv, "dropout": args.dropout,
        "total_steps": args.total_steps, "batch_size": args.batch_size,
        "lr": args.lr, "temperature": args.temperature,
        "loss_shape": args.loss_shape, "dimension": args.dimension,
        "n_params": n_params,
        "model_kind": "configurable_with_channel_mixing_active",
    }
    print(f"Experiment: {args.experiment_id}")
    print(f"Backbone: encoder={args.encoder_type}, H={args.H}, layers={args.num_layers}, "
          f"nhead={args.nhead}, ffn_mult={args.ffn_mult}")
    print(f"Parameters: {n_params:,}")
    print(f"Training: {args.total_steps} steps, bs={args.batch_size}, lr={args.lr}")
    print(f"  using model.forward(x) — channel-mixing module IS in the loss path")

    best_val_ff = best_val_ff_restored
    best_step = best_step_restored
    best_gap = -float("inf")
    best_gap_step = 0
    metrics_log = []
    start_time = time.time()

    for step in range(start_step + 1, args.total_steps + 1):
        model.train()
        optimizer.zero_grad()

        x_train, _C, _ar, _ma = generate_arma_correlated_batch(
            batch_size=args.batch_size, T_raw=args.T_raw, K=args.C,
            dimension=args.dimension, device=device,
        )

        # model.forward returns (h_hat, h):
        #   h_hat: post channel-mixing forecaster, [B, T, C, H]
        #   h    : pre  channel-mixing original,    [B, T, C, H]
        h_hat, h = model(x_train)
        loss = contrastive_latent_loss((h_hat, h), validation=False, spec=spec)
        loss.backward()
        optimizer.step()

        if step % args.val_every == 0 or step == args.total_steps:
            train_ff, train_fp, train_tp, train_cb = compute_metrics(
                h_hat.detach(), h.detach(), cld
            )

            model.eval()
            with torch.no_grad():
                fv, ov = model(x_val)
                val_ff, val_fp, val_tp, val_cb = compute_metrics(fv, ov, cld)

            elapsed = time.time() - start_time
            steps_done = step - start_step
            sps = steps_done / elapsed if elapsed > 0 else 0
            eta = (args.total_steps - step) / sps / 60 if sps > 0 else 0
            entry = {
                "step": step, "loss": loss.item(),
                "train_ff": train_ff, "train_fp": train_fp,
                "train_tp": train_tp, "train_cb": train_cb,
                "val_ff": val_ff, "val_fp": val_fp,
                "val_tp": val_tp, "val_cb": val_cb,
                "val_ff_fp_gap": val_ff - val_fp,
                "elapsed_sec": elapsed, "steps_per_sec": sps,
            }
            metrics_log.append(entry)
            print(
                f"[Step {step}] loss={loss.item():.4f} | "
                f"train FF={train_ff:.4f} FP={train_fp:.4f} CB={train_cb:.4f} | "
                f"val FF={val_ff:.4f} FP={val_fp:.4f} CB={val_cb:.4f} | "
                f"gap={val_ff - val_fp:.4f} | {sps:.1f} step/s | ETA {eta:.0f}min",
                flush=True,
            )

            if val_ff > best_val_ff:
                best_val_ff = val_ff
                best_step = step
                best_path = args.save_path.replace(".pth", "_best.pth")
                torch.save(model.state_dict(), best_path)
                save_training_state(
                    optimizer, best_path,
                    step=step, best_val_ff=best_val_ff, best_step=best_step,
                )
            val_gap = val_ff - val_fp
            if val_gap > best_gap:
                best_gap = val_gap
                best_gap_step = step
                bgp = args.save_path.replace(".pth", "_best_gap.pth")
                torch.save(model.state_dict(), bgp)
                save_training_state(
                    optimizer, bgp,
                    step=step, best_val_ff=best_val_ff, best_step=best_step,
                )

        if step % args.save_every == 0:
            torch.save(model.state_dict(), args.save_path)
            save_training_state(
                optimizer, args.save_path,
                step=step, best_val_ff=best_val_ff, best_step=best_step,
            )
            print(f"  -> Checkpoint saved to {args.save_path}", flush=True)

    torch.save(model.state_dict(), args.save_path)
    save_training_state(
        optimizer, args.save_path,
        step=step, best_val_ff=best_val_ff, best_step=best_step,
    )
    total_time = time.time() - start_time

    results = {
        **config,
        "best_val_ff": best_val_ff,
        "best_step": best_step,
        "best_gap": best_gap,
        "best_gap_step": best_gap_step,
        "total_time_sec": total_time,
        "final_metrics": metrics_log[-1] if metrics_log else None,
        "metrics_log": metrics_log,
    }
    results_dir = os.path.dirname(args.save_path) or "."
    results_path = os.path.join(
        results_dir, f"backbone_{args.experiment_id}_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete in {total_time/60:.1f} min")
    print(f"Best val FF: {best_val_ff:.4f} at step {best_step}")
    print(f"Best gap: {best_gap:.4f} at step {best_gap_step}")
    print(f"Results: {results_path}")
    print(f"Model: {args.save_path}")


if __name__ == "__main__":
    main()
