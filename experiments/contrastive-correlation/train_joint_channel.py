#!/usr/bin/env python3
"""
Backbone training for the joint-channel correlation experiment.

Same loss as `train_contrastive_corr.py` (`cosine_similarity_batch_no_time_neg`)
but the model is `JointChannelModel`, which gives the transformer a single
sequence per sample with C×H concatenated in the feature dimension. This
keeps the cross-batch contrastive objective intact (and silently disables
the cross-channel-negatives term, since C=1 at the loss level).

Default LR schedule is *constant* — the joint-channel design doesn't show
the late-training loss spikes that motivated cosine decay for the
per-channel design.
"""

import argparse
import json
import os
import time
import torch
import torch.optim as optim
from types import SimpleNamespace

from src.loss import contrastive_latent_loss
from src.checkpoint import save_training_state, load_training_state, safe_save_path
from src.models import compute_metrics, count_parameters
from src.correlation import generate_correlated_batch


def _import_joint_model():
    # The script lives in experiments/contrastive-correlation/, but Python
    # can't import a hyphenated package. Add the directory to sys.path.
    import sys, pathlib
    here = pathlib.Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from joint_channel_model import JointChannelModel  # noqa: E402
    return JointChannelModel


def main():
    parser = argparse.ArgumentParser()
    # Architecture
    parser.add_argument("--encoder-type", type=str, default="gru",
                        choices=["mlp", "mlp_wide", "residual_silu", "gru", "conv"])
    parser.add_argument("--intermediate-dim", type=int, default=None)
    parser.add_argument("--H", type=int, default=1024)
    parser.add_argument("--W", type=int, default=32)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ffn-mult", type=float, default=4.0)
    parser.add_argument("--depthwise-conv", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--loss-shape", type=str, default="cosine_similarity_batch_no_time_neg")
    parser.add_argument("--T-raw", type=int, default=4096)
    parser.add_argument("--sampler", type=str, default="uniform",
                        choices=["factor", "uniform"])
    parser.add_argument("--n-factors", type=int, default=2)
    # Logging
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--save-path", type=str, default="corr_jc.pth")
    parser.add_argument("--experiment-id", type=str, default="default")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    JointChannelModel = _import_joint_model()
    model = JointChannelModel(
        C=args.C, H=args.H, W=args.W,
        encoder_type=args.encoder_type,
        intermediate_dim=args.intermediate_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        depthwise_conv=args.depthwise_conv,
    )

    if args.resume:
        args.save_path = safe_save_path(args.save_path, args.resume)
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

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

    # Fixed validation set
    x_val, _ = generate_correlated_batch(
        batch_size=args.batch_size, T_raw=args.T_raw, K=args.C, seed=0,
        device=device, sampler=args.sampler, n_factors=args.n_factors,
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
        "ffn_mult": args.ffn_mult,
        "depthwise_conv": args.depthwise_conv, "dropout": args.dropout,
        "total_steps": args.total_steps, "batch_size": args.batch_size,
        "lr": args.lr, "temperature": args.temperature,
        "loss_shape": args.loss_shape, "sampler": args.sampler,
        "n_params": n_params,
        "model_kind": "joint_channel",
    }
    print(f"Experiment: {args.experiment_id}")
    print(f"Joint-channel backbone: H={args.H}, layers={args.num_layers}, "
          f"nhead={args.nhead}, ffn_mult={args.ffn_mult}, sampler={args.sampler}")
    print(f"Parameters: {n_params:,}")
    print(f"Training: {args.total_steps} steps, bs={args.batch_size}, lr={args.lr}")

    best_val_ff = best_val_ff_restored
    best_step = best_step_restored
    best_gap = -float("inf")
    best_gap_step = 0
    metrics_log = []
    start_time = time.time()

    for step in range(start_step + 1, args.total_steps + 1):
        model.train()
        optimizer.zero_grad()

        x_train, _ = generate_correlated_batch(
            batch_size=args.batch_size, T_raw=args.T_raw, K=args.C,
            device=device, sampler=args.sampler, n_factors=args.n_factors,
        )

        f_lat, o_lat = model(x_train)  # both [B, T, 1, H]

        loss = contrastive_latent_loss((f_lat, o_lat), validation=False, spec=spec)
        loss.backward()
        optimizer.step()

        if step % args.val_every == 0 or step == args.total_steps:
            train_ff, train_fp, train_tp, train_cb = compute_metrics(
                f_lat.detach(), o_lat.detach(), cld
            )

            model.eval()
            with torch.no_grad():
                fv, ov = model(x_val)
                val_ff, val_fp, val_tp, val_cb = compute_metrics(fv, ov, cld)

            elapsed = time.time() - start_time
            steps_done = step - start_step
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
            eta_min = (args.total_steps - step) / steps_per_sec / 60 if steps_per_sec > 0 else 0

            entry = {
                "step": step, "loss": loss.item(),
                "train_ff": train_ff, "train_fp": train_fp,
                "train_tp": train_tp, "train_cb": train_cb,
                "val_ff": val_ff, "val_fp": val_fp,
                "val_tp": val_tp, "val_cb": val_cb,
                "val_ff_fp_gap": val_ff - val_fp,
                "elapsed_sec": elapsed, "steps_per_sec": steps_per_sec,
            }
            metrics_log.append(entry)

            print(
                f"[Step {step}] loss={loss.item():.4f} | "
                f"train FF={train_ff:.4f} FP={train_fp:.4f} CB={train_cb:.4f} | "
                f"val FF={val_ff:.4f} FP={val_fp:.4f} CB={val_cb:.4f} | "
                f"gap={val_ff - val_fp:.4f} | {steps_per_sec:.1f} step/s | ETA {eta_min:.0f}min",
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
                best_gap_path = args.save_path.replace(".pth", "_best_gap.pth")
                torch.save(model.state_dict(), best_gap_path)
                save_training_state(
                    optimizer, best_gap_path,
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
        results_dir, f"corr_backbone_{args.experiment_id}_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete in {total_time/60:.1f} min")
    print(f"Best val FF: {best_val_ff:.4f} at step {best_step}")
    print(f"Best gap: {best_gap:.4f} at step {best_gap_step}")
    print(f"Results saved to {results_path}")
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
