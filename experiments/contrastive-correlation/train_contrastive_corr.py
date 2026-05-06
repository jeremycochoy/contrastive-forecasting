#!/usr/bin/env python3
"""
Contrastive backbone training for correlated random walks.

Mirrors experiments/contrastive-arma/train_contrastive_v2.py but uses the
correlated-random-walk generator from src.correlation (no per-channel temporal
parameters; channels are correlated Brownian motions).
"""

import argparse
import json
import time
import torch
import torch.optim as optim
from types import SimpleNamespace

from src.loss import contrastive_latent_loss
from src.checkpoint import save_training_state, load_training_state, safe_save_path
from src.models import ConfigurableModel, compute_metrics, count_parameters
from src.correlation import generate_correlated_batch


def generate_random_batch(batch_size, T_raw=4096, K=4, seed=None, device="cpu",
                          sampler="factor", n_factors=2):
    x, _ = generate_correlated_batch(
        batch_size=batch_size, T_raw=T_raw, K=K, seed=seed, device=device,
        sampler=sampler, n_factors=n_factors,
    )
    return x


def main():
    parser = argparse.ArgumentParser(description="Contrastive backbone for correlation recovery")
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
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "silu"])
    parser.add_argument("--depthwise-conv", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Disabled by default — set >0 only if a run shows blow-up.")
    parser.add_argument("--lr-schedule", type=str, default="cosine",
                        choices=["none", "cosine"],
                        help="LR schedule: 'cosine' decays to 10% of base by total_steps")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear warmup steps; 0 disables warmup.")
    parser.add_argument("--loss-shape", type=str, default="cosine_similarity_batch_no_time_neg")
    parser.add_argument("--T-raw", type=int, default=4096)
    parser.add_argument("--sampler", type=str, default="factor",
                        choices=["factor", "uniform"],
                        help="Correlation-matrix sampler: 'factor' (default, off-diagonals "
                             "concentrated near 0.4) or 'uniform' (each pair iid U[0,1] with "
                             "PSD rejection — covers the full [0,1] range).")
    parser.add_argument("--n-factors", type=int, default=2,
                        help="Number of latent factors used by the 'factor' sampler")
    # Logging
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--save-path", type=str, default="corr_backbone.pth")
    parser.add_argument("--experiment-id", type=str, default="default")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

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
    )

    if args.resume:
        args.save_path = safe_save_path(args.save_path, args.resume)
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    model = model.to(device)
    n_params = count_parameters(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    def lr_at(step: int) -> float:
        if args.lr_schedule == "none":
            return args.lr
        if step <= args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        # cosine to 10% of base
        progress = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        import math
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    start_step = 0
    best_val_ff_restored = -float("inf")
    best_step_restored = 0
    if args.resume:
        restored = load_training_state(optimizer, args.resume, device=device)
        start_step = restored["step"]
        best_val_ff_restored = restored["best_val_ff"]
        best_step_restored = restored["best_step"]

    # Fixed validation set
    x_val = generate_random_batch(
        args.batch_size, T_raw=args.T_raw, K=args.C, seed=0, device=device,
        sampler=args.sampler, n_factors=args.n_factors,
    ).to(device)

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
        "intermediate_dim": args.intermediate_dim,
        "H": args.H, "W": args.W, "C": args.C,
        "num_layers": args.num_layers, "nhead": args.nhead,
        "ffn_mult": args.ffn_mult, "activation": args.activation,
        "depthwise_conv": args.depthwise_conv, "dropout": args.dropout,
        "total_steps": args.total_steps, "batch_size": args.batch_size,
        "lr": args.lr, "temperature": args.temperature,
        "loss_shape": args.loss_shape,
        "n_factors": args.n_factors,
        "n_params": n_params,
    }
    print(f"Experiment: {args.experiment_id}")
    print(f"Model: encoder={args.encoder_type}, H={args.H}, layers={args.num_layers}, "
          f"nhead={args.nhead}, ffn_mult={args.ffn_mult}, act={args.activation}")
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

        x_train = generate_random_batch(
            args.batch_size, T_raw=args.T_raw, K=args.C, device=device,
            sampler=args.sampler, n_factors=args.n_factors,
        ).to(device)
        B, T_raw_actual, C_actual = x_train.shape
        T = T_raw_actual // args.W
        x_reshaped = x_train.view(B, T, args.W, C_actual).permute(0, 1, 3, 2)

        f_flat, o_flat = model.transformer(x_reshaped)
        f_lat = f_flat.reshape(B, C_actual, T, args.H).permute(0, 2, 1, 3)
        o_lat = o_flat.reshape(B, C_actual, T, args.H).permute(0, 2, 1, 3)

        loss = contrastive_latent_loss((f_lat, o_lat), validation=False, spec=spec)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if args.lr_schedule != "none":
            current_lr = lr_at(step)
            for g in optimizer.param_groups:
                g["lr"] = current_lr
        optimizer.step()

        if step % args.val_every == 0 or step == args.total_steps:
            train_ff, train_fp, train_tp, train_cb = compute_metrics(
                f_lat.detach(), o_lat.detach(), cld
            )

            model.eval()
            with torch.no_grad():
                Bv, Tr, Cv = x_val.shape
                Tv = Tr // args.W
                xv = x_val.view(Bv, Tv, args.W, Cv).permute(0, 1, 3, 2)
                fv, ov = model.transformer(xv)
                fv = fv.reshape(Bv, Cv, Tv, args.H).permute(0, 2, 1, 3)
                ov = ov.reshape(Bv, Cv, Tv, args.H).permute(0, 2, 1, 3)
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
    # Write results next to the checkpoint so they don't pollute the repo root.
    import os
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
