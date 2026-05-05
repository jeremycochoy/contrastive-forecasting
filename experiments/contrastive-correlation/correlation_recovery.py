#!/usr/bin/env python3
"""
Correlation recovery: train a small head on the frozen contrastive backbone
to predict the 6 unique pairwise correlations of the 4-channel correlation
matrix used to generate the data.

Two head variants:
  - linear: pool over time → flatten over channels → single Linear → sigmoid
  - mlp:    pool over time → flatten → MLP (2 hidden layers) → sigmoid

Targets are the 6 upper-triangular entries of C, in [0, 1) by construction.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.correlation import generate_correlated_batch, correlation_to_pairs
from src.models import ConfigurableModel


# -----------------------------------------------------------------------------
# Recovery heads
# -----------------------------------------------------------------------------

class LinearCorrelationHead(nn.Module):
    """Pool over time, flatten channels, single linear → 6 correlations."""

    def __init__(self, H: int, K: int = 4):
        super().__init__()
        self.K = K
        self.num_pairs = K * (K - 1) // 2
        self.linear = nn.Linear(H * K, self.num_pairs)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, T, K, H] -> [B, num_pairs] in (0, 1)."""
        h_pool = h.mean(dim=1)
        flat = h_pool.flatten(1)
        return torch.sigmoid(self.linear(flat))


class MLPCorrelationHead(nn.Module):
    """MLP head: pool over time, flatten, 2 hidden layers, sigmoid."""

    def __init__(self, H: int, K: int = 4, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.num_pairs = K * (K - 1) // 2
        self.net = nn.Sequential(
            nn.Linear(H * K, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_pairs),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_pool = h.mean(dim=1)
        flat = h_pool.flatten(1)
        return torch.sigmoid(self.net(flat))


class TimeAwareHead(nn.Module):
    """Per-timestep cross-channel MLP with LayerNorm, then mean-pool over time.

    Structurally matches the correlation operator:
        corr_ij = (1/T) Σ_t f(x_{t,1}, ..., x_{t,K})

    Pure mean-pool-then-MLP cannot compute pairwise correlations because
    correlation is a quadratic statistic. By applying a non-linear MLP per
    timestep first, the head can produce per-step "instantaneous correlation
    estimates" which are then averaged over time.

    No sigmoid: raw output, MSE loss. The final layer bias is initialised to
    the empirical mean of the training targets so the model starts near
    sensible values rather than near a saturated extreme.
    """

    def __init__(self, H: int, K: int = 4, hidden_dim: int = 256, dropout: float = 0.0,
                 init_bias: float = 0.4):
        super().__init__()
        self.K = K
        self.num_pairs = K * (K - 1) // 2
        self.norm = nn.LayerNorm(K * H)
        self.fc1 = nn.Linear(K * H, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, self.num_pairs)
        self.dropout = dropout
        # Initialise output bias near the empirical mean of targets so we start
        # in a sensible regime instead of zero.
        with torch.no_grad():
            self.out.bias.fill_(float(init_bias))
            self.out.weight.mul_(0.1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        flat = h.flatten(2)               # [B, T, K*H]
        flat = self.norm(flat)
        x = F.gelu(self.fc1(flat))
        if self.dropout > 0 and self.training:
            x = F.dropout(x, p=self.dropout)
        x = F.gelu(self.fc2(x))
        if self.dropout > 0 and self.training:
            x = F.dropout(x, p=self.dropout)
        per_t = self.out(x)               # [B, T, num_pairs]
        return per_t.mean(dim=1).clamp(0.0, 1.0)


HEADS = {
    "linear": LinearCorrelationHead,
    "mlp": MLPCorrelationHead,
    "timeaware": TimeAwareHead,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def extract_latents(model, x, which: str = "h"):
    """Return [B, T, K, H] backbone latent. `which` selects which tap point:

    - "h":     model's `x_original` — the per-channel encoder/transformer latent
               *before* the channel-mixing module. Used by the ARMA recovery code.
    - "h_hat": model's first return — *after* the channel-mixing module.
               Each channel here already contains a Kronecker-mixed combination
               of all channels, which can help cross-channel tasks.
    """
    with torch.no_grad():
        h_hat, h = model(x)  # both [B, T, K, H]
    return h_hat if which == "h_hat" else h


def make_batch(batch_size, K, T_raw, device, seed=None, n_factors=2):
    x, C = generate_correlated_batch(
        batch_size=batch_size, T_raw=T_raw, K=K, seed=seed,
        device=device, n_factors=n_factors,
    )
    targets = correlation_to_pairs(C, K=K)
    return x, C, targets


def empirical_corr_baseline(x: torch.Tensor) -> torch.Tensor:
    """Compute the empirical Pearson correlation matrix of the *increments* of x.
    Returns the 6 unique pairs per batch sample.
    Shape: [B, num_pairs]."""
    K = x.shape[-1]
    diffs = x[:, 1:] - x[:, :-1]  # [B, T-1, K]
    # Centre & normalise per (B, K)
    diffs = diffs - diffs.mean(dim=1, keepdim=True)
    std = diffs.std(dim=1, keepdim=True).clamp(min=1e-6)
    z = diffs / std
    # Empirical correlation: (1/(T-1)) z^T z
    Tm1 = z.shape[1]
    corr = torch.einsum("bti,btj->bij", z, z) / Tm1  # [B, K, K]
    return correlation_to_pairs(corr, K=K)


def empirical_position_corr(x: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of the *positions* (no diff)."""
    K = x.shape[-1]
    z = x - x.mean(dim=1, keepdim=True)
    std = z.std(dim=1, keepdim=True).clamp(min=1e-6)
    z = z / std
    T = z.shape[1]
    corr = torch.einsum("bti,btj->bij", z, z) / T
    return correlation_to_pairs(corr, K=K)


def evaluate(head, model, K, T_raw, device, num_samples=400, batch_size=32, seed_base=10000,
             n_factors=2, latent="h"):
    """Run `num_samples` samples through head + collect baselines.
    Returns per-pair statistics and concatenated arrays."""
    head.eval()
    all_pred, all_true, all_diffbase, all_posbase = [], [], [], []

    n_batches = (num_samples + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(n_batches):
            bs = min(batch_size, num_samples - i * batch_size)
            x, _C, target = make_batch(bs, K, T_raw, device, seed=seed_base + i, n_factors=n_factors)
            h = extract_latents(model, x, latent)
            pred = head(h)
            all_pred.append(pred.cpu().numpy())
            all_true.append(target.cpu().numpy())
            all_diffbase.append(empirical_corr_baseline(x).cpu().numpy())
            all_posbase.append(empirical_position_corr(x).cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    diffbase = np.concatenate(all_diffbase, axis=0)
    posbase = np.concatenate(all_posbase, axis=0)

    # Per-pair metrics
    num_pairs = pred.shape[1]
    per_pair = []
    for j in range(num_pairs):
        t, p, db, pb = true[:, j], pred[:, j], diffbase[:, j], posbase[:, j]
        per_pair.append({
            "pair_index": j,
            "mse": float(np.mean((t - p) ** 2)),
            "mae": float(np.mean(np.abs(t - p))),
            "pearson_r": float(np.corrcoef(t, p)[0, 1]) if np.std(p) > 1e-8 else 0.0,
            "diff_baseline_mse": float(np.mean((t - db) ** 2)),
            "diff_baseline_pearson_r": float(np.corrcoef(t, db)[0, 1]) if np.std(db) > 1e-8 else 0.0,
            "pos_baseline_mse": float(np.mean((t - pb) ** 2)),
            "pos_baseline_pearson_r": float(np.corrcoef(t, pb)[0, 1]) if np.std(pb) > 1e-8 else 0.0,
            "true_mean": float(np.mean(t)),
            "true_std": float(np.std(t)),
        })

    overall_mse = float(np.mean((true - pred) ** 2))
    overall_mae = float(np.mean(np.abs(true - pred)))
    zero_baseline_mse = float(np.mean(true ** 2))
    mean_baseline_mse = float(np.mean((true - true.mean()) ** 2))
    diff_baseline_mse = float(np.mean((true - diffbase) ** 2))
    pos_baseline_mse = float(np.mean((true - posbase) ** 2))

    summary = {
        "num_samples": int(true.shape[0]),
        "overall_mse": overall_mse,
        "overall_mae": overall_mae,
        "zero_baseline_mse": zero_baseline_mse,
        "mean_baseline_mse": mean_baseline_mse,
        "diff_baseline_mse": diff_baseline_mse,
        "pos_baseline_mse": pos_baseline_mse,
        "improvement_vs_zero": zero_baseline_mse / overall_mse if overall_mse > 0 else 0.0,
        "improvement_vs_mean": mean_baseline_mse / overall_mse if overall_mse > 0 else 0.0,
        "improvement_vs_pos_baseline": pos_baseline_mse / overall_mse if overall_mse > 0 else 0.0,
        "improvement_vs_diff_baseline": diff_baseline_mse / overall_mse if overall_mse > 0 else 0.0,
        "per_pair": per_pair,
    }
    arrays = {"pred": pred, "true": true, "diffbase": diffbase, "posbase": posbase}
    return summary, arrays


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-path", type=str, required=True)
    # Backbone config (must match)
    parser.add_argument("--encoder-type", type=str, default="gru")
    parser.add_argument("--H", type=int, default=1024)
    parser.add_argument("--W", type=int, default=32)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ffn-mult", type=float, default=4.0)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--depthwise-conv", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--intermediate-dim", type=int, default=None)
    # Head
    parser.add_argument("--head-type", type=str, default="linear", choices=list(HEADS))
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent", type=str, default="h", choices=["h", "h_hat"],
                        help="Which backbone latent to feed to the head")
    # Training
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--T-raw", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-factors", type=int, default=2)
    parser.add_argument("--head-path", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-samples", type=int, default=400)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--lr-schedule", type=str, default="none",
                        choices=["none", "cosine"])
    parser.add_argument("--warmup-epochs", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    if args.head_path is None:
        args.head_path = f"correlation_recovery_{args.head_type}.pth"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load backbone
    print(f"Loading backbone from {args.model_path}")
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
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Backbone loaded. encoder={args.encoder_type}, H={args.H}, layers={args.num_layers}")

    # Build head
    K = args.C
    head_cls = HEADS[args.head_type]
    if args.head_type in ("mlp", "timeaware"):
        head = head_cls(H=args.H, K=K, hidden_dim=args.hidden_dim)
    else:
        head = head_cls(H=args.H, K=K)
    head = head.to(device)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Head: {args.head_type}, params={n_params:,}")

    if args.evaluate:
        best_path = args.head_path.replace(".pth", "_best.pth")
        load_path = best_path if os.path.exists(best_path) else args.head_path
        print(f"Loading head from {load_path}")
        head.load_state_dict(torch.load(load_path, map_location=device))
        summary, arrays = evaluate(
            head, model, K, args.T_raw, device,
            num_samples=args.eval_samples, batch_size=args.batch_size,
            n_factors=args.n_factors, latent=args.latent,
        )
        print_summary(summary)
        return

    optimizer = optim.AdamW(head.parameters(), lr=args.lr)

    # Fixed validation set (build h_val in chunks of args.batch_size to keep the
    # GRU encoder workspace small).
    val_size = max(args.batch_size * 4, 64)
    val_chunks_x, val_chunks_h, val_chunks_t = [], [], []
    for i in range(0, val_size, args.batch_size):
        bs_i = min(args.batch_size, val_size - i)
        xv, _Cv, tv = make_batch(
            bs_i, K, args.T_raw, device, seed=10**6 + i, n_factors=args.n_factors,
        )
        val_chunks_h.append(extract_latents(model, xv, args.latent))
        val_chunks_t.append(tv)
    h_val = torch.cat(val_chunks_h, dim=0)
    target_val = torch.cat(val_chunks_t, dim=0)

    print(f"\nTraining {args.head_type} head for {args.epochs} epochs, "
          f"batch={args.batch_size}, lr={args.lr}")

    best_val = float("inf")
    best_epoch = 0
    history = []

    import math as _math

    def lr_at(epoch: int) -> float:
        if args.lr_schedule == "none":
            return args.lr
        if epoch <= args.warmup_epochs:
            return args.lr * epoch / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + _math.cos(_math.pi * progress)))

    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad()
        x, _C, target = make_batch(
            args.batch_size, K, args.T_raw, device, seed=epoch, n_factors=args.n_factors,
        )
        h = extract_latents(model, x, args.latent)
        pred = head(h)
        loss = F.mse_loss(pred, target)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        if args.lr_schedule != "none":
            current_lr = lr_at(epoch)
            for g in optimizer.param_groups:
                g["lr"] = current_lr
        optimizer.step()

        head.eval()
        with torch.no_grad():
            pred_val = head(h_val)
            val_loss = F.mse_loss(pred_val, target_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(head.state_dict(), args.head_path.replace(".pth", "_best.pth"))

        if epoch % args.log_every == 0:
            print(
                f"[Epoch {epoch:6d}] train={loss.item():.6f} | val={val_loss:.6f} | "
                f"best={best_val:.6f}@{best_epoch}",
                flush=True,
            )
            history.append({
                "epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss,
                "best_val": best_val, "best_epoch": best_epoch,
            })

        if epoch % args.save_every == 0:
            torch.save(head.state_dict(), args.head_path)

    torch.save(head.state_dict(), args.head_path)
    print(f"\nTraining complete. Best val_loss={best_val:.6f} at epoch {best_epoch}")

    # Final evaluation with best head
    head.load_state_dict(torch.load(args.head_path.replace(".pth", "_best.pth"), map_location=device))
    summary, arrays = evaluate(
        head, model, K, args.T_raw, device,
        num_samples=args.eval_samples, batch_size=args.batch_size,
        n_factors=args.n_factors,
    )
    print_summary(summary)

    summary["head_type"] = args.head_type
    summary["head_n_params"] = n_params
    summary["best_epoch"] = best_epoch
    summary["best_val_loss"] = best_val
    summary["history_excerpt"] = history[-50:]
    results_path = args.head_path.replace(".pth", "_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_path}")


def print_summary(summary):
    print("\n=== Correlation Recovery ===")
    print(f"Samples:                {summary['num_samples']}")
    print(f"Overall MSE:            {summary['overall_mse']:.6f}")
    print(f"Overall MAE:            {summary['overall_mae']:.6f}")
    print(f"Zero baseline MSE:      {summary['zero_baseline_mse']:.6f}")
    print(f"Mean baseline MSE:      {summary['mean_baseline_mse']:.6f}")
    print(f"Diff baseline MSE:      {summary['diff_baseline_mse']:.6f}  (corrcoef of diff(x))")
    print(f"Position baseline MSE:  {summary['pos_baseline_mse']:.6f}  (corrcoef of x)")
    print(f"Improvement vs zero:    {summary['improvement_vs_zero']:.2f}x")
    print(f"Improvement vs mean:    {summary['improvement_vs_mean']:.2f}x")
    print(f"Improvement vs diffbase:{summary['improvement_vs_diff_baseline']:.2f}x")
    print(f"Improvement vs posbase: {summary['improvement_vs_pos_baseline']:.2f}x")
    print()
    print("Per-pair Pearson r (head | diff-base | pos-base):")
    for p in summary["per_pair"]:
        print(
            f"  pair[{p['pair_index']}]  r_head={p['pearson_r']:.3f}  "
            f"r_diffbase={p['diff_baseline_pearson_r']:.3f}  "
            f"r_posbase={p['pos_baseline_pearson_r']:.3f}  "
            f"mae={p['mae']:.4f}"
        )


if __name__ == "__main__":
    main()
