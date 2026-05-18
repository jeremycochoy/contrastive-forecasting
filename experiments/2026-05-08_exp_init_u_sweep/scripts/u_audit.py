"""Audit dim_usage in src/metrics.py: per-slice vs global vs inverse-of-mean.

1) Reference loop implementation, agreement check.
2) Three U variants (per_slice, global_pooled, inverse_of_mean).
3) Sanity check on synthetic isotropic data.
4) Run on actual encoder-init outputs at W ∈ {16, 64, 192}.
5) Write u_audit.csv.
"""
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics import dim_usage
from src.models import ConfigurableModel


# ----------------------- reference (slow loop) implementation -----------------------

def _ref_per_slice_offmean(z_norm_slice: torch.Tensor) -> float:
    """Loop over (i, j) with i != j; compute mean cos² for one slice (n, d)."""
    n = z_norm_slice.shape[0]
    s = 0.0
    cnt = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = float(torch.dot(z_norm_slice[i], z_norm_slice[j]))
            s += c * c
            cnt += 1
    return s / cnt


def dim_usage_ref(z: torch.Tensor, axis: int) -> torch.Tensor:
    """Slow obvious reference: nested loops over slices and pairs."""
    if axis < 0:
        axis += z.ndim
    z = z.movedim(axis, 0)
    n = z.shape[0]
    d = z.shape[-1]
    if n < 2:
        return torch.ones((), device=z.device, dtype=z.dtype)
    z_norm = F.normalize(z, p=2, dim=-1, eps=1e-12)
    fixed_shape = z_norm.shape[1:-1]
    flat = z_norm.reshape(n, -1, d)            # (n, S, d)
    S = flat.shape[1]
    u_vals = []
    for s in range(S):
        slice_nd = flat[:, s, :]               # (n, d)
        off_mean = _ref_per_slice_offmean(slice_nd)
        u = min(1.0 / (d * max(off_mean, 1e-12)), 1.0)
        u_vals.append(u)
    u_t = torch.tensor(u_vals, dtype=z.dtype)
    return u_t.mean()


# ------------------------------- the three variants ---------------------------------

@torch.no_grad()
def u_per_slice(z: torch.Tensor, axis: int) -> float:
    """Existing dim_usage formula."""
    return float(dim_usage(z, axis=axis).item())


@torch.no_grad()
def u_global_pooled(z: torch.Tensor, axis: int) -> float:
    """Combine ALL non-feature axes into one sample axis. Single mean cos²."""
    if axis < 0:
        axis += z.ndim
    # Move the named "n" axis to the front to be consistent with per-slice's
    # axis semantics, then flatten ALL non-feature axes into one big sample axis.
    z = z.movedim(axis, 0)
    d = z.shape[-1]
    flat = z.reshape(-1, d)                     # (B*T*C, d)
    n = flat.shape[0]
    if n < 2:
        return 1.0
    fn = F.normalize(flat, p=2, dim=-1, eps=1e-12)
    sim = fn @ fn.t()                           # (n, n)
    sq = sim.pow(2)
    off_sum = float(sq.sum() - torch.diagonal(sq).sum())
    off_mean = off_sum / (n * (n - 1))
    u = 1.0 / (d * max(off_mean, 1e-12))
    return min(u, 1.0)


@torch.no_grad()
def u_inverse_of_mean(z: torch.Tensor, axis: int) -> float:
    """Same per-slice cos² as current code, but average off_mean over slices
    BEFORE taking 1/(d * mean). No Jensen bias from per-slice clamping."""
    if axis < 0:
        axis += z.ndim
    z = z.movedim(axis, 0)
    n = z.shape[0]
    d = z.shape[-1]
    if n < 2:
        return 1.0
    z_norm = F.normalize(z, p=2, dim=-1, eps=1e-12)
    flat = z_norm.reshape(n, -1, d).permute(1, 0, 2)   # (S, n, d)
    sim = torch.matmul(flat, flat.transpose(-1, -2))
    sq = sim.pow(2)
    sum_all = sq.sum(dim=(-1, -2))
    sum_diag = torch.diagonal(sq, dim1=-2, dim2=-1).sum(dim=-1)
    off_mean = (sum_all - sum_diag) / (n * (n - 1))    # (S,)
    avg_off = float(off_mean.mean())
    u = 1.0 / (d * max(avg_off, 1e-12))
    return min(u, 1.0)


# --------------------------- agreement on small tensors -----------------------------

def _check_agreement():
    diffs = []
    # 1) random small
    z = torch.randn(5, 7, 3, 11)
    for ax in (0, 1, 2):
        a = float(dim_usage(z, axis=ax).item())
        b = float(dim_usage_ref(z, axis=ax).item())
        diffs.append(abs(a - b))

    # 2) collinear: all rows identical -> cos²=1 -> U = 1/d
    d = 8
    base = torch.randn(d)
    z = base.unsqueeze(0).repeat(6, 1)         # (6, 8)
    a = float(dim_usage(z, axis=0).item())
    b = float(dim_usage_ref(z, axis=0).item())
    diffs.append(abs(a - b))
    # also check the value is ~1/d
    diffs.append(abs(a - 1.0 / d))

    # 3) orthonormal: identity rows -> cos²=0 -> U clamped to 1
    z = torch.eye(8)                           # (8, 8) rows are orthonormal
    a = float(dim_usage(z, axis=0).item())
    b = float(dim_usage_ref(z, axis=0).item())
    diffs.append(abs(a - b))
    diffs.append(abs(a - 1.0))

    return max(diffs)


# --------------------------- encoder init forward (mirrored) ------------------------

BASE_CFG = dict(
    C=1, H=384,
    encoder_type="gru",
    num_layers=6, nhead=6, ffn_mult=4.0,
    activation="gelu", depthwise_conv=3, dropout=0.0,
    rev_norm_kind="ewma", rev_norm_span=128,
)
B = 32
INPUT_SCALE = 0.5


def _t_raw_for_w(w: int) -> int:
    if w == 192:
        return 4032
    return 4096


def encoder_init_o_lat(*, seed: int, w: int) -> torch.Tensor:
    torch.manual_seed(seed)
    cfg = dict(BASE_CFG)
    cfg.update(W=w, freq_emb_dim=3, seasonality_emb_dim=3)
    m = ConfigurableModel(**cfg).to("cpu")
    m.eval()

    g = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    t_raw = _t_raw_for_w(w)
    x = torch.randn(B, t_raw, 1, generator=g) * INPUT_SCALE
    freq_ids = torch.randint(0, 4, (B,), generator=g)
    seas_ids = torch.randint(0, 4, (B,), generator=g)

    with torch.no_grad():
        H = m.H
        x_norm = m.rev_norm(x, mode="norm") if m.rev_norm is not None else x
        T = x_norm.shape[1] // m.W
        xr = m.prepare_encoder_input(
            x_norm,
            freq_ids=freq_ids,
            seasonality_ids=seas_ids,
        )
        _f_flat, o_flat = m.transformer(xr)
        o_lat = o_flat.reshape(B, 1, T, H).permute(0, 2, 1, 3)  # (B, T, C=1, H)
    return o_lat


# ----------------------------------- main -------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: agreement
    max_diff = _check_agreement()
    print(f"reference vs. torch agreement, max abs diff: {max_diff:.3e}")

    rows: list[dict] = []

    # Step 3: synthetic isotropic, 3 seeds
    print("=== isotropic synthetic ===")
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        z = torch.randn(B, 256, 1, 384)
        row = dict(
            setup="isotropic_synthetic",
            W=-1,
            seed=seed,
            u_b_per_slice=u_per_slice(z, axis=0),
            u_b_global_pooled=u_global_pooled(z, axis=0),
            u_b_inverse_of_mean=u_inverse_of_mean(z, axis=0),
            u_t_per_slice=u_per_slice(z, axis=1),
            u_t_global_pooled=u_global_pooled(z, axis=1),
            u_t_inverse_of_mean=u_inverse_of_mean(z, axis=1),
        )
        print(f"  seed={seed}  "
              f"u_b ps={row['u_b_per_slice']:.4f} gp={row['u_b_global_pooled']:.4f} "
              f"iom={row['u_b_inverse_of_mean']:.4f}  "
              f"u_t ps={row['u_t_per_slice']:.4f} gp={row['u_t_global_pooled']:.4f} "
              f"iom={row['u_t_inverse_of_mean']:.4f}")
        rows.append(row)

    # Step 4: encoder at init for W ∈ {16, 64, 192}, 3 seeds
    print("=== encoder init ===")
    for w in (16, 64, 192):
        for seed in (42, 43, 44):
            o_lat = encoder_init_o_lat(seed=seed, w=w)
            row = dict(
                setup="encoder_init",
                W=w,
                seed=seed,
                u_b_per_slice=u_per_slice(o_lat, axis=0),
                u_b_global_pooled=u_global_pooled(o_lat, axis=0),
                u_b_inverse_of_mean=u_inverse_of_mean(o_lat, axis=0),
                u_t_per_slice=u_per_slice(o_lat, axis=1),
                u_t_global_pooled=u_global_pooled(o_lat, axis=1),
                u_t_inverse_of_mean=u_inverse_of_mean(o_lat, axis=1),
            )
            print(f"  W={w:3d} seed={seed}  "
                  f"u_b ps={row['u_b_per_slice']:.4f} gp={row['u_b_global_pooled']:.4f} "
                  f"iom={row['u_b_inverse_of_mean']:.4f}  "
                  f"u_t ps={row['u_t_per_slice']:.4f} gp={row['u_t_global_pooled']:.4f} "
                  f"iom={row['u_t_inverse_of_mean']:.4f}")
            rows.append(row)

    # Step 5: write CSV
    csv_path = out_dir / "u_audit.csv"
    fieldnames = [
        "setup", "W", "seed",
        "u_b_per_slice", "u_b_global_pooled", "u_b_inverse_of_mean",
        "u_t_per_slice", "u_t_global_pooled", "u_t_inverse_of_mean",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}  rows={len(rows)}  max_ref_diff={max_diff:.3e}")
    return max_diff, rows


if __name__ == "__main__":
    main()
