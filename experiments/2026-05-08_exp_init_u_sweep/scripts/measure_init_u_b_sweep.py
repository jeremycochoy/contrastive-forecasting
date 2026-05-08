"""Measure encoder-latent dimension usage U at init across batch sizes B.

Re-measure U at init for the patch-encoder output with the *training-time*
batch B=256, not the small B=32 used in measure_init_u_w_sweep.py.

Hypothesis (to test, not assume): the per-slice U_b for encoder_init at
B=32 is biased low because the (B, B-1) cosine matrix has too few samples
relative to d=H=384. Increasing B to 256 should bring the per-slice
estimate closer to the global-pooled (over B*T*C samples) estimate.

For each (B, seed):
  - synthetic isotropic input z ~ N(0, I)_{B,T,1,H}: should give U≈1
    for global_pooled, with per_slice rising toward 1 as B grows.
  - encoder_init: forward synthetic input through default-init encoder
    and measure U on o_lat = (B, T, 1, H).

Config: default init, W=16, freq_emb_dim=3, seasonality_emb_dim=3,
dropout=0.0. T_raw=4096 -> T=256. CPU. 3 model seeds (42, 43, 44),
except B=256 may drop to 1 seed if too slow.
"""

import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models import ConfigurableModel
from src.metrics import dim_usage


BASE_CFG = dict(
    C=1, H=384,
    encoder_type="gru",
    num_layers=6, nhead=6, ffn_mult=4.0,
    activation="gelu", depthwise_conv=3, dropout=0.0,
    rev_norm_kind="ewma", rev_norm_span=128,
    W=16,
    freq_emb_dim=3,
    seasonality_emb_dim=3,
)

T_RAW = 4096
INPUT_SCALE = 0.5
SEEDS = (42, 43, 44)
BATCH_SIZES = (32, 64, 128, 256)


def _u_global_pooled(z: torch.Tensor) -> float:
    """Mean cos² over all B·T·C flattened samples once and 1/(d * mean cos²).

    z is (B, T, C, H). Flatten the first three axes into one n axis of
    size N = B*T*C, and compute the off-diagonal mean of cos²(z_i, z_j).
    Returns a single scalar U = 1 / (d * mean_cos2), clipped to [0, 1].
    """
    B, T, C, H = z.shape
    flat = z.reshape(B * T * C, H)
    n = flat.shape[0]
    d = flat.shape[-1]
    flat_norm = F.normalize(flat, p=2, dim=-1, eps=1e-12)
    sim = flat_norm @ flat_norm.t()
    sq = sim.pow(2)
    sum_all = sq.sum()
    sum_diag = torch.diagonal(sq).sum()
    off_mean = (sum_all - sum_diag) / (n * (n - 1))
    u = (1.0 / (d * off_mean.clamp_min(1e-12))).clamp_max(1.0)
    return float(u)


def measure_encoder(*, seed: int, B: int) -> dict:
    torch.manual_seed(seed)
    cfg = dict(BASE_CFG)
    m = ConfigurableModel(**cfg).to("cpu")
    m.eval()

    g = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    x = torch.randn(B, T_RAW, 1, generator=g) * INPUT_SCALE
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
        o_lat = o_flat.reshape(B, 1, T, H).permute(0, 2, 1, 3)  # (B, T, 1, H)

        u_b_per_slice = dim_usage(o_lat, axis=0).item()
        u_t_per_slice = dim_usage(o_lat, axis=1).item()
        u_b_global_pooled = _u_global_pooled(o_lat)

    return dict(
        u_b_per_slice=u_b_per_slice,
        u_b_global_pooled=u_b_global_pooled,
        u_t_per_slice=u_t_per_slice,
    )


def measure_isotropic(*, seed: int, B: int) -> dict:
    """Synthetic isotropic input directly: z ~ N(0,I)_{B,T,1,H}."""
    torch.manual_seed(seed + 99_999)
    T = T_RAW // BASE_CFG["W"]
    H = BASE_CFG["H"]
    z = torch.randn(B, T, 1, H)
    with torch.no_grad():
        u_b_per_slice = dim_usage(z, axis=0).item()
        u_t_per_slice = dim_usage(z, axis=1).item()
        u_b_global_pooled = _u_global_pooled(z)
    return dict(
        u_b_per_slice=u_b_per_slice,
        u_b_global_pooled=u_b_global_pooled,
        u_t_per_slice=u_t_per_slice,
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    fieldnames = [
        "setup", "B", "seed",
        "u_b_per_slice", "u_b_global_pooled", "u_t_per_slice",
    ]

    print("=== isotropic_synthetic ===")
    for B in BATCH_SIZES:
        seeds = SEEDS  # isotropic is cheap
        for seed in seeds:
            t0 = time.time()
            r = measure_isotropic(seed=seed, B=B)
            dt = time.time() - t0
            print(
                f"  B={B:>3d} seed={seed}  "
                f"u_b_ps={r['u_b_per_slice']:.4f}  "
                f"u_b_gp={r['u_b_global_pooled']:.4f}  "
                f"u_t_ps={r['u_t_per_slice']:.4f}  ({dt:.1f}s)"
            )
            rows.append(dict(setup="isotropic_synthetic", B=B, seed=seed, **r))

    print("=== encoder_init ===")
    for B in BATCH_SIZES:
        seeds = SEEDS if B < 256 else (42,)  # B=256 may be slow: 1 seed
        for seed in seeds:
            t0 = time.time()
            r = measure_encoder(seed=seed, B=B)
            dt = time.time() - t0
            print(
                f"  B={B:>3d} seed={seed}  "
                f"u_b_ps={r['u_b_per_slice']:.4f}  "
                f"u_b_gp={r['u_b_global_pooled']:.4f}  "
                f"u_t_ps={r['u_t_per_slice']:.4f}  ({dt:.1f}s)"
            )
            rows.append(dict(setup="encoder_init", B=B, seed=seed, **r))

    csv_path = out_dir / "init_u_b_sweep.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print()
    print(f"wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
