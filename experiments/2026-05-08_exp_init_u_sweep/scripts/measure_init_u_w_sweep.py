"""Measure encoder-latent dimension usage U at init across W, freq_emb_dim,
and a 'gru_zero' ablation.

Hypothesis: U at init on `o_lat` (post-patch-encoder, post-LN, pre-transformer)
is rank-limited by the per-token input dim feeding the patch encoder, ≈
(W + freq_emb_dim + seasonality_emb_dim) / H.

Three sub-experiments, all on the patch encoder of ConfigurableModel:
  A. Vary W ∈ {16, 32, 64, 128, 192} with freq=3, seas=3 (so the only
     thing that changes is W).
  B. Vary freq_emb_dim ∈ {3, 16, 64, 192} with W=16, seas=0.
  C. gru_zero: same as A's W=16 row but the GRU's output is zeroed —
     leaves only the linear `skip` path active.
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn

from src.models import ConfigurableModel
from src.metrics import dim_usage


# Static config bits not swept here (mirrors measure_init_u.py).
BASE_CFG = dict(
    C=1, H=384,
    encoder_type="gru",
    num_layers=6, nhead=6, ffn_mult=4.0,
    activation="gelu", depthwise_conv=3, dropout=0.0,
    rev_norm_kind="ewma", rev_norm_span=128,
)

B = 32
INPUT_SCALE = 0.5
SEEDS = (42, 43, 44)


def _t_raw_for_w(w: int) -> int:
    """Pick a T_raw that is a multiple of W, ~4096.
    W=16/32/64/128 -> 4096; W=192 -> 4032 (= 21 * 192)."""
    if w == 192:
        return 4032
    return 4096


def _zero_gru_output(model: ConfigurableModel) -> None:
    """Zero all GRU weights+biases so the GRU output is identically zero.
    Leaves `skip` and `proj` and `layer_norm` untouched. The encoder then
    reduces to LN(skip(x_concat)) at init."""
    gru = model.encoder.gru
    for name, p in gru.named_parameters():
        nn.init.zeros_(p)


def measure_one(
    *,
    seed: int,
    w: int,
    freq_emb_dim: int,
    seasonality_emb_dim: int,
    gru_zero: bool,
    t_raw: int,
) -> dict:
    torch.manual_seed(seed)
    cfg = dict(BASE_CFG)
    cfg.update(
        W=w,
        freq_emb_dim=freq_emb_dim,
        seasonality_emb_dim=seasonality_emb_dim,
    )
    m = ConfigurableModel(**cfg).to("cpu")
    m.eval()

    if gru_zero:
        _zero_gru_output(m)

    g = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    x = torch.randn(B, t_raw, 1, generator=g) * INPUT_SCALE
    freq_ids = torch.randint(0, 4, (B,), generator=g)
    seas_ids = torch.randint(0, 4, (B,), generator=g)

    with torch.no_grad():
        H = m.H
        x_norm = m.rev_norm(x, mode="norm") if m.rev_norm is not None else x
        T = x_norm.shape[1] // m.W
        xr = m.prepare_encoder_input(
            x_norm,
            freq_ids=freq_ids if freq_emb_dim > 0 else None,
            seasonality_ids=seas_ids if seasonality_emb_dim > 0 else None,
        )
        # We only need o_lat (the patch encoder's output, post-LN, pre-6L
        # transformer). transformer.forward returns (final, encoder_out).
        _f_flat, o_flat = m.transformer(xr)
        o_lat = o_flat.reshape(B, 1, T, H).permute(0, 2, 1, 3)  # (B, T, C=1, H)

        u_b_o = dim_usage(o_lat, axis=0).item()
        u_t_o = dim_usage(o_lat, axis=1).item()

    return dict(u_b_o=u_b_o, u_t_o=u_t_o)


def aggregate(rs: list[dict]) -> dict:
    keys = ("u_b_o", "u_t_o")
    out = {}
    for k in keys:
        v = torch.tensor([r[k] for r in rs])
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"] = float(v.std(unbiased=False))
    return out


def run_combo(
    *,
    experiment: str,
    w: int,
    freq_emb_dim: int,
    seasonality_emb_dim: int,
    gru_zero: bool,
    seeds=SEEDS,
) -> dict:
    t_raw = _t_raw_for_w(w)
    rs = []
    for seed in seeds:
        r = measure_one(
            seed=seed,
            w=w,
            freq_emb_dim=freq_emb_dim,
            seasonality_emb_dim=seasonality_emb_dim,
            gru_zero=gru_zero,
            t_raw=t_raw,
        )
        rs.append(r)
        print(
            f"  {experiment} W={w} f={freq_emb_dim} s={seasonality_emb_dim} "
            f"gru_zero={gru_zero} seed={seed}  "
            f"u_b_o={r['u_b_o']:.4f}  u_t_o={r['u_t_o']:.4f}"
        )
    agg = aggregate(rs)
    return dict(
        experiment=experiment,
        W=w,
        freq_emb_dim=freq_emb_dim,
        seasonality_emb_dim=seasonality_emb_dim,
        gru_zero=int(gru_zero),
        **agg,
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # Sub-experiment A: vary W with freq=3, seas=3 (matches baseline cfg).
    print("=== sub-experiment A: vary W ===")
    for w in (16, 32, 64, 128, 192):
        rows.append(run_combo(
            experiment="A_vary_W",
            w=w,
            freq_emb_dim=3,
            seasonality_emb_dim=3,
            gru_zero=False,
        ))

    # Sub-experiment B: vary freq_emb_dim, W=16, seas=0.
    print("=== sub-experiment B: vary freq_emb_dim ===")
    for f in (3, 16, 64, 192):
        rows.append(run_combo(
            experiment="B_vary_freq_emb",
            w=16,
            freq_emb_dim=f,
            seasonality_emb_dim=0,
            gru_zero=False,
        ))

    # Sub-experiment C: gru_zero ablation at W=16, freq=3, seas=3.
    print("=== sub-experiment C: gru_zero ablation ===")
    rows.append(run_combo(
        experiment="C_gru_zero",
        w=16,
        freq_emb_dim=3,
        seasonality_emb_dim=3,
        gru_zero=True,
    ))

    csv_path = out_dir / "init_u_w_sweep.csv"
    fieldnames = [
        "experiment", "W", "freq_emb_dim", "seasonality_emb_dim", "gru_zero",
        "u_b_o_mean", "u_b_o_std", "u_t_o_mean", "u_t_o_std",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"wrote {csv_path} ({len(rows)} rows)")
    print()
    print(f"{'experiment':<18s} {'W':>4s} {'f':>4s} {'s':>4s} {'gz':>3s} "
          f"{'u_b_o':>16s} {'u_t_o':>16s}")
    for r in rows:
        print(
            f"{r['experiment']:<18s} {r['W']:>4d} {r['freq_emb_dim']:>4d} "
            f"{r['seasonality_emb_dim']:>4d} {r['gru_zero']:>3d} "
            f"{r['u_b_o_mean']:.4f}±{r['u_b_o_std']:.4f}  "
            f"{r['u_t_o_mean']:.4f}±{r['u_t_o_std']:.4f}"
        )


if __name__ == "__main__":
    main()
