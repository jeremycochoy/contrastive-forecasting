"""Measure encoder-latent dimension usage U at init under various init schemes.

Builds a fresh ConfigurableModel for each (scheme, seed) pair, applies the
init, runs one forward on synthetic input, and records U_t, U_b on both
o_lat (post-patch-head, pre-transformer) and f_lat (post-transformer).
"""

import csv
import math
from pathlib import Path

import torch
import torch.nn as nn

from src.models import ConfigurableModel
from src.metrics import dim_usage


CFG = dict(
    C=1, H=384, W=16,
    encoder_type="gru",
    num_layers=6, nhead=6, ffn_mult=4.0,
    activation="gelu", depthwise_conv=3, dropout=0.0,
    freq_emb_dim=3, seasonality_emb_dim=3,
    rev_norm_kind="ewma", rev_norm_span=128,
)

B, T_RAW, C = 32, 4096, 1
INPUT_SCALE = 0.5

SEEDS = (42, 43, 44)

SCHEMES = (
    "default",
    "ortho_patch",
    "ortho_all",
    "xavier_small_patch",
    "gpt_scaled",
    "skip_zero",
    "skip_smallgain",
    "proj_smallgain",
)


def _zero_bias(layer: nn.Module) -> None:
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)


def apply_init(model: ConfigurableModel, scheme: str) -> None:
    """Apply the named init scheme in-place to model."""
    if scheme == "default":
        return

    enc = model.encoder

    if scheme == "ortho_patch":
        nn.init.orthogonal_(enc.proj.weight, gain=math.sqrt(2.0))
        _zero_bias(enc.proj)
        nn.init.orthogonal_(enc.skip.weight, gain=math.sqrt(2.0))
        _zero_bias(enc.skip)
        return

    if scheme == "ortho_all":
        nn.init.orthogonal_(enc.proj.weight, gain=math.sqrt(2.0))
        _zero_bias(enc.proj)
        nn.init.orthogonal_(enc.skip.weight, gain=math.sqrt(2.0))
        _zero_bias(enc.skip)
        for layer in model.transformer.layers:
            nn.init.orthogonal_(layer.linear1.weight, gain=math.sqrt(2.0))
            _zero_bias(layer.linear1)
            nn.init.orthogonal_(layer.linear2.weight, gain=math.sqrt(2.0))
            _zero_bias(layer.linear2)
            # MHA in_proj_weight is (3*d_model, d_model) — rectangular but
            # PyTorch's orthogonal_ handles non-square matrices.
            nn.init.orthogonal_(layer.self_attn.in_proj_weight,
                                gain=math.sqrt(2.0))
            if layer.self_attn.in_proj_bias is not None:
                nn.init.zeros_(layer.self_attn.in_proj_bias)
            nn.init.orthogonal_(layer.self_attn.out_proj.weight,
                                gain=math.sqrt(2.0))
            _zero_bias(layer.self_attn.out_proj)
        return

    if scheme == "xavier_small_patch":
        nn.init.xavier_uniform_(enc.proj.weight, gain=0.5)
        nn.init.xavier_uniform_(enc.skip.weight, gain=0.5)
        return

    if scheme == "gpt_scaled":
        # Scaled GPT-style init: residual-stream output projections shrunk
        # by 1/sqrt(2*L). proj is the patch-head residual-stream output.
        std = 1.0 / math.sqrt(2.0 * CFG["num_layers"])
        nn.init.normal_(enc.proj.weight, mean=0.0, std=std)
        for layer in model.transformer.layers:
            nn.init.normal_(layer.linear2.weight, mean=0.0, std=std)
            nn.init.normal_(layer.self_attn.out_proj.weight,
                            mean=0.0, std=std)
        return

    if scheme == "skip_zero":
        nn.init.zeros_(enc.skip.weight)
        if enc.skip.bias is not None:
            nn.init.zeros_(enc.skip.bias)
        return

    if scheme == "skip_smallgain":
        nn.init.xavier_uniform_(enc.skip.weight, gain=0.1)
        return

    if scheme == "proj_smallgain":
        nn.init.xavier_uniform_(enc.proj.weight, gain=0.1)
        return

    raise ValueError(f"Unknown scheme: {scheme}")


def measure_one(scheme: str, seed: int) -> dict:
    torch.manual_seed(seed)
    m = ConfigurableModel(**CFG).to("cpu")
    m.eval()

    apply_init(m, scheme)

    # Inputs (deterministic per seed for fair comparison across schemes)
    g = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    x = torch.randn(B, T_RAW, C, generator=g) * INPUT_SCALE
    freq_ids = torch.randint(0, 4, (B,), generator=g)
    seas_ids = torch.randint(0, 4, (B,), generator=g)

    with torch.no_grad():
        H = m.H
        x_norm = m.rev_norm(x, mode="norm") if m.rev_norm is not None else x
        T_raw_ = x_norm.shape[1]
        T = T_raw_ // m.W
        xr = m.prepare_encoder_input(
            x_norm,
            freq_ids=freq_ids,
            seasonality_ids=seas_ids,
        )
        f_flat, o_flat = m.transformer(xr)
        f_lat = f_flat.reshape(B, C, T, H).permute(0, 2, 1, 3)  # (B, T, C, H)
        o_lat = o_flat.reshape(B, C, T, H).permute(0, 2, 1, 3)

        u_t_o = dim_usage(o_lat, axis=1).item()
        u_b_o = dim_usage(o_lat, axis=0).item()
        u_t_f = dim_usage(f_lat, axis=1).item()
        u_b_f = dim_usage(f_lat, axis=0).item()

    return dict(
        name=scheme, seed=seed,
        u_t_o=u_t_o, u_b_o=u_b_o,
        u_t_f=u_t_f, u_b_f=u_b_f,
    )


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean ± std per scheme over its 3 seeds."""
    by_name: dict[str, list[dict]] = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r)

    out = []
    for name, rs in by_name.items():
        vals = {k: torch.tensor([r[k] for r in rs])
                for k in ("u_t_o", "u_b_o", "u_t_f", "u_b_f")}
        rec = {"name": name}
        for k, v in vals.items():
            rec[f"{k}_mean"] = float(v.mean())
            rec[f"{k}_std"] = float(v.std(unbiased=False))
        out.append(rec)
    return out


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    for scheme in SCHEMES:
        for seed in SEEDS:
            r = measure_one(scheme, seed)
            raw_rows.append(r)
            print(f"  {scheme:<22s} seed={seed}  "
                  f"u_b_o={r['u_b_o']:.4f}  u_t_o={r['u_t_o']:.4f}  "
                  f"u_b_f={r['u_b_f']:.4f}  u_t_f={r['u_t_f']:.4f}")

    raw_csv = out_dir / "init_u_sweep_raw.csv"
    with raw_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "name", "seed", "u_t_o", "u_b_o", "u_t_f", "u_b_f"])
        w.writeheader()
        w.writerows(raw_rows)

    agg_rows = aggregate(raw_rows)
    agg_csv = out_dir / "init_u_sweep.csv"
    with agg_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "name",
            "u_t_o_mean", "u_t_o_std",
            "u_b_o_mean", "u_b_o_std",
            "u_t_f_mean", "u_t_f_std",
            "u_b_f_mean", "u_b_f_std",
        ])
        w.writeheader()
        w.writerows(agg_rows)

    print()
    print(f"wrote {raw_csv} ({len(raw_rows)} rows)")
    print(f"wrote {agg_csv} ({len(agg_rows)} rows)")

    # Summary sorted by u_b_o_mean desc
    print()
    print(f"{'scheme':<22s}  {'u_b_o':>14s}  {'u_t_o':>14s}  "
          f"{'u_b_f':>14s}  {'u_t_f':>14s}")
    for r in sorted(agg_rows, key=lambda r: -r["u_b_o_mean"]):
        print(f"{r['name']:<22s}  "
              f"{r['u_b_o_mean']:.4f}±{r['u_b_o_std']:.4f}  "
              f"{r['u_t_o_mean']:.4f}±{r['u_t_o_std']:.4f}  "
              f"{r['u_b_f_mean']:.4f}±{r['u_b_f_std']:.4f}  "
              f"{r['u_t_f_mean']:.4f}±{r['u_t_f_std']:.4f}")


if __name__ == "__main__":
    main()
