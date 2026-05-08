"""Per-encoder-type U_b probe at default init.

For each of the 5 encoder types in src/encoders.py, build a ConfigurableModel
with the *same* surrounding kwargs (transformer 6L, RevEWMA 128, freq+seas
embeddings dim 3 each), forward the same synthetic input, and capture the
encoder output (the [B, T, C, H] tensor pre-transformer). Report U_b on that.

We monkey-patch the encoder.forward so we can grab the encoder output even
though ConfigurableModel.forward chains it into the transformer immediately.
The encoder is invoked from inside TransformerBlock; we only need its output,
so the simplest approach is to call the encoder directly on the prepared
encoder input — bypassing the transformer altogether.
"""
import csv
import time
from pathlib import Path

import torch

from src.metrics import dim_usage
from src.models import ConfigurableModel


BASE_CFG = dict(
    C=1, H=384, W=16,
    num_layers=6, nhead=6, ffn_mult=4.0,
    activation="gelu", depthwise_conv=3, dropout=0.0,
    freq_emb_dim=3, seasonality_emb_dim=3,
    rev_norm_kind="ewma", rev_norm_span=128,
)
T_RAW = 4096
INPUT_SCALE = 0.5
B = 256
SEEDS = (42, 43, 44)
ENCODER_TYPES = ("mlp", "mlp_wide", "residual_silu", "gru", "conv")


def _build_inputs(*, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    x = torch.randn(B, T_RAW, 1, generator=g) * INPUT_SCALE
    freq_ids = torch.randint(0, 4, (B,), generator=g)
    seas_ids = torch.randint(0, 4, (B,), generator=g)
    return x, freq_ids, seas_ids


def _build_model(*, seed: int, encoder_type: str) -> ConfigurableModel:
    torch.manual_seed(seed)
    m = ConfigurableModel(encoder_type=encoder_type, **BASE_CFG).to("cpu")
    m.eval()
    return m


@torch.no_grad()
def _encoder_out_for(*, seed: int, encoder_type: str) -> torch.Tensor:
    m = _build_model(seed=seed, encoder_type=encoder_type)
    x, freq_ids, seas_ids = _build_inputs(seed=seed)
    x_norm = m.rev_norm(x, mode="norm") if m.rev_norm is not None else x
    xr = m.prepare_encoder_input(x_norm, freq_ids=freq_ids, seasonality_ids=seas_ids)
    o_lat = m.encoder(xr)  # [B, T, C, H]
    return o_lat


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "init_u_per_encoder.csv"

    rows = []
    t0 = time.time()
    for enc_type in ENCODER_TYPES:
        for seed in SEEDS:
            o_lat = _encoder_out_for(seed=seed, encoder_type=enc_type)
            assert o_lat.shape == (B, T_RAW // 16, 1, 384), o_lat.shape
            u = float(dim_usage(o_lat, axis=0).item())
            er = u * 384
            print(f"  enc={enc_type:14s} seed={seed} U_b={u:.4f} eff_rank={er:.2f}")
            rows.append(dict(
                seed=seed, encoder_type=enc_type,
                u_b_per_slice=u, effective_rank=er,
            ))
    elapsed = time.time() - t0
    print(f"elapsed: {elapsed:.1f}s")

    fieldnames = ["seed", "encoder_type", "u_b_per_slice", "effective_rank"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}  rows={len(rows)}")


if __name__ == "__main__":
    main()
