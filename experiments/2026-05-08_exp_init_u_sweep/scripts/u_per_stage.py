"""Per-stage U_b probe for GRUEncoder at init.

Captures every intermediate tensor inside GRUEncoder.forward whose last dim
is a feature dim where dim_usage(z, axis=0) is meaningful.

Stages:
  - input_concat  : (B, T, C, 22)  — concat of patch + freq + seas
  - gru_output    : (B*T*C, 256)   — bidir hidden cat (we reshape to (B, T, C, 256))
  - proj_out      : (B, T, C, 384) — Linear(256, 384)(gru_output)
  - skip_out      : (B, T, C, 384) — Linear(22, 384)(input_concat)
  - sum_pre_norm  : (B, T, C, 384) — proj_out + skip_out
  - encoder_out   : (B, T, C, 384) — LayerNorm(sum_pre_norm)

GRU internal gate non-linearities (tanh/sigmoid) are buried in PyTorch's
optimised cuDNN/CPU GRU and are not directly accessible without
re-implementing GRU step-by-step — noted and skipped per task spec.

The encoder is the only module with a non-affine hidden projection here;
GRUEncoder has NO optional non-linearity between proj and add.

Run with B=256 (per task spec). If wall time exceeds budget, fall back to
B=128 (also captured; we record actual_B in CSV).
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.metrics import dim_usage
from src.models import ConfigurableModel
from src.encoders import GRUEncoder


BASE_CFG = dict(
    C=1, H=384, W=16,
    encoder_type="gru",
    num_layers=6, nhead=6, ffn_mult=4.0,
    activation="gelu", depthwise_conv=3, dropout=0.0,
    freq_emb_dim=3, seasonality_emb_dim=3,
    rev_norm_kind="ewma", rev_norm_span=128,
)
T_RAW = 4096
INPUT_SCALE = 0.5
SEEDS = (42, 43, 44)


def _instrumented_forward(self: GRUEncoder, x: torch.Tensor):
    """Replicates GRUEncoder.forward exactly while stashing intermediates
    on self._captured. Verified line-for-line against src/encoders.py."""
    captured = {}
    # x: [B, T, C, W_eff]  (W_eff = W + freq_emb_dim + seasonality_emb_dim)
    shape = x.shape[:-1]               # [B, T, C]
    captured["input_concat"] = x       # (B, T, C, 22)

    flat = x.reshape(-1, self.W, 1)    # [B*T*C, W_eff, 1]
    _, hidden = self.gru(flat)
    h = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # (B*T*C, 256)
    captured["gru_output"] = h.reshape(*shape, -1)   # (B, T, C, 256)

    h_proj_flat = self.proj(h)         # (B*T*C, 384)
    h_proj = h_proj_flat.reshape(*shape, -1)         # (B, T, C, 384)
    captured["proj_out"] = h_proj

    s = self.skip(x)                   # (B, T, C, 384)
    captured["skip_out"] = s

    sum_pre = h_proj + s
    captured["sum_pre_norm"] = sum_pre

    out = self.layer_norm(sum_pre)
    captured["encoder_out"] = out

    self._captured = captured
    return out


STAGE_FEATURE_DIMS = {
    "input_concat": 22,
    "gru_output":   256,
    "proj_out":     384,
    "skip_out":     384,
    "sum_pre_norm": 384,
    "encoder_out":  384,
}
STAGE_ORDER = [
    "input_concat", "gru_output", "proj_out",
    "skip_out", "sum_pre_norm", "encoder_out",
]


def _orthogonal_init_proj_skip(model: ConfigurableModel) -> None:
    """Apply orthogonal init with gain=sqrt(2), zero bias, to proj and skip
    of the GRU encoder ONLY. GRU weights and LayerNorm are untouched."""
    enc = model.encoder
    assert isinstance(enc, GRUEncoder)
    gain = (2.0) ** 0.5
    nn.init.orthogonal_(enc.proj.weight, gain=gain)
    if enc.proj.bias is not None:
        nn.init.zeros_(enc.proj.bias)
    nn.init.orthogonal_(enc.skip.weight, gain=gain)
    if enc.skip.bias is not None:
        nn.init.zeros_(enc.skip.bias)


def _build_inputs(*, B: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    x = torch.randn(B, T_RAW, 1, generator=g) * INPUT_SCALE
    freq_ids = torch.randint(0, 4, (B,), generator=g)
    seas_ids = torch.randint(0, 4, (B,), generator=g)
    return x, freq_ids, seas_ids


def _build_model(seed: int) -> ConfigurableModel:
    torch.manual_seed(seed)
    m = ConfigurableModel(**BASE_CFG).to("cpu")
    m.eval()
    return m


@torch.no_grad()
def _capture_for_seed(*, seed: int, B: int, ortho: bool) -> dict[str, torch.Tensor]:
    m = _build_model(seed)
    if ortho:
        _orthogonal_init_proj_skip(m)

    # Patch the encoder forward.
    enc = m.encoder
    assert isinstance(enc, GRUEncoder)
    original_forward = enc.forward
    enc.forward = _instrumented_forward.__get__(enc, GRUEncoder)

    try:
        x, freq_ids, seas_ids = _build_inputs(B=B, seed=seed)
        x_norm = m.rev_norm(x, mode="norm") if m.rev_norm is not None else x
        xr = m.prepare_encoder_input(
            x_norm, freq_ids=freq_ids, seasonality_ids=seas_ids,
        )
        # Trigger the encoder by calling it directly. (We do not need the
        # transformer outputs for stage probing; the encoder is the
        # input_to_latent and we only care about its internals.)
        _ = enc(xr)
        captured = enc._captured
    finally:
        enc.forward = original_forward

    return captured


def _u_b_for_stage(z: torch.Tensor) -> tuple[float, int]:
    """Return (u_b_per_slice, feature_dim). z's last axis must be feature.
    Batch axis is 0 by construction in our captures."""
    d = z.shape[-1]
    u = float(dim_usage(z, axis=0).item())
    return u, d


def _run_setup(*, ortho: bool, B: int, out_csv: Path) -> list[dict]:
    rows: list[dict] = []
    label = "ortho" if ortho else "default"
    print(f"=== {label} init  B={B} ===")
    t0 = time.time()
    for seed in SEEDS:
        captured = _capture_for_seed(seed=seed, B=B, ortho=ortho)
        for stage in STAGE_ORDER:
            z = captured[stage]
            u, d = _u_b_for_stage(z)
            assert d == STAGE_FEATURE_DIMS[stage], (
                f"stage={stage} expected d={STAGE_FEATURE_DIMS[stage]} got {d}")
            er = u * d
            rows.append(dict(
                seed=seed, stage_name=stage, feature_dim=d,
                u_b_per_slice=u, effective_rank=er,
                actual_B=B,
            ))
            print(f"  seed={seed} stage={stage:13s} d={d:3d} U_b={u:.4f} eff_rank={er:.2f}")
    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.1f}s")

    fieldnames = ["seed", "stage_name", "feature_dim",
                  "u_b_per_slice", "effective_rank", "actual_B"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out_csv}  rows={len(rows)}\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    _run_setup(
        ortho=False, B=args.batch,
        out_csv=out_dir / "u_per_stage_default.csv",
    )
    _run_setup(
        ortho=True, B=args.batch,
        out_csv=out_dir / "u_per_stage_ortho.csv",
    )


if __name__ == "__main__":
    main()
