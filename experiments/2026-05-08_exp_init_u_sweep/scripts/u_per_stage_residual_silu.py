"""Per-stage U_b probe for ResidualSiLUEncoder at default init.

Mirrors u_per_stage.py (which probed GRUEncoder). Captures every intermediate
tensor inside ResidualSiLUEncoder.forward, plus the two intermediates inside
the self.mlp Sequential (pre-SiLU and post-SiLU). U_b is computed on the batch
axis (axis=0) for each stage.

ResidualSiLUEncoder.forward is:
    h = self.proj(x)                          # proj_out: Linear(W -> H)
    h = h + self.mlp(h)                       # mlp(h) = Linear(H, intermediate) -> SiLU -> Linear(intermediate, H)
    s = self.skip(x)                          # skip_out: Linear(W -> H)
    return self.layer_norm(h + s)             # encoder_out

intermediate_dim defaults to H (=384) when not provided.

Stages captured (in order):
  - input_concat   : (B, T, C, 22)   — raw patch + freq_emb(3) + seas_emb(3)
  - proj_out       : (B, T, C, 384)  — self.proj(x)
  - mlp_pre_silu   : (B, T, C, 384)  — first Linear in self.mlp
  - mlp_post_silu  : (B, T, C, 384)  — after SiLU
  - mlp_out        : (B, T, C, 384)  — second Linear in self.mlp (output of self.mlp(h))
  - h_after_residual : (B, T, C, 384) — proj_out + mlp_out
  - skip_out       : (B, T, C, 384)  — self.skip(x)
  - pre_norm       : (B, T, C, 384)  — h_after_residual + skip_out
  - encoder_out    : (B, T, C, 384)  — LayerNorm(pre_norm)
"""
import csv
import time
from pathlib import Path

import torch

from src.metrics import dim_usage
from src.models import ConfigurableModel
from src.encoders import ResidualSiLUEncoder


BASE_CFG = dict(
    C=1, H=384, W=16,
    encoder_type="residual_silu",
    num_layers=6, nhead=6, ffn_mult=4.0,
    activation="gelu", depthwise_conv=3, dropout=0.0,
    freq_emb_dim=3, seasonality_emb_dim=3,
    rev_norm_kind="ewma", rev_norm_span=128,
)
T_RAW = 4096
INPUT_SCALE = 0.5
B = 256
SEEDS = (42, 43, 44)


def _instrumented_forward(self: ResidualSiLUEncoder, x: torch.Tensor):
    """Replicates ResidualSiLUEncoder.forward exactly, stashes intermediates.

    The self.mlp Sequential is unrolled inline so we can grab its inner
    intermediates (pre-SiLU, post-SiLU). Layer references are taken from
    self.mlp's children; this preserves the exact same nn.Linear/nn.SiLU
    instances and weights as the real encoder.
    """
    captured = {}
    captured["input_concat"] = x

    proj_out = self.proj(x)
    captured["proj_out"] = proj_out

    # Unroll self.mlp = Sequential(Linear, SiLU, Linear)
    mlp_lin1, mlp_silu, mlp_lin2 = self.mlp[0], self.mlp[1], self.mlp[2]
    pre_silu = mlp_lin1(proj_out)
    captured["mlp_pre_silu"] = pre_silu
    post_silu = mlp_silu(pre_silu)
    captured["mlp_post_silu"] = post_silu
    mlp_out = mlp_lin2(post_silu)
    captured["mlp_out"] = mlp_out

    h_after_residual = proj_out + mlp_out
    captured["h_after_residual"] = h_after_residual

    skip_out = self.skip(x)
    captured["skip_out"] = skip_out

    pre_norm = h_after_residual + skip_out
    captured["pre_norm"] = pre_norm

    out = self.layer_norm(pre_norm)
    captured["encoder_out"] = out

    self._captured = captured
    return out


STAGE_FEATURE_DIMS = {
    "input_concat":     22,
    "proj_out":         384,
    "mlp_pre_silu":     384,
    "mlp_post_silu":    384,
    "mlp_out":          384,
    "h_after_residual": 384,
    "skip_out":         384,
    "pre_norm":         384,
    "encoder_out":      384,
}
STAGE_ORDER = [
    "input_concat", "proj_out", "mlp_pre_silu", "mlp_post_silu", "mlp_out",
    "h_after_residual", "skip_out", "pre_norm", "encoder_out",
]


def _build_inputs(*, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    x = torch.randn(B, T_RAW, 1, generator=g) * INPUT_SCALE
    freq_ids = torch.randint(0, 4, (B,), generator=g)
    seas_ids = torch.randint(0, 4, (B,), generator=g)
    return x, freq_ids, seas_ids


def _build_model(*, seed: int) -> ConfigurableModel:
    torch.manual_seed(seed)
    m = ConfigurableModel(**BASE_CFG).to("cpu")
    m.eval()
    return m


@torch.no_grad()
def _capture_for_seed(*, seed: int) -> dict:
    m = _build_model(seed=seed)
    enc = m.encoder
    assert isinstance(enc, ResidualSiLUEncoder)
    original = enc.forward
    enc.forward = _instrumented_forward.__get__(enc, ResidualSiLUEncoder)
    try:
        x, freq_ids, seas_ids = _build_inputs(seed=seed)
        x_norm = m.rev_norm(x, mode="norm") if m.rev_norm is not None else x
        xr = m.prepare_encoder_input(
            x_norm, freq_ids=freq_ids, seasonality_ids=seas_ids,
        )
        _ = enc(xr)
        captured = enc._captured
    finally:
        enc.forward = original
    return captured


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "u_per_stage_residual_silu.csv"

    rows = []
    t0 = time.time()
    for seed in SEEDS:
        captured = _capture_for_seed(seed=seed)
        for stage in STAGE_ORDER:
            z = captured[stage]
            d = z.shape[-1]
            assert d == STAGE_FEATURE_DIMS[stage], (stage, d)
            u = float(dim_usage(z, axis=0).item())
            er = u * d
            print(f"  seed={seed} stage={stage:18s} d={d:3d} U_b={u:.4f} eff_rank={er:.2f}")
            rows.append(dict(
                seed=seed, stage_name=stage, feature_dim=d,
                u_b_per_slice=u, effective_rank=er,
            ))
    elapsed = time.time() - t0
    print(f"elapsed: {elapsed:.1f}s")

    fieldnames = ["seed", "stage_name", "feature_dim", "u_b_per_slice", "effective_rank"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}  rows={len(rows)}")


if __name__ == "__main__":
    main()
