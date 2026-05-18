"""Per-stage U_b probe for ResidualSiLUEncoder under orthogonal-subspace inits.

We test whether constructing W_proj and W_skip (and optionally mlp[2]) so that
their column spaces (image subspaces in R^H=R^384) are mutually orthogonal can
lift encoder_out effective rank above the default ~16.4.

Schemes:
  - default
  - ortho_skip_proj_orthogonal_images
        Q ∈ R^(384, 44) via QR(randn(384, 44)).
        proj.weight = Q[:, :22] * sqrt(1/22)
        skip.weight = Q[:, 22:44] * sqrt(1/22)
        mlp left at default.
  - ortho_all_three_subspaces
        Q ∈ R^(384, 66) via QR(randn(384, 66)).
        proj.weight = Q[:, :22] * sqrt(1/22)
        skip.weight = Q[:, 22:44] * sqrt(1/22)
        mlp[2].weight image space projected onto Q[:, 44:66]:
            mlp[2].weight ← P @ default_init, where P = Q[:, 44:66] @ Q[:, 44:66].T.
  - ortho_skip_proj_only_random_mlp
        Same as ortho_skip_proj_orthogonal_images but explicit no-op on mlp.

Stages captured (same as u_per_stage_residual_silu.py):
  input_concat, proj_out, mlp_pre_silu, mlp_post_silu, mlp_out,
  h_after_residual, skip_out, pre_norm, encoder_out.
"""
import csv
import math
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
H = 384
W_EFF = 22  # input width into the encoder (W=16 + freq_emb=3 + seas_emb=3)


def _instrumented_forward(self: ResidualSiLUEncoder, x: torch.Tensor):
    captured = {}
    captured["input_concat"] = x

    proj_out = self.proj(x)
    captured["proj_out"] = proj_out

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


# -------- Init schemes --------------------------------------------------------

def _apply_default(model: ConfigurableModel) -> None:
    """No-op."""
    return


def _q_with_seed(rows: int, cols: int, seed: int) -> torch.Tensor:
    """QR of a Gaussian matrix; returns rows x cols orthonormal Q."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(rows, cols, generator=g)
    Q, _ = torch.linalg.qr(A)  # Q: rows x cols (since cols <= rows)
    return Q


def _apply_ortho_skip_proj(model: ConfigurableModel, *, seed: int) -> None:
    """proj.weight and skip.weight have orthogonal column spaces in R^384.

    nn.Linear(in, out) stores weight as (out, in); the column space (image)
    of the linear map y = W x is span of the COLUMNS of W as an
    (out x in) matrix. Q[:, :22] is (384, 22) with orthonormal columns whose
    span is a 22-D subspace of R^384, exactly as needed for the image of
    Linear(22, 384). We assign that directly and zero the bias.
    """
    enc = model.encoder
    assert isinstance(enc, ResidualSiLUEncoder)
    Q = _q_with_seed(H, 2 * W_EFF, seed=seed + 99_000)
    scale = math.sqrt(1.0 / W_EFF)
    enc.proj.weight.data = Q[:, :W_EFF].clone() * scale
    enc.skip.weight.data = Q[:, W_EFF:2 * W_EFF].clone() * scale
    if enc.proj.bias is not None:
        enc.proj.bias.data.zero_()
    if enc.skip.bias is not None:
        enc.skip.bias.data.zero_()


def _apply_ortho_skip_proj_only_random_mlp(model: ConfigurableModel, *, seed: int) -> None:
    """Identical to _apply_ortho_skip_proj; MLP explicitly untouched."""
    _apply_ortho_skip_proj(model, seed=seed)
    # No-op on mlp by construction.


def _apply_ortho_all_three(model: ConfigurableModel, *, seed: int) -> None:
    """Three orthogonal 22-D image subspaces of R^384 for proj, skip, and
    mlp[2] outputs.

    For mlp[2] (Linear(384, 384)), we want its image to lie in span(Q[:, 44:66]).
    With weight W' (shape (384, 384)), output is W' @ x; its image is the
    column space of W'. Setting W' ← P @ W'_default, where P = Q[:, 44:66] @
    Q[:, 44:66].T is the orthogonal projector onto the 22-D subspace,
    forces every column of the resulting matrix to lie inside that subspace,
    so the image of mlp[2] is contained in span(Q[:, 44:66]). The bias is
    zeroed so it can't add a component outside that subspace.
    """
    enc = model.encoder
    assert isinstance(enc, ResidualSiLUEncoder)
    Q = _q_with_seed(H, 3 * W_EFF, seed=seed + 99_000)
    scale = math.sqrt(1.0 / W_EFF)
    enc.proj.weight.data = Q[:, :W_EFF].clone() * scale
    enc.skip.weight.data = Q[:, W_EFF:2 * W_EFF].clone() * scale
    if enc.proj.bias is not None:
        enc.proj.bias.data.zero_()
    if enc.skip.bias is not None:
        enc.skip.bias.data.zero_()

    Q_mlp = Q[:, 2 * W_EFF:3 * W_EFF]  # (384, 22)
    P = Q_mlp @ Q_mlp.T  # (384, 384) — projector onto 22-D subspace
    mlp_lin2 = enc.mlp[2]
    assert isinstance(mlp_lin2, torch.nn.Linear)
    mlp_lin2.weight.data = P @ mlp_lin2.weight.data
    if mlp_lin2.bias is not None:
        mlp_lin2.bias.data.zero_()


SCHEMES = {
    "default":                            _apply_default,
    "ortho_skip_proj_orthogonal_images":  _apply_ortho_skip_proj,
    "ortho_all_three_subspaces":          _apply_ortho_all_three,
    "ortho_skip_proj_only_random_mlp":    _apply_ortho_skip_proj_only_random_mlp,
}


# -------- Driver --------------------------------------------------------------

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
def _capture(*, seed: int, scheme: str) -> dict:
    m = _build_model(seed=seed)
    apply_fn = SCHEMES[scheme]
    if scheme == "default":
        apply_fn(m)
    else:
        apply_fn(m, seed=seed)

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
    out_csv = out_dir / "init_u_orthogonal_subspaces.csv"

    rows = []
    t0 = time.time()
    for scheme in SCHEMES:
        print(f"=== scheme={scheme} ===")
        for seed in SEEDS:
            captured = _capture(seed=seed, scheme=scheme)
            for stage in STAGE_ORDER:
                z = captured[stage]
                d = z.shape[-1]
                assert d == STAGE_FEATURE_DIMS[stage], (stage, d)
                u = float(dim_usage(z, axis=0).item())
                er = u * d
                rows.append(dict(
                    seed=seed,
                    init_scheme=scheme,
                    stage=stage,
                    feature_dim=d,
                    u_b_per_slice=u,
                    effective_rank=er,
                ))
                print(f"  seed={seed} stage={stage:18s} d={d:3d} U_b={u:.4f} er={er:.2f}")
    elapsed = time.time() - t0
    print(f"elapsed: {elapsed:.1f}s")

    fieldnames = ["seed", "init_scheme", "stage",
                  "feature_dim", "u_b_per_slice", "effective_rank"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}  rows={len(rows)}")


if __name__ == "__main__":
    main()
