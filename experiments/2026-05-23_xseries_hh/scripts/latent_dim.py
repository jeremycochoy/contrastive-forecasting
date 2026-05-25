#!/usr/bin/env python3
"""Latent-dimensionality analysis for #318's frozen backbones.

Method replicated from the PRIOR experiment
``experiments/2026-05-08_exp_init_u_sweep`` (init-u-sweep). That card's
dimensionality metric is the cosine-based **dimension-usage**

    U = dim_usage(z, axis) = 1 / (d * mean_{i!=j} cos²(z_i, z_j)) ∈ (0, 1]

(``src.metrics.dim_usage``; per-slice, clamped to [0,1], slices averaged)
with **effective rank = U · feature_dim**. For isotropic z, U→1 (eff_rank→d);
for collinear z, U→1/d (eff_rank→1). We report it on the **encoder latent
``h`` = o_lat** (the contrastive representation, the 2nd output of
``model.transformer``; d=H=384), over both the batch axis (``U_b``,
axis=0) and the time axis (``U_t``, axis=1) — the two axes the template
uses. Companion plot: the normalized singular-value spectrum of the
mean-centred h matrix (visual only; eff_rank is the reported number).

ONE batch of REAL HF data (the #318 training stream, mix_ratio=0 → pure
HF) is fed through each frozen backbone. Forward-only, CPU by default so
it cannot disturb the live training jobs on either GPU.

Backbones are loaded exactly the way
``experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py``
loads ``--backbone-path`` (state-dict autodetect of freq/seas emb dims,
num_encoder_layers, patch_stats), with the #318 arch overrides:
GRU patch-enc → 6L causal encoder (d_model=384, n_heads=6) → 1L
forecaster (d=128, n_heads=4), ewma RevNorm span=128, t_raw=4096, C=1.

Writes:
  results/latent_dim.csv   (per-arm U_b, U_t, eff_rank_b, eff_rank_t, SV)
  plots/latent_dim.png     (SV spectrum log-y | effective-rank bars)
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- repo wiring (worktree = code; main checkout = run/results data) --------
WT = Path("/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/cross-series-hh")
EXP_WT = WT / "experiments/2026-05-23_xseries_hh"          # code/plots/results (writable)
EXP_DATA = Path("/home/jupyter/workspaces/contrastive-forecasting"
                "/experiments/2026-05-23_xseries_hh")       # runs/ (read-only inputs)
RUNS = EXP_DATA / "runs"
BETA_RUNS = Path("/home/jupyter/contrastive-forecasting"
                 "/experiments/2026-05-20_bottleneck_beta2_confound/runs")
PLOTS = EXP_WT / "plots"
RESULTS = EXP_WT / "results"
PLOTS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(WT))
from src.models import ConfigurableModel          # noqa: E402
from src.metrics import dim_usage                  # noqa: E402
from src.dataloader import create_mixed_periodic_dataloader  # noqa: E402

# --- #318 arch (matches train_backbone.sh / the eval-script overrides) ------
H = 384
ARCH = dict(
    C=1, H=H, W=16,
    encoder_type="gru", num_layers=1, nhead=6,
    num_encoder_layers=6, ffn_mult=4.0, activation="gelu",
    depthwise_conv=3, dropout=0.0,
    forecaster_d_model=128, forecaster_n_heads=4,
    rev_norm_kind="ewma", rev_norm_span=128,
)
T_RAW = 4096
HF_REPO = "jeremycochoy/gift-pretrain-full-4096"
HF_PATH = "small_v1"
BATCH = 128          # batch×time positions ≈ 128 × (4096/16=256) = 32,768 — plenty for the spectrum
DATA_SEED = 318

# --- arms (label, backbone path, colour) — colours mirror scripts/plots.py --
ARMS = [
    ("same-step",       RUNS / "bb_xshh_50k_FINAL.pth",              "#1f77b4"),
    ("all-time",        RUNS / "bb_xshh_allt_50k_FINAL.pth",         "#2ca02c"),
    ("forked allt·50%", RUNS / "bb_xshh_allt_forked_50k_FINAL.pth",  "#ff7f0e"),
    ("forked β·2/b",    RUNS / "bb_beta_forked2_50k_FINAL.pth",      "#e377c2"),
    ("β",               BETA_RUNS / "bb_beta_50k_FINAL.pth",         "#d62728"),
    # forked allt·2/b (bb_xshh_allt_forked2_50k_FINAL.pth) is added below
    # ONLY if its FINAL exists — it is still training at time of writing.
]
_FORKED2 = ("forked allt·2/b", RUNS / "bb_xshh_allt_forked2_50k_FINAL.pth", "#8c564b")
if _FORKED2[1].exists():
    ARMS.insert(3, _FORKED2)


def build_backbone(sd: dict) -> ConfigurableModel:
    """Construct + load a frozen backbone, autodetecting emb dims / patch_stats
    from the state dict exactly like eval_gift_eval_official.load_models."""
    cfg = dict(ARCH)
    w = sd.get("freq_embedding.embedding.weight")
    cfg["freq_emb_dim"] = (w.shape[1] if w is not None else 0)
    sw = sd.get("seasonality_embedding.embedding.weight")
    cfg["seasonality_emb_dim"] = (sw.shape[1] if sw is not None else 0)
    if "log_inv_tau" in sd:
        cfg["learnable_tau"] = True
    enc = {int(k.split(".")[2]) for k in sd
           if k.startswith("transformer.encoder_layers.")}
    if enc:
        cfg["num_encoder_layers"] = max(enc) + 1
    # patch_stats autodetect from the encoder input width (W + emb + extra)
    from src.norm import PATCH_STATS_DIM
    ref = sd.get("encoder.skip.weight")
    if ref is None:
        ref = sd.get("encoder.linear1.weight")
    if ref is None:
        cfg["patch_stats_kind"] = "none"
    else:
        extra = ref.shape[1] - cfg["W"] - cfg["freq_emb_dim"] - cfg["seasonality_emb_dim"]
        cfg["patch_stats_kind"] = {0: "none", PATCH_STATS_DIM: "diff"}[extra]
    m = ConfigurableModel(**cfg)
    m.load_state_dict(sd)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


@torch.no_grad()
def encoder_h(model: ConfigurableModel, x: torch.Tensor,
              freq_ids: torch.Tensor, seas_ids: torch.Tensor) -> torch.Tensor:
    """Encoder latent h = o_lat, shape (B, T, C=1, H). Mirrors train.forward_step:
    RevNorm(norm) -> prepare_encoder_input -> transformer -> 2nd output."""
    B, T_raw, C = x.shape
    T = T_raw // model.W
    xn = model.rev_norm(x, mode="norm") if model.rev_norm is not None else x
    xr = model.prepare_encoder_input(xn, freq_ids=freq_ids, seasonality_ids=seas_ids)
    _f_flat, o_flat = model.transformer(xr)            # o_flat = encoder out ("h")
    o_lat = o_flat.reshape(B, C, T, model.H).permute(0, 2, 1, 3)
    return o_lat                                        # (B, T, 1, H)


def sv_spectrum(h_flat: np.ndarray, k: int = 60) -> np.ndarray:
    """Normalized singular-value spectrum (σ_i / σ_0) of mean-centred h.
    h_flat: (N, H). Returns first k normalized singular values."""
    hc = h_flat - h_flat.mean(axis=0, keepdims=True)
    s = np.linalg.svd(hc, compute_uv=False)
    s = s / s[0]
    return s[:k]


def get_one_batch():
    """One batch of REAL HF data from the #318 training stream (pure HF)."""
    os.environ.setdefault("HF_TOKEN", (WT / "experiments/hf_token.txt").read_text().strip())
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
    loader = create_mixed_periodic_dataloader(
        repo_id=HF_REPO, batch_size=BATCH, C=ARCH["C"], mix_ratio=0.0,
        path_in_repo=HF_PATH, split="train", skip_rows=0, T_raw=T_RAW,
        seed=DATA_SEED, emit_freq_ids=True,
    )
    x, freq_ids, seas_ids = next(iter(loader))
    return x.float(), freq_ids, seas_ids


def main() -> None:
    device = torch.device(os.environ.get("LATENT_DIM_DEVICE", "cpu"))
    torch.manual_seed(DATA_SEED)
    x, freq_ids, seas_ids = get_one_batch()
    x, freq_ids, seas_ids = x.to(device), freq_ids.to(device), seas_ids.to(device)
    print(f"[latent_dim] device={device} x={tuple(x.shape)} "
          f"(B×T positions = {x.shape[0] * (x.shape[1] // ARCH['W'])})")

    rows: list[dict] = []
    spectra: dict[str, np.ndarray] = {}
    colours: dict[str, str] = {}

    for label, path, colour in ARMS:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        model = build_backbone(sd).to(device)
        h = encoder_h(model, x, freq_ids, seas_ids)        # (B, T, 1, H)
        # Template metric: dim_usage on h over batch axis and time axis.
        u_b = float(dim_usage(h, axis=0).item())
        u_t = float(dim_usage(h, axis=1).item())
        er_b, er_t = u_b * H, u_t * H
        # SV spectrum over all (B*T) positions, feature dim H.
        h_flat = h.reshape(-1, H).cpu().numpy().astype(np.float64)
        s = sv_spectrum(h_flat)
        # participation ratio of squared singular values: (Σσ²)² / Σσ⁴
        s_full = np.linalg.svd(h_flat - h_flat.mean(0, keepdims=True),
                               compute_uv=False)
        ev = s_full ** 2
        pr = float((ev.sum() ** 2) / (ev ** 2).sum())
        spectra[label] = s
        colours[label] = colour
        rows.append(dict(arm=label, path=path.name, U_b=u_b, U_t=u_t,
                         eff_rank_b=er_b, eff_rank_t=er_t, participation_ratio=pr))
        print(f"  {label:18s} U_b={u_b:.4f} U_t={u_t:.4f} "
              f"eff_rank_b={er_b:6.2f} eff_rank_t={er_t:6.2f} PR={pr:6.2f}")
        del model, h, h_flat

    # ---- write CSV ----
    csv_path = RESULTS / "latent_dim.csv"
    with csv_path.open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"wrote {csv_path}")

    # ---- plot: SV spectrum (log-y) | effective-rank bars ----
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    ax = axs[0]
    for r in rows:
        lab = r["arm"]
        s = spectra[lab]
        ax.plot(np.arange(1, len(s) + 1), s, color=colours[lab], lw=1.6,
                marker="o", markersize=2.5, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel("normalized singular value  σ_i / σ_0")
    ax.set_title("Encoder-latent h spectrum (mean-centred, all B×T positions)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")

    ax = axs[1]
    labs = [r["arm"] for r in rows]
    xs = np.arange(len(labs))
    width = 0.38
    cols = [colours[l] for l in labs]
    ax.bar(xs - width / 2, [r["eff_rank_b"] for r in rows], width=width,
           color=cols, alpha=0.9, edgecolor="black", linewidth=0.5,
           label="batch axis (U_b·H)")
    ax.bar(xs + width / 2, [r["eff_rank_t"] for r in rows], width=width,
           color=cols, alpha=0.55, edgecolor="black", linewidth=0.5, hatch="//",
           label="time axis (U_t·H)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("effective rank = U · H   (H = 384)")
    ax.set_title("Dimension-usage effective rank  (dim_usage, #2026-05-08 method)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "#318 — latent dimensionality of the frozen encoder representation h "
        "(d=384), one real-HF batch", fontsize=12)
    fig.tight_layout()
    out = PLOTS / "latent_dim.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
