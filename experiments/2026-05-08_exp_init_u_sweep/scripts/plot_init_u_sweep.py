#!/usr/bin/env python3
"""init-u-sweep plots.

Renders the categorical / scan plots for the init-u-sweep experiment:
- 8-init-scheme bar chart of U_b(o_lat) with ±1σ over 3 seeds
- W-sweep line plot (U_b vs W) with the (W+f+s)/H ceiling overlay
- freq-embedding-dim sweep (U_b vs freq_emb_dim) with the (W+f+s)/H ceiling
- B-sweep (U_b vs B) for encoder_init and isotropic_synthetic
- per-stage U_b inside GRUEncoder (default vs ortho_patch)
- 5-encoder eff_rank bar chart with ±1σ
- orthogonal-subspace per-stage breakdown for ResidualSiLUEncoder

All plots saved under experiments/2026-05-08_exp_init_u_sweep/plots/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
PLOTS = EXP / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

H = 384  # transformer hidden dim
DEFAULT_INPUT_CONCAT = 22  # W=16 + freq=3 + seas=3
BACKBONE_BETA_167K_UB = 0.0762  # from tau-sweep RESULTS reference

GRAY_REF = "#6b6b6b"
COLOR_DEFAULT = "#1f77b4"
COLOR_ORTHO = "#ff7f0e"


def plot_init_scheme_sweep() -> None:
    """8-init-scheme bar chart of U_b(o_lat) and U_b(f_lat), ±1σ."""
    df = pd.read_csv(RESULTS / "init_u_sweep.csv")
    df = df.sort_values("u_b_o_mean", ascending=False).reset_index(drop=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # U_b(o_lat) — encoder output, batch axis
    ax = axs[0]
    xs = np.arange(len(df))
    bars = ax.bar(xs, df["u_b_o_mean"], yerr=df["u_b_o_std"], capsize=3,
                  color=COLOR_DEFAULT, alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(df["name"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("U_b(o_lat)")
    ax.set_title("U_b(o_lat) — encoder output, batch axis (mean ± 1σ over 3 seeds)")
    ax.axhline(BACKBONE_BETA_167K_UB, color=GRAY_REF, linestyle="--", linewidth=1.0,
               label=f"backbone-β 167k = {BACKBONE_BETA_167K_UB:.4f}")
    ax.set_ylim(0, max(BACKBONE_BETA_167K_UB, df["u_b_o_mean"].max()) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # U_b(f_lat) — forecaster latent, batch axis
    ax = axs[1]
    bars = ax.bar(xs, df["u_b_f_mean"], yerr=df["u_b_f_std"], capsize=3,
                  color=COLOR_ORTHO, alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(df["name"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("U_b(f_lat)")
    ax.set_title("U_b(f_lat) — forecaster latent, batch axis (mean ± 1σ over 3 seeds)")
    ax.set_ylim(0, df["u_b_f_mean"].max() * 1.4)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "8-init-scheme sweep — sorted by U_b(o_lat) desc; defaults at index 3.",
        fontsize=12)
    fig.tight_layout()
    out = PLOTS / "init_scheme_sweep.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_w_sweep() -> None:
    """U_b vs W (sub-experiment A) and U_b vs freq_emb_dim (sub-experiment B).

    Overlay the (W+f+s)/H rank ceiling.
    """
    df = pd.read_csv(RESULTS / "init_u_w_sweep.csv")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # A: vary W
    a = df[df["experiment"] == "A_vary_W"].copy()
    a["ceiling"] = (a["W"] + a["freq_emb_dim"] + a["seasonality_emb_dim"]) / H
    ax = axs[0]
    ax.errorbar(a["W"], a["u_b_o_mean"], yerr=a["u_b_o_std"],
                marker="o", capsize=4, color=COLOR_DEFAULT,
                linewidth=2, markersize=7, label="measured U_b(o_lat)")
    ax.plot(a["W"], a["ceiling"], linestyle="--", color=GRAY_REF,
            linewidth=1.2, marker="s", markersize=5,
            label="rank ceiling (W+f+s)/H")
    ax.set_xscale("log", base=2)
    ax.set_xticks(a["W"])
    ax.set_xticklabels(a["W"].astype(int), rotation=0)
    ax.minorticks_off()
    ax.set_xlabel("patch width W")
    ax.set_ylabel("U_b(o_lat)")
    ax.set_title("A: vary W (freq=3, seas=3)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # B: vary freq_emb_dim
    b = df[df["experiment"] == "B_vary_freq_emb"].copy()
    b["ceiling"] = (b["W"] + b["freq_emb_dim"] + b["seasonality_emb_dim"]) / H
    ax = axs[1]
    ax.errorbar(b["freq_emb_dim"], b["u_b_o_mean"], yerr=b["u_b_o_std"],
                marker="o", capsize=4, color=COLOR_DEFAULT,
                linewidth=2, markersize=7, label="measured U_b(o_lat)")
    ax.plot(b["freq_emb_dim"], b["ceiling"], linestyle="--", color=GRAY_REF,
            linewidth=1.2, marker="s", markersize=5,
            label="rank ceiling (W+f+s)/H")
    ax.set_xscale("log", base=2)
    ax.set_xticks(b["freq_emb_dim"])
    ax.set_xticklabels(b["freq_emb_dim"].astype(int), rotation=0)
    ax.minorticks_off()
    ax.set_xlabel("freq_emb_dim")
    ax.set_ylabel("U_b(o_lat)")
    ax.set_title("B: vary freq_emb_dim (W=16, seas=0)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "W-sweep — measured U_b(o_lat) vs rank ceiling (W+f+s)/H",
        fontsize=12)
    fig.tight_layout()
    out = PLOTS / "w_sweep.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_b_sweep() -> None:
    """U_b vs B for encoder_init and isotropic_synthetic, per-slice vs global-pooled."""
    df = pd.read_csv(RESULTS / "init_u_b_sweep.csv")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    for ax, setup in zip(axs, ["encoder_init", "isotropic_synthetic"]):
        sub = df[df["setup"] == setup]
        per_slice = sub.groupby("B")["u_b_per_slice"].agg(["mean", "std"]).reset_index()
        global_pooled = sub.groupby("B")["u_b_global_pooled"].agg(["mean", "std"]).reset_index()

        ax.errorbar(per_slice["B"], per_slice["mean"], yerr=per_slice["std"],
                    marker="o", capsize=4, color=COLOR_DEFAULT,
                    linewidth=2, markersize=7, label="per-slice")
        ax.errorbar(global_pooled["B"], global_pooled["mean"], yerr=global_pooled["std"],
                    marker="s", capsize=4, color=COLOR_ORTHO,
                    linewidth=2, markersize=7, label="global-pooled")
        ax.set_xscale("log", base=2)
        ax.set_xticks(per_slice["B"])
        ax.set_xticklabels(per_slice["B"].astype(int), rotation=0)
        ax.minorticks_off()
        ax.set_xlabel("batch size B")
        ax.set_ylabel("U_b")
        ax.set_title(f"{setup}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        "B-sweep — per-slice vs global-pooled U_b across batch size",
        fontsize=12)
    fig.tight_layout()
    out = PLOTS / "b_sweep.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_per_stage_gru() -> None:
    """Per-stage U_b inside GRUEncoder, default vs ortho_patch."""
    df_def = pd.read_csv(RESULTS / "u_per_stage_default.csv")
    df_ort = pd.read_csv(RESULTS / "u_per_stage_ortho.csv")

    stages_order = ["input_concat", "gru_output", "proj_out", "skip_out",
                    "sum_pre_norm", "encoder_out"]

    def _agg(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("stage_name").agg(
            u_mean=("u_b_per_slice", "mean"),
            u_std=("u_b_per_slice", "std"),
            er_mean=("effective_rank", "mean"),
            er_std=("effective_rank", "std"),
            feature_dim=("feature_dim", "first"),
        ).reset_index()
        g["order"] = g["stage_name"].map({s: i for i, s in enumerate(stages_order)})
        return g.sort_values("order").reset_index(drop=True)

    g_def = _agg(df_def)
    g_ort = _agg(df_ort)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = np.arange(len(stages_order))
    width = 0.38

    ax.bar(xs - width / 2, g_def["er_mean"], width=width, yerr=g_def["er_std"],
           capsize=3, color=COLOR_DEFAULT, alpha=0.85, edgecolor="black",
           linewidth=0.5, label="default init")
    ax.bar(xs + width / 2, g_ort["er_mean"], width=width, yerr=g_ort["er_std"],
           capsize=3, color=COLOR_ORTHO, alpha=0.85, edgecolor="black",
           linewidth=0.5, label="ortho_patch (gain=√2, zero bias)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}\n(d={d})" for s, d in
                        zip(stages_order, g_def["feature_dim"])], fontsize=9)
    ax.set_ylabel("effective rank = U_b · feature_dim")
    ax.set_title(
        "Per-stage effective rank inside GRUEncoder — default vs ortho_patch (B=256, 3 seeds)",
        fontsize=11)
    ax.axhline(DEFAULT_INPUT_CONCAT, color=GRAY_REF, linestyle="--", linewidth=1.0,
               label=f"input-concat width = {DEFAULT_INPUT_CONCAT}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = PLOTS / "per_stage_gru.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_encoder_comparison() -> None:
    """5-encoder eff_rank bar chart, ±1σ over 3 seeds."""
    df = pd.read_csv(RESULTS / "init_u_per_encoder.csv")
    g = df.groupby("encoder_type").agg(
        u_mean=("u_b_per_slice", "mean"),
        u_std=("u_b_per_slice", "std"),
        er_mean=("effective_rank", "mean"),
        er_std=("effective_rank", "std"),
    ).reset_index().sort_values("er_mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(g))
    ax.bar(xs, g["er_mean"], yerr=g["er_std"], capsize=4,
           color=COLOR_DEFAULT, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(g["encoder_type"], fontsize=10)
    ax.set_ylabel("effective rank = U_b · feature_dim")
    ax.set_title("5-encoder comparison — encoder_out eff_rank (mean ± 1σ over 3 seeds)",
                 fontsize=11)
    ax.axhline(DEFAULT_INPUT_CONCAT, color=GRAY_REF, linestyle="--", linewidth=1.0,
               label=f"input-concat width = {DEFAULT_INPUT_CONCAT}")
    ax.set_ylim(14.5, 17.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = PLOTS / "encoder_comparison.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_orthogonal_subspaces() -> None:
    """Per-stage eff_rank inside ResidualSiLUEncoder for the 4 orth-subspace schemes."""
    df = pd.read_csv(RESULTS / "init_u_orthogonal_subspaces.csv")
    schemes_order = [
        "default",
        "ortho_all_three_subspaces",
        "ortho_skip_proj_orthogonal_images",
        "ortho_skip_proj_only_random_mlp",
    ]
    stages_order = [
        "input_concat", "proj_out", "mlp_pre_silu", "mlp_post_silu",
        "mlp_out", "h_after_residual", "skip_out", "pre_norm", "encoder_out",
    ]

    g = df.groupby(["init_scheme", "stage"]).agg(
        u_mean=("u_b_per_slice", "mean"),
        u_std=("u_b_per_slice", "std"),
        er_mean=("effective_rank", "mean"),
        er_std=("effective_rank", "std"),
        feature_dim=("feature_dim", "first"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(13, 5.5))
    xs = np.arange(len(stages_order))
    width = 0.20
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    offsets = np.linspace(-1.5, 1.5, 4) * width
    for off, scheme, color in zip(offsets, schemes_order, palette):
        sub = g[g["init_scheme"] == scheme].set_index("stage").reindex(stages_order)
        ax.bar(xs + off, sub["er_mean"], width=width, yerr=sub["er_std"],
               capsize=2, color=color, alpha=0.85, edgecolor="black",
               linewidth=0.4, label=scheme)
    feat_dims = (g[g["init_scheme"] == "default"].set_index("stage")
                 .reindex(stages_order)["feature_dim"])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}\n(d={int(d)})" for s, d in zip(stages_order, feat_dims)],
                       fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel("effective rank = U_b · feature_dim")
    ax.set_title(
        "Orthogonal-subspace schemes for ResidualSiLUEncoder — per-stage eff_rank "
        "(mean ± 1σ over 3 seeds)", fontsize=11)
    ax.axhline(DEFAULT_INPUT_CONCAT, color=GRAY_REF, linestyle="--", linewidth=1.0,
               label=f"input-concat width = {DEFAULT_INPUT_CONCAT}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out = PLOTS / "orthogonal_subspaces.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    plot_init_scheme_sweep()
    plot_w_sweep()
    plot_b_sweep()
    plot_per_stage_gru()
    plot_encoder_comparison()
    plot_orthogonal_subspaces()


if __name__ == "__main__":
    main()
