"""Latent movement per arm — displacement of the encoder-output and
patch-embedding latents between adjacent training checkpoints (#379).

For each arm and each adjacent checkpoint pair ``(step_i, step_j)``:

    movement_h = mean over (b, t, c) of  1 - cos(h_t(model_j), h_t(model_i))
    movement_e = mean over (b, t, c) of  1 - cos(e_t(model_j), e_t(model_i))

Both are computed on a fixed held-out batch (torch seed 20260722,
B=64/T=4096/C=1 by default), so movements are directly comparable
across arms and pairs. x-axis is the later checkpoint's step (log
scale) so the early-training dynamics stay legible.

CLI / env knobs (all optional):
    --runs-dir DIR       runs/ dir with the ``*_<k>k.pth`` files
                         (default: env RUNS_DIR, else EXP/runs)
    --arms a,b,c         restrict to these arm keys
    --batch-size B       held-out-batch size (default 64; smoke: <=8)
    --device DEV         torch device (default cuda if available)
    --skip-rows N        HF held-out offset (default 40_000_000, past
                         the ~12.8M rows the 200k-step training touches)
    --synthetic          bypass HF, use a deterministic ARMA batch instead
    --batch-cache PATH   cache path for the fixed batch (default:
                         HERE/_latent_movement_batch.pt)
    --out PATH           output png (default: HERE/latent_movement_per_arm.png)
    --dump-csv PATH      also write per-pair drift values as CSV, so downstream
                         figures do not need a GPU to re-derive them
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Support execution from anywhere: put repo root on sys.path so
# ``from src.eval_latent_movement import ...`` resolves whether the
# caller is inside the worktree or a sibling checkout.
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.eval_latent_movement import (           # noqa: E402
    compute_latents, load_backbone, mean_one_minus_cos, small_backbone_kwargs,
)

EXP = ROOT / "experiments" / "2026-07-21_split_pred_rep_small"

# Colours mirror _make_cos_error.py so viewers can cross-reference.
RUNS = [
    ("arm 1  (L_pred + L_rep)",
     "bb_small_arm1_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#2a78d6"),
    ("arm 3  (L_pred_moco + L_rep)",
     "bb_small_arm3_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#eb6834"),
    ("arm 4  (pooled + MoCo)",
     "bb_small_arm4_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#008300"),
    ("arm 5  (L_align + L_rep)",
     "bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#8b1e8b"),
    ("arm 6 v2  (L_align + L_rep_moco)",
     "bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#b8860b"),
    ("bimoco  (L_pred_moco + L_rep_moco)",
     "bb_small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#00a3a3"),
    ("arm 1 tr1  (L_pred + L_rep, all τ=1.0)",
     "bb_small_arm1_tr1_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#7fb0e8"),
    ("arm 3 tr1  (L_pred_moco + L_rep, all τ=1.0)",
     "bb_small_arm3_tr1_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#f4a680"),
    ("arm 4 tr1  (pooled + MoCo, all τ=1.0)",
     "bb_small_arm4_tr1_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#7fc17f"),
    ("arm 5 tr1  (L_align + L_rep, all τ=1.0)",
     "bb_small_arm5_tr1_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#c98cc9"),
    ("arm 6 v2 tr1  (L_align + L_rep_moco, all τ=1.0)",
     "bb_small_arm6_v2_tr1_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#dcbb60"),
    ("bimoco tr1  (L_pred_moco + L_rep_moco, all τ=1.0)",
     "bb_small_bimoco_tr1_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#66c4c4"),
    # #379 no-sigreg-embedding reruns — palest variant of the base colour.
    ("arm 1 nse  (L_pred + L_rep, sigreg_e=0)",
     "bb_small_arm1_nse_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#a6c8ee"),
    ("arm 3 nse  (L_pred_moco + L_rep, sigreg_e=0)",
     "bb_small_arm3_nse_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#f5b39a"),
    ("arm 4 nse  (pooled + MoCo, sigreg_e=0)",
     "bb_small_arm4_nse_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#7fc17f"),
    ("arm 5 nse  (L_align + L_rep, sigreg_e=0)",
     "bb_small_arm5_nse_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#c58fc5"),
    ("arm 6 v2 nse  (L_align + L_rep_moco, sigreg_e=0)",
     "bb_small_arm6_v2_nse_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#dcc385"),
    ("bimoco nse  (L_pred_moco + L_rep_moco, sigreg_e=0)",
     "bb_small_bimoco_nse_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#7fd1d1"),
    # #379 no-CPC reruns — same base colour, dashed via STYLE below.
    ("arm 1 ncpc  (L_pred + L_rep, cpc=0)",
     "bb_small_arm1_ncpc_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#2a78d6"),
    ("arm 3 ncpc  (L_pred_moco + L_rep, cpc=0)",
     "bb_small_arm3_ncpc_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#eb6834"),
    ("arm 4 ncpc  (pooled + MoCo, cpc=0)",
     "bb_small_arm4_ncpc_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#008300"),
    ("arm 5 ncpc  (L_align + L_rep, cpc=0)",
     "bb_small_arm5_ncpc_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#8b1e8b"),
    ("arm 6 v2 ncpc  (L_align + L_rep_moco, cpc=0)",
     "bb_small_arm6_v2_ncpc_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#b8860b"),
    ("bimoco ncpc  (L_pred_moco + L_rep_moco, cpc=0)",
     "bb_small_bimoco_ncpc_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#00a3a3"),
    # combab: all τ=1.0 + cpc=0 + nse (arm1/3/4 only, per nse stat test)
    ("arm 1 combab  (τ=1.0 + cpc=0 + sigreg_e=0)",
     "bb_small_arm1_combab_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#2a78d6"),
    ("arm 3 combab  (τ=1.0 + cpc=0 + sigreg_e=0)",
     "bb_small_arm3_combab_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#eb6834"),
    ("arm 4 combab  (τ=1.0 + cpc=0 + sigreg_e=0)",
     "bb_small_arm4_combab_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#008300"),
    ("arm 5 combab  (τ_rep=1.0 + cpc=0)",
     "bb_small_arm5_combab_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#8b1e8b"),
    ("arm 6 v2 combab  (τ_rep=1.0 + cpc=0)",
     "bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#b8860b"),
    ("bimoco combab  (all τ=1.0 + cpc=0)",
     "bb_small_bimoco_combab_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#00a3a3"),
]
# Short slug per arm — used to select via --arms. Parallel to RUNS by
# index; STYLE below is keyed on slug (dashed for ncpc, solid default).
SLUGS = ["arm1", "arm3", "arm4", "arm5", "arm6_v2", "bimoco",
         "arm1_tr1", "arm3_tr1", "arm4_tr1", "arm5_tr1", "arm6_v2_tr1", "bimoco_tr1",
         "arm1_nse", "arm3_nse", "arm4_nse", "arm5_nse", "arm6_v2_nse", "bimoco_nse",
         "arm1_ncpc", "arm3_ncpc", "arm4_ncpc", "arm5_ncpc", "arm6_v2_ncpc", "bimoco_ncpc",
         "arm1_combab", "arm3_combab", "arm4_combab", "arm5_combab", "arm6_v2_combab", "bimoco_combab"]
# Per-slug linestyle override for the h_t panel. ncpc → dashed on top of
# the shared base colour; everything else stays solid. The e_t panel is
# already dashed as an axis-level convention, so ncpc's dashed override
# only applies to ax_h.
STYLE = {
    "arm1_ncpc": "--",
    "arm3_ncpc": "--",
    "arm4_ncpc": "--",
    "arm5_ncpc": "--",
    "arm6_v2_ncpc": "--",
    "bimoco_ncpc": "--",
}

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

FIXED_SEED = 20260722


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR"))
    p.add_argument("--arms", default=None,
                   help="Comma-separated slugs to plot (default: all).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--t-raw", type=int, default=4096)
    p.add_argument("--c", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-rows", type=int, default=40_000_000)
    p.add_argument("--synthetic", action="store_true",
                   help="Use a deterministic ARMA batch instead of HF stream.")
    p.add_argument("--batch-cache", default=str(HERE / "_latent_movement_batch.pt"))
    p.add_argument("--out", default=str(HERE / "latent_movement_per_arm.png"))
    p.add_argument("--dump-csv", default=None,
                   help="write per-checkpoint-pair drift values to this CSV")
    return p.parse_args()


def list_periodic_checkpoints(runs_dir: Path, name: str) -> list[tuple[int, Path]]:
    """Return ``[(step, path), ...]`` sorted by step for ``<name>_<k>k.pth``.

    Optimizer companions are filtered out; ``_final``, ``_best_*`` and the
    ``_FINAL`` sentinel are skipped so only the periodic snapshots that
    correspond to training milestones appear.
    """
    out: list[tuple[int, Path]] = []
    for p in runs_dir.glob(f"{name}_*k.pth"):
        stem = p.stem
        if stem.endswith("_optimizer"):
            continue
        # Match ``..._<digits>k`` — anything else (best_gap, best_loss, final,
        # FINAL, tr1 milestones under a different base name) is not a periodic
        # snapshot of THIS run.
        tail = stem.rsplit("_", 1)[-1]
        if not tail.endswith("k") or not tail[:-1].isdigit():
            continue
        step_k = int(tail[:-1])
        out.append((step_k * 1000, p))
    out.sort(key=lambda kv: kv[0])
    return out


def make_fixed_batch(args: argparse.Namespace) -> torch.Tensor:
    """Return a deterministic ``[B, T, C]`` batch, cached on disk."""
    cache = Path(args.batch_cache)
    if cache.exists():
        x = torch.load(cache, map_location="cpu", weights_only=False)
        if x.shape == (args.batch_size, args.t_raw, args.c):
            return x
        # Shape drift → regenerate rather than silently use the wrong batch.
        cache.unlink()
    torch.manual_seed(FIXED_SEED)
    if args.synthetic or not os.environ.get("HF_TOKEN"):
        # Deterministic ARMA fallback: fast, no network, matches training's
        # per-step data shape. Used for the smoke test and when HF is not
        # authenticated.
        from src.arma import generate_arma_batch
        x, _ = generate_arma_batch(
            batch_size=args.batch_size, T_raw=args.t_raw,
            C=args.c, seed=FIXED_SEED, dimension=4)
    else:
        # Held-out HF slice: skip past the rows any training-step index at
        # 200k*B=64*C=1 could have touched (12.8M rows).
        from src.dataloader import HFStreamingLoader
        loader = HFStreamingLoader(
            repo_id="jeremycochoy/gift-pretrain-full-4096",
            path_in_repo="small_v1",
            batch_size=args.batch_size, C=args.c,
            skip_rows=args.skip_rows, prefetch=1)
        it = iter(loader)
        x = next(it)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x, cache)
    return x


def compute_arm_movements(runs_dir: Path, name: str, batch: torch.Tensor,
                          device: str) -> list[tuple[int, float, float]]:
    """Return ``[(later_step, movement_h, movement_e), ...]``.

    Loads adjacent checkpoints one pair at a time so only two backbones
    live on the GPU concurrently — keeps memory well below the trainer's
    working set even on a shared card.
    """
    cks = list_periodic_checkpoints(runs_dir, name)
    if len(cks) < 2:
        return []
    kwargs = small_backbone_kwargs(C=batch.shape[-1])
    x = batch.to(device)
    # Every arm was trained with --freq-emb-dim 3 --seasonality-emb-dim 3,
    # so both embeddings are configured on the reloaded model. Pass a frozen
    # ``0`` (the "unknown" bucket in FREQ_NAMES/SEASONALITY_NAMES) for every
    # sample so consecutive checkpoints see the same conditioning input.
    B = x.shape[0]
    freq_ids = torch.zeros(B, dtype=torch.long, device=device)
    seasonality_ids = torch.zeros(B, dtype=torch.long, device=device)
    prev_step, prev_path = cks[0]
    prev_model = load_backbone(str(prev_path), kwargs, device)
    prev_h, prev_e = compute_latents(prev_model, x, freq_ids, seasonality_ids)
    del prev_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    out: list[tuple[int, float, float]] = []
    for cur_step, cur_path in cks[1:]:
        cur_model = load_backbone(str(cur_path), kwargs, device)
        cur_h, cur_e = compute_latents(cur_model, x, freq_ids, seasonality_ids)
        out.append((cur_step,
                    mean_one_minus_cos(prev_h, cur_h),
                    mean_one_minus_cos(prev_e, cur_e)))
        del cur_model
        prev_h, prev_e, prev_step = cur_h, cur_e, cur_step
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir) if args.runs_dir else EXP / "runs"
    if not runs_dir.exists():
        raise SystemExit(f"runs dir not found: {runs_dir}")

    selected = set(args.arms.split(",")) if args.arms else set(SLUGS)
    plan = [(lbl, name, colour, slug) for (lbl, name, colour), slug
            in zip(RUNS, SLUGS) if slug in selected]

    batch = make_fixed_batch(args)
    print(f"fixed batch: shape={tuple(batch.shape)} device=cpu → {args.device}")

    # Compute all curves first so we can compute shared y-limits per column.
    per_arm: dict[str, tuple[list, list, list, str, str]] = {}
    for label, name, colour, slug in plan:
        points = compute_arm_movements(runs_dir, name, batch, args.device)
        if not points:
            print(f"  skip {label}: fewer than 2 periodic checkpoints under {name}")
            continue
        steps = [p[0] for p in points]
        mv_h  = [p[1] for p in points]
        mv_e  = [p[2] for p in points]
        per_arm[slug] = (steps, mv_h, mv_e, colour, label)
        print(f"  {label}: {len(points)} pairs → h_last={mv_h[-1]:.4f}  e_last={mv_e[-1]:.4f}")

    if args.dump_csv:
        import csv as _csv
        with open(args.dump_csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["arm_slug", "label", "step_later", "drift_h", "drift_e"])
            for slug, (steps, mv_h, mv_e, _c, label) in per_arm.items():
                for st, h, e in zip(steps, mv_h, mv_e):
                    w.writerow([slug, label, st, f"{h:.6f}", f"{e:.6f}"])
        print(f"wrote {args.dump_csv}")

    y_h_max = max((max(mv_h) for _, mv_h, _, _, _ in per_arm.values()), default=1.0)
    y_h_min = min((min(mv_h) for _, mv_h, _, _, _ in per_arm.values()), default=0.0)
    y_e_max = max((max(mv_e) for _, _, mv_e, _, _ in per_arm.values()), default=1.0)
    y_e_min = min((min(mv_e) for _, _, mv_e, _, _ in per_arm.values()), default=0.0)

    # 4 rows (variants) × 2 cols (h_t, e_t); shared y per column.
    PANELS = [
        ("base  (all τ=0.10, sigreg_e=1.0, cpc=1.0)", ""),
        ("tr1  (all τ=1.00)", "_tr1"),
        ("nse  (sigreg_e=0)", "_nse"),
        ("ncpc  (cpc=0)", "_ncpc"),
        ("combab  (all τ=1.0 + cpc=0 + cond. nse)", "_combab"),
    ]
    fig, axes = plt.subplots(len(PANELS), 2, figsize=(14, 3.4 * len(PANELS)),
                             sharex=True)
    for row_idx, (panel_title, suffix) in enumerate(PANELS):
        ax_h, ax_e = axes[row_idx]
        for slug, (steps, mv_h, mv_e, colour, label) in per_arm.items():
            if suffix == "":
                if any(s in slug for s in ("_tr1", "_nse", "_ncpc", "_combab")):
                    continue
            else:
                if not slug.endswith(suffix):
                    continue
            ax_h.plot(steps, mv_h, marker="o", markersize=6, color=colour,
                      lw=1.4, label=label)
            ax_e.plot(steps, mv_e, marker="o", markersize=6, color=colour,
                      lw=1.4, linestyle="--", label=label)
        for ax in (ax_h, ax_e):
            ax.set_xscale("log")
            ax.grid(True, color=GRID, alpha=0.6, which="both")
            ax.legend(loc="upper right", fontsize=7, frameon=False)
        ax_h.set_ylim(y_h_min, y_h_max)
        ax_e.set_ylim(y_e_min, y_e_max)
        ax_h.set_ylabel(f"{panel_title}\n1 − cos(h_prev, h_next)", fontsize=9)
        ax_e.set_ylabel("1 − cos(e_prev, e_next)", fontsize=9)
    axes[0, 0].set_title("h_t  (encoder out)", fontsize=10)
    axes[0, 1].set_title("e_t  (patch embedding)", fontsize=10)
    axes[-1, 0].set_xlabel("training step of the later checkpoint (log)")
    axes[-1, 1].set_xlabel("training step of the later checkpoint (log)")
    fig.suptitle("Latent movement between adjacent checkpoints  "
                 "(fixed held-out batch, shared y per column)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
