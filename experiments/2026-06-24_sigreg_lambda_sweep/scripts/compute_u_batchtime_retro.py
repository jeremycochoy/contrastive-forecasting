#!/usr/bin/env python3
"""Retroactive cross-(batch × time) dim-usage on existing FINAL.pth backbones.

#363 follow-up: the training CSV now records u_batchtime / u_batchtime_e
prospectively, but the four already-finished arms of the SIGReg λ-sweep
and the two prior anchors have no such column. This script reloads each
FINAL backbone, runs one fixed batch from the training distribution
(gift-pretrain-full-4096 / small_v1, deterministic via SEED below), and
emits u_batchtime + u_batchtime_e per checkpoint.

Output: ``results/u_batchtime_retro.csv`` with
    arm, recipe, backbone_ckpt, u_batchtime, u_batchtime_e
one row per backbone.

Usage:
    PYTHONPATH=. python3 \\
      experiments/2026-06-24_sigreg_lambda_sweep/scripts/compute_u_batchtime_retro.py
"""

from __future__ import annotations

import csv
import os
import sys

import torch

import src.dataloader as dataloader
from src.dataloader import create_hf_dataloader
from src.models import ConfigurableModel
from src.metrics import u_batchtime


# -- Determinism: same seed as the SIGReg λ-sweep training runs so the
# held-out batch is reproducible across machines / Python invocations.
SEED = 20260520

# -- Held-out batch: matches the #363 training config (B=512 at the same
# T_raw and HF source). gift-pretrain-full-4096 / small_v1.
BATCH_SIZE = 512
T_RAW = 4096
HF_REPO = "jeremycochoy/gift-pretrain-full-4096"
HF_PATH = "small_v1"

# -- Backbone arch from the launch script (train_backbone_sigreg.sh):
# d_model=384, n_heads=6, decoder=6L, encoder=3L, GRU patch-embed, W=16
# (the train.py MODEL_CONFIG default; no --patch-size flag). freq_emb_dim
# / seasonality_emb_dim / patch_stats_kind are auto-detected from the
# state_dict shapes — same logic as
# experiments/2026-05-05_exp_qhead_improvements/scripts/eval_backbone_metrics.py.
BACKBONE_BASE = dict(
    C=1, H=384, W=16,
    encoder_type="gru", num_layers=6, nhead=6,
    ffn_mult=4.0, activation="gelu",
    depthwise_conv=3, dropout=0.1,
    num_encoder_layers=3,
    encoder_dropkey=0.70,
    encoder_dropkey_share_heads=True,
    encoder_dropkey_share_layers=True,
    rev_norm_kind="ewma", rev_norm_span=128,
    qk_norm=True, attn_out_norm=True,
    # ema_embedding / ema_encoder / cpc_infonce are all left at their
    # defaults (False / 0). The trained checkpoints DID enable these,
    # but the corresponding state_dict entries (teacher_*, cpc_w1.*) are
    # auxiliary — they don't affect a student forward. Stripping them
    # below and constructing a non-EMA / non-CPC model lets the strict
    # load succeed, and the e_t / h_t we read are the student values
    # (same path the training loop's u_*_e diagnostic reads).
)

# Forward chunk so 512 samples fit a 24 GB card.
FORWARD_CHUNK = 32


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP_DIR = os.path.join(
    REPO_ROOT, "experiments", "2026-06-24_sigreg_lambda_sweep")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
CSV_OUT = os.path.join(RESULTS_DIR, "u_batchtime_retro.csv")


def _pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free = []
    for i in range(torch.cuda.device_count()):
        free_b, _ = torch.cuda.mem_get_info(i)
        free.append((free_b // (1024 * 1024), i))
    free.sort(reverse=True)
    free_mib, idx = free[0]
    if free_mib < 6000:
        return "cpu"
    return f"cuda:{idx}"


def _resolve_checkpoint(rel: str) -> str | None:
    """Return the absolute path to ``rel`` if it exists.

    Searches three roots:
      1. The current worktree (REPO_ROOT).
      2. ``/tmp/contrastive-forecasting-<issue>/`` worktrees holding the
         main checkout where FINAL.pth files live for arms 1-4 and the
         two prior anchors. Sweeps over a handful of known issue numbers.
      3. ``/home/jupyter/contrastive-forecasting/`` (elisa main).

    Returns the first existing path; ``None`` if not found.
    """
    candidates = [os.path.join(REPO_ROOT, rel)]
    for issue in ("363", "359", "357", "355", "353"):
        candidates.append(
            os.path.join(f"/tmp/contrastive-forecasting-{issue}", rel))
    candidates.append(
        os.path.join("/home/jupyter/contrastive-forecasting", rel))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# -- Checkpoints to evaluate. The four arms already-finished plus the
# two prior anchors. arms 4 and 6 (emb10000_enc10 / emb10_enc10) are
# still in flight on the worktree as of 2026-06-27 — those are
# DELIBERATELY OMITTED here; the next iteration appends them.
CHECKPOINTS = [
    dict(
        arm="emb100_enc01",
        recipe="λ_e=10.0, λ_h=0.1",
        rel="experiments/2026-06-24_sigreg_lambda_sweep/runs/"
            "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_"
            "emb100_enc01_FINAL.pth",
    ),
    dict(
        arm="emb100_enc10",
        recipe="λ_e=10.0, λ_h=1.0",
        rel="experiments/2026-06-24_sigreg_lambda_sweep/runs/"
            "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_"
            "emb100_enc10_FINAL.pth",
    ),
    dict(
        arm="emb100_enc100",
        recipe="λ_e=10.0, λ_h=10.0",
        rel="experiments/2026-06-24_sigreg_lambda_sweep/runs/"
            "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_"
            "emb100_enc100_FINAL.pth",
    ),
    dict(
        arm="emb1000_enc01",
        recipe="λ_e=100.0, λ_h=0.1",
        rel="experiments/2026-06-24_sigreg_lambda_sweep/runs/"
            "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_"
            "emb1000_enc01_FINAL.pth",
    ),
    # Prior anchors. Same recipe family; bundled so the new dim-usage
    # column lines up against the report's anchor rows.
    dict(
        arm="anchor_emb01",
        recipe="λ_e=0.1, λ_h=0.1 (#355 anchor)",
        rel="reports/2026-06-20_lejepa_sigreg/runs/"
            "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_"
            "FINAL.pth",
    ),
    dict(
        arm="anchor_emb10",
        recipe="λ_e=1.0, λ_h=1.0 (#359 anchor)",
        rel="reports/2026-06-22_lejepa_sigreg_emb10/runs/"
            "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_"
            "emb10_FINAL.pth",
    ),
]


def load_held_out_batch(device: str):
    """Return ``(x, freq_ids, seasonality_ids)`` for one fixed batch.

    Stream the first ``BATCH_SIZE`` rows of the same training source as
    the SIGReg λ-sweep (gift-pretrain-full-4096 / small_v1). The
    ``torch.manual_seed`` pin makes any internal-RNG choices reproducible;
    streaming order is shard-deterministic on its own.
    """
    dataloader.T_RAW = T_RAW
    torch.manual_seed(SEED)
    loader = create_hf_dataloader(
        repo_id=HF_REPO, batch_size=BATCH_SIZE, C=1,
        path_in_repo=HF_PATH, skip_rows=0,
    )
    batch = next(iter(loader))
    if isinstance(batch, tuple):
        if len(batch) == 3:
            x, freq_ids, seas_ids = batch
        else:
            x = batch[0]
            freq_ids = None
            seas_ids = None
    else:
        x = batch
        freq_ids = None
        seas_ids = None
    assert x.shape == (BATCH_SIZE, T_RAW, 1), x.shape
    return (
        x.to(device),
        freq_ids.to(device) if freq_ids is not None else None,
        seas_ids.to(device) if seas_ids is not None else None,
    )


def build_backbone(sd: dict) -> ConfigurableModel:
    """Construct a ``ConfigurableModel`` matching this checkpoint.

    Auto-detects ``freq_emb_dim``, ``seasonality_emb_dim``, and
    ``patch_stats_kind`` from the GRU encoder's ``skip.weight`` shape —
    same logic as ``eval_backbone_metrics.py``.
    """
    cfg = dict(BACKBONE_BASE)
    fw = sd.get("freq_embedding.embedding.weight")
    cfg["freq_emb_dim"] = int(fw.shape[1]) if fw is not None else 0
    sw = sd.get("seasonality_embedding.embedding.weight")
    cfg["seasonality_emb_dim"] = int(sw.shape[1]) if sw is not None else 0
    cfg["learnable_tau"] = "log_inv_tau" in sd
    ref = sd.get("encoder.skip.weight")
    if ref is None:
        cfg["patch_stats_kind"] = "none"
    else:
        extra = (ref.shape[1] - cfg["W"]
                 - cfg["freq_emb_dim"] - cfg["seasonality_emb_dim"])
        if extra == 0:
            cfg["patch_stats_kind"] = "none"
        elif extra == 2:
            cfg["patch_stats_kind"] = "diff"
        else:
            raise ValueError(
                f"Unexpected encoder.skip in_features={ref.shape[1]}: "
                f"extra={extra}")
    return ConfigurableModel(**cfg)


def _resolve_default_ids(backbone, x, freq_ids, seas_ids):
    """Fall back to class 0 (unknown) when an embedding is configured
    but the corresponding id tensor is missing — same convention as
    ``extract_encoder_latents``."""
    B = x.shape[0]
    device = x.device
    if (getattr(backbone, 'freq_embedding', None) is not None
            and freq_ids is None):
        d = int(getattr(backbone, '_eval_freq_id', 0))
        freq_ids = torch.full((B,), d, dtype=torch.long, device=device)
    if (getattr(backbone, 'seasonality_embedding', None) is not None
            and seas_ids is None):
        d = int(getattr(backbone, '_eval_seasonality_id', 0))
        seas_ids = torch.full((B,), d, dtype=torch.long, device=device)
    return freq_ids, seas_ids


@torch.no_grad()
def extract_e_and_h(backbone, x, freq_ids, seas_ids):
    """Run the backbone forward and return ``(e_lat, h_lat)`` in
    ``(B, T, C, H)`` layout. ``e_lat`` is the patch embedding (#355's
    SIGReg target on the encoder input); ``h_lat`` is the encoder
    output (the contrastive target ``o_lat``).

    Chunks over batch so 512 samples fit a 24 GB GPU.
    """
    backbone.eval()
    e_parts, h_parts = [], []
    for s in range(0, x.shape[0], FORWARD_CHUNK):
        e = s + FORWARD_CHUNK
        x_c = x[s:e]
        f_c = freq_ids[s:e] if freq_ids is not None else None
        s_c = seas_ids[s:e] if seas_ids is not None else None

        f_c, s_c = _resolve_default_ids(backbone, x_c, f_c, s_c)

        x_norm = (backbone.rev_norm(x_c, mode='norm')
                  if backbone.rev_norm is not None else x_c)
        xr = backbone.prepare_encoder_input(
            x_norm, freq_ids=f_c, seasonality_ids=s_c)
        out = backbone.transformer(xr, return_embed=True)
        # transformer.forward with return_embed=True returns
        # (f_out, x_original, embed). x_original is the encoder output
        # h_t in (B*C, T, H) layout; embed is e_t in (B, T, C, H).
        assert len(out) == 3, (
            f"expected (f, h, e) from return_embed=True; got {len(out)}")
        _, h_flat, e_lat = out
        Bc, T, C, H = e_lat.shape
        h_lat = h_flat.reshape(Bc, C, T, H).permute(0, 2, 1, 3).contiguous()
        e_parts.append(e_lat.float().cpu())
        h_parts.append(h_lat.float().cpu())
    return torch.cat(e_parts, dim=0), torch.cat(h_parts, dim=0)


def eval_one(ckpt: dict, x_dev, freq_ids, seas_ids, device: str) -> dict:
    path = _resolve_checkpoint(ckpt["rel"])
    if path is None:
        print(f"  [retro] {ckpt['arm']}: MISSING -> {ckpt['rel']}",
              file=sys.stderr)
        return dict(
            arm=ckpt["arm"], recipe=ckpt["recipe"],
            backbone_ckpt=ckpt["rel"],
            u_batchtime="", u_batchtime_e="",
        )

    sd = torch.load(path, map_location="cpu", weights_only=True)
    bb = build_backbone(sd)
    # Drop training-only state (CPC InfoNCE W_1, EMA teacher copies) so a
    # strict load passes. Same prune as eval_backbone_metrics.py.
    sd_strict = {
        k: v for k, v in sd.items()
        if not k.startswith("cpc_w1") and not k.startswith("teacher_")}
    bb.load_state_dict(sd_strict)
    bb.to(device)

    e_lat, h_lat = extract_e_and_h(bb, x_dev, freq_ids, seas_ids)
    u_bt_h = u_batchtime(h_lat).item()
    u_bt_e = u_batchtime(e_lat).item()

    del bb
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return dict(
        arm=ckpt["arm"], recipe=ckpt["recipe"],
        backbone_ckpt=os.path.relpath(path, REPO_ROOT),
        u_batchtime=f"{u_bt_h:.6f}",
        u_batchtime_e=f"{u_bt_e:.6f}",
    )


def main() -> None:
    device = _pick_device()
    print(f"  [retro] device={device}")

    x_dev, freq_ids, seas_ids = load_held_out_batch(device)
    print(f"  [retro] batch shape={tuple(x_dev.shape)}  "
          f"freq_ids={tuple(freq_ids.shape) if freq_ids is not None else None}  "
          f"seas_ids={tuple(seas_ids.shape) if seas_ids is not None else None}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cols = ["arm", "recipe", "backbone_ckpt",
            "u_batchtime", "u_batchtime_e"]
    rows: list[dict] = []
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for ckpt in CHECKPOINTS:
            row = eval_one(ckpt, x_dev, freq_ids, seas_ids, device)
            rows.append(row)
            w.writerow(row)
            f.flush()
            print(f"  [retro] {row['arm']:20s}  "
                  f"u_batchtime(h_t)={row['u_batchtime']}  "
                  f"u_batchtime_e(e_t)={row['u_batchtime_e']}")
    print(f"  [retro] wrote {CSV_OUT}")


if __name__ == "__main__":
    main()
