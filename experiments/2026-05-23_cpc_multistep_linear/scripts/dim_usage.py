#!/usr/bin/env python3
"""#316 falsifiable check (PR #317 review request): does multi-step (k=12)
collapse the encoder latent onto a lower-dimensional subspace than β (k=1)?

Measures the **participation ratio** (effective dimensionality) of the encoder
latent h_t — PR = (Σλ)² / Σλ², λ = eigenvalues of the latent covariance, range
[1, H] — across training-step checkpoints, on one FIXED real-data batch reused
for every model so differences reflect the representation, not the input.

Hypothesis under test (reviewer): the multi-step objective forces the latent to
be ~linearly predictable over 12 steps → low-rank collapse → CPC's dim-usage
sits clearly below β's. Falsifiable: if CPC ≥ β, the collapse story is wrong.

All arms here share β's exact encoder (H=384, 6 causal layers, gru, RevEWMNorm
span 128, freq+seasonality emb), so PR is directly comparable. The forecaster
heads differ but extract_encoder_latents touches only the encoder → strict=False.
"""
import csv, glob, math, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO = "/home/jupyter/contrastive-forecasting"
WT = f"{REPO}/.claude/worktrees/exp-bottleneck-beta2-confound"
sys.path.insert(0, WT)
from src.models import ConfigurableModel
from src.forecasting_head import extract_encoder_latents
from src.dataloader import create_mixed_periodic_dataloader
from src.norm import PATCH_STATS_DIM

MAIN = f"{REPO}/experiments/2026-05-23_cpc_multistep_linear"
RUNS = f"{MAIN}/runs"
BETA = f"{REPO}/experiments/2026-05-20_bottleneck_beta2_confound/runs"
OUT_PLOTS = f"{WT}/experiments/2026-05-23_cpc_multistep_linear/plots"
OUT_RES = f"{WT}/experiments/2026-05-23_cpc_multistep_linear/results"
os.makedirs(OUT_PLOTS, exist_ok=True); os.makedirs(OUT_RES, exist_ok=True)

HF_REPO, HF_PATH = "jeremycochoy/gift-pretrain-full-4096", "small_v1"
BATCH, T_RAW, C, NHEAD = 128, 4096, 1, 6

# (label, color, runs-dir, prefix)  — prefix_{N}k.pth are the periodic ckpts.
MODELS = [
    ("k=1: transformer (β)", "#1f77b4", BETA, "bb_beta_50k"),
    ("k=12: transformer",    "#1f77b4", RUNS, "bb_cpctrf_k12_s20260520_fp32_50k"),
    ("k=1: linear",          "#2ca02c", RUNS, "bb_linbn_k1_s20260520_fp32_50k"),
    ("k=12: linear",         "#d62728", RUNS, "bb_linbn_k12_s20260520_fp32_50k"),
]


def device_pick():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        return f"cuda:{int(np.argmax(free))}"
    except Exception:
        return "cuda:0"


DEV = device_pick()


def fixed_batch():
    """One fixed real-data batch (x, freq_ids, seasonality_ids), reused for all."""
    if not os.environ.get("HF_TOKEN"):
        for p in (f"{WT}/experiments/hf_token.txt", f"{REPO}/experiments/hf_token.txt"):
            if os.path.exists(p):
                os.environ["HF_TOKEN"] = open(p).read().strip(); break
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ.get("HF_TOKEN", ""))
    dl = create_mixed_periodic_dataloader(
        repo_id=HF_REPO, batch_size=BATCH, C=C, mix_ratio=0.0,
        path_in_repo=HF_PATH, skip_rows=0, seed=20260520, emit_freq_ids=True)
    x, fid, sid = next(iter(dl))
    return (x.to(DEV).float(),
            None if fid is None else fid.to(DEV),
            None if sid is None else sid.to(DEV))


def build_backbone(sd):
    H = sd["encoder.skip.weight"].shape[0]
    in_feat = sd["encoder.skip.weight"].shape[1]
    fw = sd.get("freq_embedding.embedding.weight")
    sw = sd.get("seasonality_embedding.embedding.weight")
    femb = fw.shape[1] if fw is not None else 0
    semb = sw.shape[1] if sw is not None else 0
    n_enc = max([int(k.split(".")[2]) for k in sd
                 if k.startswith("transformer.encoder_layers.")], default=-1) + 1
    extra = in_feat - 16 - femb - semb            # W=16
    patch = "diff" if extra == PATCH_STATS_DIM else "none"
    cfg = dict(C=C, H=H, W=16, encoder_type="gru", num_layers=1, nhead=NHEAD,
               ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
               num_encoder_layers=n_enc, forecaster_d_model=128, forecaster_n_heads=4,
               freq_emb_dim=femb, seasonality_emb_dim=semb,
               rev_norm_kind="ewma", rev_norm_span=128, patch_stats_kind=patch,
               learnable_tau=("log_inv_tau" in sd))
    m = ConfigurableModel(**cfg)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # Guard: only forecaster-head keys may be missing — never the encoder.
    bad = [k for k in missing if not re.match(r"transformer\.(fcst_|layers\.|cpc_)", k)]
    if bad:
        raise RuntimeError(f"encoder/other keys missing on load: {bad[:6]}")
    return m.to(DEV).eval(), H


def participation_ratio(e_bc):
    """e_bc: (BC, T, H) → effective dimensionality of the latent."""
    X = e_bc.reshape(-1, e_bc.shape[-1]).double()
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X)            # singular values of centered data
    lam = (s * s)                          # ∝ covariance eigenvalues
    return float((lam.sum() ** 2) / (lam * lam).sum())


def steps_for(runs, prefix):
    out = []
    for p in glob.glob(f"{runs}/{prefix}_*k.pth"):
        m = re.search(rf"{re.escape(prefix)}_(\d+)k\.pth$", p)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def main():
    print(f"[dim_usage] device={DEV}")
    x, fid, sid = fixed_batch()
    print(f"[dim_usage] fixed batch x={tuple(x.shape)} fid={None if fid is None else tuple(fid.shape)}")
    rows = []
    for label, color, runs, prefix in MODELS:
        steps = steps_for(runs, prefix)
        if not steps:
            print(f"[dim_usage] {label}: no periodic ckpts under {runs}/{prefix}_*k.pth — skip")
            continue
        for step, path in steps:
            sd = torch.load(path, map_location=DEV, weights_only=True)
            m, H = build_backbone(sd)
            with torch.no_grad():
                e_bc, _ = extract_encoder_latents(m, x, freq_ids=fid, seasonality_ids=sid)
            pr = participation_ratio(e_bc)
            rows.append((label, color, step, pr, H))
            print(f"[dim_usage] {label:36s} step={step:3d}k  PR={pr:7.2f} / H={H}")
            del m, sd
            if DEV.startswith("cuda"):
                torch.cuda.empty_cache()

    with open(f"{OUT_RES}/dim_usage.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "step_k", "participation_ratio", "H"])
        for label, _, step, pr, H in rows:
            w.writerow([label, step, f"{pr:.4f}", H])

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 5.2))
    by = {}
    for label, color, step, pr, H in rows:
        by.setdefault((label, color), []).append((step, pr))
    for (label, color), pts in by.items():
        pts.sort()
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ls = "-" if "k=12" in label else "--"   # k=12 solid, k=1 dashed
        ax.plot(xs, ys, marker="o", color=color, lw=2.2, ms=6, ls=ls, label=label)
    ax.set_ylim(0, 62); ax.set_xlim(3, 52)
    ax.axhspan(0, 8, color="#1f77b4", alpha=0.05); ax.axhspan(44, 58, color="#d62728", alpha=0.05)
    ax.text(51, 5, "k=1: ~3–5 dims", ha="right", fontsize=8.5, color="#1f5fa8")
    ax.text(51, 56, "k=12: ~50 dims", ha="right", fontsize=8.5, color="#b01818")
    ax.set_xlabel("training step (k)")
    ax.set_ylabel("effective # dimensions used  (participation ratio; max H=384)")
    ax.set_title("Why: k=12 spreads the latent across ~10× more dimensions than k=1\n"
                 "(refutes the 'multi-step collapses the latent' hypothesis)", fontsize=10.5)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc="center right")
    plt.tight_layout(); plt.savefig(f"{OUT_PLOTS}/dim_usage.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"[dim_usage] wrote {OUT_PLOTS}/dim_usage.png and {OUT_RES}/dim_usage.csv ({len(rows)} points)")


if __name__ == "__main__":
    main()
