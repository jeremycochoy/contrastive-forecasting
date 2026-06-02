#!/usr/bin/env python3
"""Relative-gap analysis for V6 / V7 backbones.

The training JSONs log FF, FP, TP, CB over time. The cross-channel similarity
(neg_xx-equivalent: mean cos(h^{b,t,c1}, h^{b,t,c2}) for c1!=c2) and the
forecaster-cross-channel similarity (neg_xy_hat-equivalent:
cos(h^{b,t,c1}, h_hat^{b,t,c2}) for c1!=c2) are NOT in the logs, so we
recompute them by loading each backbone and running a small validation pass.

Outputs:
- plots/v6_v7_ratios.png with four panels:
    1) FF, FP, CB over training (V6, V7)
    2) ratios FF/FP, FF/CB, CB/FP over training
    3) final-state bar chart: FF, FP, TP, CC(h×h), CC(h×h_hat), CB
    4) final-state ratios bar chart
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from data import generate_arma_correlated_batch  # noqa: E402
from src.models import ConfigurableModel  # noqa: E402


def load_backbone(path: str, device: str = "cuda"):
    m = ConfigurableModel(
        C=4, H=1024, W=32, encoder_type="gru",
        num_layers=12, nhead=8, ffn_mult=4.0, dropout=0.1,
        activation="gelu", depthwise_conv=3,
    )
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


@torch.no_grad()
def final_state_metrics(model, x):
    h_hat, h = model(x)  # [B, T, C, H]
    h_hat_n = F.normalize(h_hat, p=2, dim=-1)
    h_n = F.normalize(h, p=2, dim=-1)

    cld = 1
    hyh = h_hat_n[:, :-cld]   # h_hat[t]
    hyn = h_n[:, cld:]        # h[t+1]
    hxn = h_n[:, :-cld]       # h[t]

    ff = (hyh * hyn).sum(-1).mean().item()                    # h_hat[t] · h[t+1]  same b,c
    fp = (hyh * hxn).sum(-1).mean().item()                    # h_hat[t] · h[t]    same b,c
    tp = (hyn * hxn).sum(-1).mean().item()                    # h[t+1]   · h[t]    same b,c

    B, T, C, H = hxn.shape
    eye_C = torch.eye(C, dtype=torch.bool, device=hxn.device).view(1, 1, C, C)

    # Cross-channel h × h : cos(h^{b,t,c1}, h^{b,t,c2}), c1 != c2
    cc_hh = (hxn.unsqueeze(3) * hxn.unsqueeze(2)).sum(-1)
    cc_hh = cc_hh.masked_fill(eye_C, 0).sum() / ((~eye_C).sum() * B * T)

    # Cross-channel h × h_hat: cos(h^{b,t,c1}, h_hat^{b,t,c2}), c1 != c2
    cc_hhhat = (hxn.unsqueeze(3) * hyh.unsqueeze(2)).sum(-1)
    cc_hhhat = cc_hhhat.masked_fill(eye_C, 0).sum() / ((~eye_C).sum() * B * T)

    # Cross-batch h_hat × h: same channel, time-shifted
    cb = (hyh.unsqueeze(0) * hyn.unsqueeze(1)).sum(-1)        # [B, B, T-1, C]
    mask_b = (~torch.eye(B, dtype=torch.bool, device=cb.device)).view(B, B, 1, 1)
    cb = cb.masked_fill(~mask_b, 0).sum() / (mask_b.sum() * (T) * C)  # T already T-1 here

    return {
        "ff": ff, "fp": fp, "tp": tp,
        "cc_hh": cc_hh.item(), "cc_hhhat": cc_hhhat.item(),
        "cb": cb.item(),
    }


def load_training_metrics(json_path):
    data = json.load(open(json_path))
    log = data["metrics_log"]
    return {
        "step":     np.array([r["step"] for r in log]),
        "val_ff":   np.array([r["val_ff"] for r in log]),
        "val_fp":   np.array([r["val_fp"] for r in log]),
        "val_cb":   np.array([r["val_cb"] for r in log]),
        "val_tp":   np.array([r["val_tp"] for r in log]),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    v6_json = HERE / "checkpoints" / "backbone_corrV6_results.json"
    v7_json = HERE / "checkpoints" / "backbone_corrV7_results.json"
    v6 = load_training_metrics(v6_json)
    v7 = load_training_metrics(v7_json)

    # Recompute cross-channel metrics at final state
    torch.manual_seed(0)
    x_val, _, _, _ = generate_arma_correlated_batch(
        batch_size=16, T_raw=4096, K=4, seed=0, device=device,
    )

    finals = {}
    for label, path in [
        ("V6", HERE / "checkpoints" / "corrV6_best_gap.pth"),
        ("V7", HERE / "checkpoints" / "corrV7_best_gap.pth"),
    ]:
        m = load_backbone(str(path), device=device)
        finals[label] = final_state_metrics(m, x_val)
        del m
        torch.cuda.empty_cache()

    print("Final-state metrics:")
    for k, v in finals.items():
        print(f"  {k}: {v}")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel 1: time-series FF, FP, CB
    ax = axes[0, 0]
    for color, label, m in [("C3", "V6", v6), ("C0", "V7", v7)]:
        ax.plot(m["step"], m["val_ff"], label=f"{label} FF", color=color, lw=1.4)
        ax.plot(m["step"], m["val_fp"], label=f"{label} FP", color=color, lw=1.0, ls="--")
        ax.plot(m["step"], m["val_cb"], label=f"{label} CB", color=color, lw=1.0, ls=":")
    ax.set_xlabel("step"); ax.set_ylabel("cos similarity"); ax.set_title("FF / FP / CB over training")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # Panel 2: ratios over training
    ax = axes[0, 1]
    for color, label, m in [("C3", "V6", v6), ("C0", "V7", v7)]:
        eps = 1e-6
        ax.plot(m["step"], m["val_ff"] / np.clip(m["val_fp"], eps, None), label=f"{label} FF/FP", color=color, lw=1.4)
        ax.plot(m["step"], m["val_ff"] / np.clip(m["val_cb"], eps, None), label=f"{label} FF/CB", color=color, lw=1.0, ls="--")
        ax.plot(m["step"], m["val_cb"] / np.clip(m["val_fp"], eps, None), label=f"{label} CB/FP", color=color, lw=1.0, ls=":")
    ax.set_xlabel("step"); ax.set_ylabel("ratio"); ax.set_title("Ratios over training")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, which="both")

    # Panel 3: final-state bar chart of absolute values
    ax = axes[1, 0]
    keys = ["ff", "fp", "tp", "cc_hh", "cc_hhhat", "cb"]
    nicekeys = ["FF\n(h_hat[t]·h[t+1]\nsame b,c)",
                "FP\n(h_hat[t]·h[t]\nsame b,c)",
                "TP\n(h[t+1]·h[t]\nsame b,c)",
                "CC(h,h)\n(neg_xx)\nsame b,t, c1≠c2",
                "CC(h,h_hat)\n(neg_xy_hat)\nsame b,t, c1≠c2",
                "CB\n(h_hat·h[t+1]\nb1≠b2, same c)"]
    x = np.arange(len(keys))
    w = 0.38
    ax.bar(x - w/2, [finals["V6"][k] for k in keys], w, color="C3", label="V6 (broken loss)")
    ax.bar(x + w/2, [finals["V7"][k] for k in keys], w, color="C0", label="V7 (fixed loss)")
    ax.set_xticks(x); ax.set_xticklabels(nicekeys, fontsize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("cos similarity"); ax.set_title("Final-state cosine similarities")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    # Panel 4: the three ratios of interest at final state.
    # Use |denominator| because CC values can be near zero or change sign.
    ax = axes[1, 1]
    ratio_defs = [
        ("FF / FP",      "ff",    "fp"),
        ("CC(h,h) / FP", "cc_hh", "fp"),
        ("FF / CC(h,h)", "ff",    "cc_hh"),
    ]
    x = np.arange(len(ratio_defs))
    eps = 1e-6
    v6r = [finals["V6"][a] / max(abs(finals["V6"][b]), eps) for _, a, b in ratio_defs]
    v7r = [finals["V7"][a] / max(abs(finals["V7"][b]), eps) for _, a, b in ratio_defs]
    ax.bar(x - w/2, v6r, w, color="C3", label="V6 (broken loss)")
    ax.bar(x + w/2, v7r, w, color="C0", label="V7 (fixed loss)")
    for i, (v6v, v7v) in enumerate(zip(v6r, v7r)):
        ax.text(i - w/2, v6v, f"{v6v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w/2, v7v, f"{v7v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in ratio_defs], fontsize=9)
    ax.axhline(1.0, color="k", lw=0.5, ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("ratio (log scale, |denom|)")
    ax.set_title("Final-state ratios (FF/FP, CC/FP, FF/CC)")
    ax.legend(); ax.grid(alpha=0.3, axis="y", which="both")

    fig.suptitle("V6 vs V7: cosine-similarity gaps and ratios", y=0.995, fontsize=11)
    fig.tight_layout()

    out = HERE / "plots" / "v6_v7_ratios.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")

    # Also dump the final-state numbers for the report
    out_json = HERE / "plots" / "v6_v7_ratios_finals.json"
    json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in finals.items()},
              open(out_json, "w"), indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
