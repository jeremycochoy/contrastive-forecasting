#!/usr/bin/env python3
"""#415 — what one GIFT-Eval series costs under A2 and under B4, on one core.

Both strategies score the same quantile head on the same backbone. B4 encodes
the context once and rolls the forecaster alone. A2 re-runs the whole cell,
input head included, once per forecast patch. So A2 costs more, and the ratio
says how much longer the A2 eval of a stop runs than its B4 eval.

The backbone and the head are random, at this card's shape: the cost of a
forward does not depend on the weights.

Usage:  python3 a2_cost.py [horizon ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.forecasting_head import (  # noqa: E402
    TransformerQuantileForecastingHead,
    forecast_with_strategy,
)
from src.models import ConfigurableModel  # noqa: E402

T_RAW = 4096


def build_backbone():
    """#414's cell at `d_model` 384, as the eval builds it."""
    return ConfigurableModel(
        C=1, H=384, W=16, encoder_type="gru", num_layers=3, nhead=8,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        freq_emb_dim=3, seasonality_emb_dim=3, rev_norm_kind="ewma",
        rev_norm_span=128, num_encoder_layers=3, qk_norm=True,
        attn_out_norm=True).eval()


def build_head():
    """#373's head: 2 layers, 8 heads, forecast length 16."""
    return TransformerQuantileForecastingHead(
        H=384, num_layers=2, nhead=8, ffn_mult=4.0, forecast_len=16,
        dropout=0.1, causal=True).eval()


def seconds(strategy, backbone, head, context, horizon):
    t0 = time.perf_counter()
    forecast_with_strategy(strategy, backbone, head, context, horizon, "cpu")
    return time.perf_counter() - t0


def main(horizons):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    backbone, head = build_backbone(), build_head()
    context = torch.randn(1, T_RAW, 1)
    seconds("B4", backbone, head, context, 16)    # warm the kernels once
    print("horizon  B4_s   A2_s   A2/B4")
    for h in horizons:
        b4 = seconds("B4", backbone, head, context, h)
        a2 = seconds("A2", backbone, head, context, h)
        print(f"{h:7d}  {b4:5.2f}  {a2:5.2f}  {a2 / b4:5.1f}")


if __name__ == "__main__":
    main([int(h) for h in sys.argv[1:]] or [48, 480, 720])
