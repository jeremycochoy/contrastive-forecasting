#!/usr/bin/env python3
"""#401 — a rank-1 encoder scored 1.7939. Does it still vary along time?

`diag_collapse.py` measures ONE vector per series: `h` at the last step.
Its `eff_rank` is therefore the rank ACROSS SERIES at a fixed time. Rank 1
there means all series share one direction at that instant. It does not say
whether `h` moves as the series moves, and the forecasting head reads the
whole `h` sequence, not the last step alone.

This script measures the other axis, on the same windows, with the same
loader. Per series, over time:

    time_pair_cos   mean cosine between `h_t` and `h_s` for t != s, inside
                    one series. 1.0 means `h` never turns: the encoder is a
                    constant function of time as well.
    time_eff_rank   participation ratio of the covariance of `h` over time,
                    inside one series, averaged over series. 1.0 is one
                    direction.
    time_dim_std    mean per-dimension standard deviation over time.

Both scripts read the same 21 real GIFT-Eval windows through the same
`load_backbone`. This file adds no measurement to `diag_collapse.py` and
changes none of its numbers.

Usage:
    python3 diag_time_rank.py --out results/diag/time_rank.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from diag_collapse import (                                    # noqa: E402
    SUBJECTS, discover, load_backbone, real_windows,
)
from src.forecasting_head import extract_encoder_latents       # noqa: E402


@torch.no_grad()
def measure_time(model, x, device, stride=8):
    """Spread of `h` along time, inside each series, then averaged."""
    h = extract_encoder_latents(model.to(device), x.to(device))[0]
    h = h.reshape(h.shape[0], h.shape[1], -1).float()[:, ::stride, :]
    n, T, H = h.shape

    hn = torch.nn.functional.normalize(h, dim=-1)
    cos = hn @ hn.transpose(1, 2)                      # (n, T, T)
    off = ~torch.eye(T, dtype=torch.bool, device=cos.device)
    pair = cos[:, off].mean(dim=1)                     # per series

    ranks, stds = [], []
    for i in range(n):
        v = h[i]                                       # (T, H)
        c = torch.cov(v.T.double())
        lam = torch.linalg.eigvalsh(c).clamp(min=0)
        ranks.append(((lam.sum() ** 2) / (lam.pow(2).sum() + 1e-30)).item())
        stds.append(v.std(dim=0).mean().item())

    return dict(time_pair_cos=pair.mean().item(),
                time_eff_rank=float(sum(ranks) / len(ranks)),
                time_dim_std=float(sum(stds) / len(stds)),
                n_series=n, n_times=T, H=H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/diag/time_rank.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--per-dataset", type=int, default=8)
    ap.add_argument("--stride", type=int, default=8)
    a = ap.parse_args()

    have = {s[3] for s in SUBJECTS}
    subjects = list(SUBJECTS) + [r for r in discover() if r[3] not in have]

    x = real_windows(a.per_dataset)
    print(f"{x.shape[0]} real windows of {x.shape[1]} steps\n")
    print(f"{'backbone':<24} {'t_pair_cos':>11} {'t_eff_rank':>11} "
          f"{'t_dim_std':>10}")

    cols = ["label", "k", "stop_k", "step_k", "time_pair_cos",
            "time_eff_rank", "time_dim_std", "n_series", "n_times", "H",
            "checkpoint"]
    rows = []
    for label, k, stop, path in subjects:
        if not Path(path).is_file():
            continue
        model, _ = load_backbone(path, a.device)
        m = measure_time(model, x, a.device, a.stride)
        print(f"{label:<24} {m['time_pair_cos']:11.5f} "
              f"{m['time_eff_rank']:11.3f} {m['time_dim_std']:10.5f}")
        st = Path(path).stem.rsplit("_", 1)[1]
        st = int(st[:-1]) if st.endswith("k") and st[:-1].isdigit() else stop
        rows.append(dict(label=label, k=k, stop_k=stop, step_k=st,
                         checkpoint=path, **m))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r[c] for c in cols})
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
