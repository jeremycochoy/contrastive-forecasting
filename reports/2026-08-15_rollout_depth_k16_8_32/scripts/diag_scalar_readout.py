#!/usr/bin/env python3
"""#401 — a rank-1 encoder still passes one number. Does that number move
with the input?

`diag_collapse.py` and `diag_time_rank.py` agree: every #401 backbone has
effective rank near 1 on both axes. So `h(x, t)` is close to `a(x, t) * u`
for one fixed direction `u`. Rank 1 bounds the encoder to ONE channel. It
does not say the channel is empty.

This script measures the channel. Per checkpoint, on the same 21 windows,
with the same loader:

    u              the top eigenvector of the covariance of `h` over
                   (series, time).
    a(x, t)        the projection of `h(x, t)` on `u`.
    readout_r      mean |Pearson r| between `a(x, .)` and the input series
                   `x`, over series. 0.0 means the one channel is deaf to
                   the input. High means the head can still read the series
                   through it.

A forecasting head sees `h`, so `readout_r` bounds what the head can use.

Usage:
    python3 diag_scalar_readout.py --out results/diag/scalar_readout.csv
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
def measure_readout(model, x, device):
    """How much of the input survives in the top latent direction."""
    h = extract_encoder_latents(model.to(device), x.to(device))[0]
    h = h.reshape(h.shape[0], h.shape[1], -1).float()           # (n, T, H)
    n, T, H = h.shape

    flat = h.reshape(-1, H).double()
    c = torch.cov(flat.T)
    lam, vec = torch.linalg.eigh(c)
    u = vec[:, -1]                                              # top direction
    share = (lam[-1] / (lam.sum() + 1e-30)).item()              # its variance
    a = (h.double() @ u)                                        # (n, T)

    # The encoder pools the raw series to T latent steps. Pool the input the
    # same way, by mean over each block, so the two line up in time.
    blk = x.shape[1] // T
    xin = x[:, :blk * T, 0].reshape(n, T, blk).mean(-1).double().to(a.device)

    rs = []
    for i in range(n):
        p, q = a[i], xin[i]
        p = p - p.mean()
        q = q - q.mean()
        d = (p.norm() * q.norm())
        rs.append(abs((p @ q / d).item()) if d > 1e-20 else 0.0)
    return dict(readout_r=float(sum(rs) / len(rs)),
                top_dir_share=share, n_series=n, n_times=T, H=H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/diag/scalar_readout.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--per-dataset", type=int, default=8)
    a = ap.parse_args()

    have = {s[3] for s in SUBJECTS}
    subjects = list(SUBJECTS) + [r for r in discover() if r[3] not in have]

    x = real_windows(a.per_dataset)
    print(f"{x.shape[0]} real windows of {x.shape[1]} steps\n")
    print(f"{'backbone':<24} {'readout_r':>10} {'top_dir_share':>14}")

    cols = ["label", "k", "stop_k", "step_k", "readout_r", "top_dir_share",
            "n_series", "n_times", "H", "checkpoint"]
    rows = []
    for label, k, stop, path in subjects:
        if not Path(path).is_file():
            continue
        model, _ = load_backbone(path, a.device)
        m = measure_readout(model, x, a.device)
        print(f"{label:<24} {m['readout_r']:10.4f} {m['top_dir_share']:14.5f}")
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
