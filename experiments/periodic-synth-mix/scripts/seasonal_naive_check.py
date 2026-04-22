#!/usr/bin/env python3
"""Sanity check: does seasonal-naive trounce naive on our synthetic periodic data?

The whole reason we are training on this data is to teach the backbone to
imitate seasonal-naive on periodic input. If seasonal-naive itself is NOT
substantially better than the trivial "persist last value" baseline on the
synthetic stream, the training signal would be uninformative.

We compute MASE where the scale denominator is the mean absolute step-to-step
difference of the context (matches how GIFT-Eval normalises) and report:

- naive:            ``yhat[t+h] = y[T-1]``           for h in [0, H)
- seasonal-naive:   ``yhat[t+h] = y[T-1 - P + (h mod P)]``
  (with ``P`` the TRUE synth period rounded to nearest integer)
- also an "inferred-P" variant that picks P by autocorrelation so we can
  see if the model would still benefit when P isn't given.
"""

from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.synthetic_periodic import generate_periodic_batch, primitive_name


def mase(y_true: np.ndarray, y_pred: np.ndarray, ctx: np.ndarray) -> float:
    """Mean Absolute Scaled Error with naive context scale.

    scale = mean |y_ctx[t] - y_ctx[t-1]|  (naive 1-step RW error on context).
    """
    denom = np.mean(np.abs(np.diff(ctx)))
    if denom < 1e-12:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def infer_period_by_autocorr(ctx: np.ndarray, lo: int = 4, hi: int | None = None) -> int:
    """Pick P that maximises normalised autocorrelation in [lo, hi]."""
    n = len(ctx)
    if hi is None:
        hi = n // 2
    c = ctx - ctx.mean()
    denom = np.sum(c * c) + 1e-12
    best_p, best_r = lo, -np.inf
    for p in range(lo, hi + 1):
        r = np.sum(c[:-p] * c[p:]) / denom
        if r > best_r:
            best_r, best_p = r, p
    return best_p


def main():
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(out_dir, exist_ok=True)

    N = 1000
    T = 1024
    CTX = 768
    H = 128          # forecast horizon
    SEED = 20260422

    X, meta = generate_periodic_batch(
        batch_size=N, T_raw=T, C=1, seed=SEED, return_meta=True)
    x = X.squeeze(-1).numpy()                # [N, T]
    ctx = x[:, :CTX]
    truth = x[:, CTX:CTX + H]

    # -- naive (copy last) ----------------------------------------------------
    naive_pred = np.broadcast_to(ctx[:, -1:], truth.shape)
    naive_mase = np.array([mase(truth[i], naive_pred[i], ctx[i]) for i in range(N)])

    # -- seasonal-naive with true period --------------------------------------
    # y_pred[t+h] = y[T-1 - P + (h mod P)] = last P context values tiled.
    sn_pred = np.empty_like(truth)
    for i in range(N):
        P = int(round(meta["spp"][i]))
        P = max(2, min(P, CTX - 1))
        last_period = ctx[i, -P:]
        offs = np.arange(H) % P
        sn_pred[i] = last_period[offs]
    sn_mase = np.array([mase(truth[i], sn_pred[i], ctx[i]) for i in range(N)])

    # -- seasonal-naive with inferred period ---------------------------------
    sn_inf_pred = np.empty_like(truth)
    inferred_P = np.empty(N, dtype=int)
    for i in range(N):
        P = infer_period_by_autocorr(ctx[i], lo=4, hi=CTX // 2)
        inferred_P[i] = P
        last_period = ctx[i, -P:]
        offs = np.arange(H) % P
        sn_inf_pred[i] = last_period[offs]
    sn_inf_mase = np.array([mase(truth[i], sn_inf_pred[i], ctx[i]) for i in range(N)])

    # -- finite-only aggregation ---------------------------------------------
    def gm(a: np.ndarray) -> float:
        a = a[np.isfinite(a) & (a > 0)]
        if len(a) == 0:
            return np.nan
        return float(np.exp(np.mean(np.log(a))))

    def mean_(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        return float(a.mean()) if len(a) else np.nan

    def med_(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        return float(np.median(a)) if len(a) else np.nan

    n_finite = int(np.isfinite(naive_mase & (naive_mase > 0)).sum() if False else
                   np.isfinite(naive_mase).sum())

    # -- Per-primitive breakdown ----------------------------------------------
    rows = []
    header = (
        f"{'group':>18s} {'N':>5s} "
        f"{'naive mean':>12s} {'naive med':>10s} {'naive GM':>10s} "
        f"{'SN mean':>10s} {'SN med':>10s} {'SN GM':>10s} "
        f"{'SN-inf mean':>12s} {'SN-inf med':>10s} {'SN-inf GM':>10s} "
        f"{'ratio GM':>10s}"
    )
    print(header)
    rows.append(header)

    def report(name: str, idx: np.ndarray):
        if len(idx) == 0:
            return
        n = len(idx)
        nv = naive_mase[idx]
        sn = sn_mase[idx]
        sni = sn_inf_mase[idx]
        # finite count sanity
        line = (
            f"{name:>18s} {n:>5d} "
            f"{mean_(nv):>12.3f} {med_(nv):>10.3f} {gm(nv):>10.3f} "
            f"{mean_(sn):>10.3f} {med_(sn):>10.3f} {gm(sn):>10.3f} "
            f"{mean_(sni):>12.3f} {med_(sni):>10.3f} {gm(sni):>10.3f} "
            f"{gm(sn)/gm(nv) if gm(nv) > 0 else float('nan'):>10.3f}"
        )
        print(line)
        rows.append(line)

    report("ALL", np.arange(N))
    report("sinusoid", np.where(meta["primitive"] == 0)[0])
    report("square", np.where(meta["primitive"] == 1)[0])
    report("saw", np.where(meta["primitive"] == 2)[0])
    report("env=True", np.where(meta["use_env"])[0])
    report("env=False", np.where(~meta["use_env"])[0])
    # short vs long period buckets
    report("P<=32", np.where(meta["spp"] <= 32)[0])
    report("P in (32,96]", np.where((meta["spp"] > 32) & (meta["spp"] <= 96))[0])
    report("P>96", np.where(meta["spp"] > 96)[0])

    # -- Save ------------------------------------------------------------------
    out_path = os.path.join(out_dir, "seasonal_naive_sanity.txt")
    with open(out_path, "w") as f:
        f.write(f"# Seasonal-naive sanity check on synthetic periodic data\n")
        f.write(f"# N={N} T={T} CTX={CTX} H={H} seed={SEED}\n")
        f.write(f"# naive: persist last value\n")
        f.write(f"# SN: seasonal-naive with TRUE period (from synth meta)\n")
        f.write(f"# SN-inf: seasonal-naive with P inferred by autocorrelation\n\n")
        for r in rows:
            f.write(r + "\n")
        # How often inferred-P matches true-P
        true_P = np.round(meta["spp"]).astype(int)
        exact = int((inferred_P == true_P).sum())
        close_10pct = int((np.abs(inferred_P - true_P) <= np.maximum(1, true_P // 10)).sum())
        f.write(f"\n# P inference: exact match {exact}/{N}, "
                f"within 10% {close_10pct}/{N}\n")
    print(f"\nwrote {out_path}")

    # Assertion: seasonal-naive (true P) GM-MASE must be << naive GM-MASE on ALL
    assert gm(sn_mase) < 0.3 * gm(naive_mase), \
        f"SN GM {gm(sn_mase):.3f} not << naive GM {gm(naive_mase):.3f}"
    print(f"\nASSERT PASS: SN/naive GM ratio = {gm(sn_mase)/gm(naive_mase):.3f} (< 0.3)")


if __name__ == "__main__":
    main()
