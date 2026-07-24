"""Paired-bootstrap CIs of every sibling arm cell against seed-2 arm C.

Arm C's original seed-20260520 per-task file (from experiment
`2026-06-28_sigreg_lambda_tau_cross`) was not committed and is now
lost. A seed-2 retrain of the same recipe
(`lC_emb10_enc10_tau090` — λ_e = 1, λ_h = 1, τ = 0.90) exists in
`2026-07-07_b512_armC_seed2_traj/` with full per-task files at
backbone steps 12,500 / 25,000 / 50,000, for both 2L and 6L. We
match each sibling arm cell to the arm-C snapshot at the same
backbone step:

    sibling `last`  ↔  arm C step-12,500
    sibling  25k    ↔  arm C step-25,000
    sibling  50k    ↔  arm C step-50,000

Sibling `best` cells and 2k cells are skipped (arm C has no
best_loss save and no 2k save). bimoco 50k is skipped (no data).
Emits results/pairwise_bootstrap_ci_vs_armC.csv (34 rows).
"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd

EXP = Path(__file__).parent.parent
SN = EXP / "results" / "seasonal_naive_all_results.csv"
OUT = EXP / "results" / "pairwise_bootstrap_ci_vs_armC.csv"
MASE = "eval_metrics/MASE[0.5]"
N_BOOT = 200_000
SEED = 42

SIBLINGS = {
    "arm1":   ("results",           "gift_eval_full_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"),
    "arm3":   ("results",           "gift_eval_full_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"),
    "arm4":   ("results_arm4",      "gift_eval_full_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090"),
    "arm5":   ("results_arm5",      "gift_eval_full_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090"),
    "arm6":   ("results_arm6_v2",   "gift_eval_full_lalign_lrepmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6v2_tau090"),
    "bimoco": ("results_bimoco_v2", "gift_eval_full_split_pred_rep_bimoco_v2_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"),
}

CELL_TO_STEP = {"last": 12500, "25k": 25000, "50k": 50000}


def sibling_csv(arm, cell, head):
    d, base = SIBLINGS[arm]
    suffix = {"last": f"_last_{head}", "25k": f"_25k_{head}", "50k": f"_50k_{head}"}[cell]
    return EXP / d / (base + suffix) / "all_results.csv"


def armC_csv(step, head):
    return EXP / "results_armC_seed2" / f"gift_eval_full_armC_seed2_step{step}_{head}" / "all_results.csv"


def rel_mase(path, sn):
    df = pd.read_csv(path).set_index("dataset")
    common = df.index.intersection(sn.index)
    return (df.loc[common, MASE] / sn.loc[common, MASE]).dropna()


def bootstrap(ra, rb, seed):
    common = ra.index.intersection(rb.index)
    a, b = np.log(ra.loc[common].values), np.log(rb.loc[common].values)
    n = len(common)
    rng = np.random.default_rng(seed)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots[i] = np.exp(a[idx].mean()) / np.exp(b[idx].mean())
    gm_a, gm_b = float(np.exp(a.mean())), float(np.exp(b.mean()))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p_one = float((boots < 1.0).mean())
    return gm_a, gm_b, gm_a / gm_b, float(lo), float(hi), p_one, n


def main():
    sn = pd.read_csv(SN).set_index("dataset")
    rows = []
    for arm in SIBLINGS:
        for cell, step in CELL_TO_STEP.items():
            for head in ("2L", "6L"):
                sib = sibling_csv(arm, cell, head)
                if not sib.exists():
                    print(f"skip {arm} {head} {cell}: no sibling file")
                    continue
                ac = armC_csv(step, head)
                ra, rb = rel_mase(sib, sn), rel_mase(ac, sn)
                ga, gb, r, lo, hi, p_one, n = bootstrap(ra, rb, SEED)
                p_two = 2 * min(p_one, 1 - p_one)
                rows.append([head, cell, step, arm, "armC_seed2",
                             ga, gb, r, lo, hi, p_one, p_two, n, N_BOOT, SEED])
                print(f"{arm} {head}/{cell} vs armC step{step}: ratio={r:.4f} CI[{lo:.4f},{hi:.4f}] p1={p_one:.4f}")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "cell", "step", "arm_a", "arm_b",
                    "gm_a", "gm_b", "ratio_a_over_b", "ci_lo", "ci_hi",
                    "one_sided_p", "two_sided_p", "n", "n_boot", "seed"])
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
