"""Build `results/pairwise_bootstrap_ci.csv` — the CI panel the report cites.

Iterates over every pair of arms in {arm1, arm3, arm4, arm5} at every
(head, ckpt) in {2L, 6L} × {best, last}, calling the paired-bootstrap
utility with n_boot=20000, seed 42, and the branch-committed
seasonal-naive reference. Emits four panels:
  (i)   task-level over all 97 configs;
  (ii)  the periodic-cluster subset the card names (37 configs — solar,
        electricity, ett1, m4_hourly, bizitobs_* — selected by family
        prefix, not by rel ≥ 1.25, to avoid conditioning on the outcome);
  (iii) a dataset-clustered bootstrap that resamples the 28 base
        datasets rather than the 97 configs;
  (iv)  the medium+long horizon subset the card's secondary read names
        (42 configs — dataset/*/{medium,long}).

Idempotent; writes six panels:
  results/pairwise_bootstrap_ci.csv                    (task-level, 97 configs, 24 rows)
  results/pairwise_bootstrap_ci_periodic.csv           (37-config periodic subset, task, 24 rows)
  results/pairwise_bootstrap_ci_clustered.csv          (28-dataset clustered over 97, 24 rows)
  results/pairwise_bootstrap_ci_medlong.csv            (42-config medium+long subset, task, 24 rows)
  results/pairwise_bootstrap_ci_periodic_clustered.csv (7-dataset clustered over periodic, 24 rows)
  results/pairwise_bootstrap_ci_medlong_clustered.csv  (14-dataset clustered over medium+long, 24 rows)
  results/pairwise_bootstrap_ci_arm5_nboot200k.csv     (12 arm-5 rows re-run at n_boot=200000, task-level)

Usage:  python3 scripts/build_ci_panel.py
"""
import csv, itertools, re
from pathlib import Path
import numpy as np
import pandas as pd

EXP = Path(__file__).parent.parent
SN = EXP / "results" / "seasonal_naive_all_results.csv"
OUT_ALL = EXP / "results" / "pairwise_bootstrap_ci.csv"
OUT_PER = EXP / "results" / "pairwise_bootstrap_ci_periodic.csv"
OUT_CLU = EXP / "results" / "pairwise_bootstrap_ci_clustered.csv"
OUT_HOR = EXP / "results" / "pairwise_bootstrap_ci_medlong.csv"
OUT_PER_CLU = EXP / "results" / "pairwise_bootstrap_ci_periodic_clustered.csv"
OUT_HOR_CLU = EXP / "results" / "pairwise_bootstrap_ci_medlong_clustered.csv"
OUT_ARM5_200K = EXP / "results" / "pairwise_bootstrap_ci_arm5_nboot200k.csv"
N_BOOT_ARM5 = 200000

ARMS = {
    "arm1": ("results", "gift_eval_full_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"),
    "arm3": ("results", "gift_eval_full_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"),
    "arm4": ("results_arm4", "gift_eval_full_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090"),
    "arm5": ("results_arm5", "gift_eval_full_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090"),
}
GROUPS = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
MASE = "eval_metrics/MASE[0.5]"
N_BOOT = 20000
SEED = 42

PERIODIC_PREFIXES = ("solar/", "electricity/", "ett1/", "m4_hourly/", "bizitobs_")
MEDLONG_SUFFIXES = ("/medium", "/long")  # the term suffix used in the config name


def base_dataset(name: str) -> str:
    # e.g. "electricity/15T/short" -> "electricity"
    return name.split("/", 1)[0]


def csv_path(arm: str, head: str, ckpt: str) -> Path:
    d, base = ARMS[arm]
    suffix = f"_{head}" if ckpt == "best" else f"_last_{head}"
    return EXP / d / (base + suffix) / "all_results.csv"


def rel_mase(all_results: Path, sn_df: pd.DataFrame) -> pd.Series:
    df = pd.read_csv(all_results).set_index("dataset")
    common = df.index.intersection(sn_df.index)
    return (df.loc[common, MASE] / sn_df.loc[common, MASE]).dropna()


def bootstrap_task(ra: pd.Series, rb: pd.Series, rng, n_boot=None):
    common = ra.index.intersection(rb.index)
    a, b = np.log(ra.loc[common].values), np.log(rb.loc[common].values)
    n = len(common)
    gm_a, gm_b = float(np.exp(a.mean())), float(np.exp(b.mean()))
    B = n_boot if n_boot is not None else N_BOOT
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        boots[i] = np.exp(a[idx].mean()) / np.exp(b[idx].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p_a_lt_b = float((boots < 1.0).mean())
    return gm_a, gm_b, gm_a / gm_b, float(lo), float(hi), p_a_lt_b, n


def bootstrap_cluster(ra: pd.Series, rb: pd.Series, rng, n_boot=None):
    common = ra.index.intersection(rb.index)
    a, b = np.log(ra.loc[common].values), np.log(rb.loc[common].values)
    ds = np.array([base_dataset(x) for x in common])
    unique_ds = sorted(set(ds))
    n_ds = len(unique_ds)
    idx_by_ds = {d: np.where(ds == d)[0] for d in unique_ds}
    gm_a, gm_b = float(np.exp(a.mean())), float(np.exp(b.mean()))
    B = n_boot if n_boot is not None else N_BOOT
    boots = np.empty(B)
    for i in range(B):
        sampled_ds = [unique_ds[k] for k in rng.integers(0, n_ds, n_ds)]
        idx = np.concatenate([idx_by_ds[d] for d in sampled_ds])
        boots[i] = np.exp(a[idx].mean()) / np.exp(b[idx].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p_a_lt_b = float((boots < 1.0).mean())
    return gm_a, gm_b, gm_a / gm_b, float(lo), float(hi), p_a_lt_b, len(common), n_ds


def main():
    sn = pd.read_csv(SN).set_index("dataset")

    print(f"n_boot={N_BOOT} seed={SEED}  sn={SN}")
    rows_all, rows_per, rows_clu, rows_hor = [], [], [], []
    rows_per_clu, rows_hor_clu = [], []
    rows_arm5_200k = []
    for HL, ck in GROUPS:
        # cache each arm's per-task rel-MASE at this cell
        rel = {a: rel_mase(csv_path(a, HL, ck), sn) for a in ARMS}
        rel_periodic = {a: v[[t for t in v.index if t.startswith(PERIODIC_PREFIXES)]]
                        for a, v in rel.items()}
        rel_medlong = {a: v[[t for t in v.index if t.endswith(MEDLONG_SUFFIXES)]]
                       for a, v in rel.items()}
        for a, b in itertools.combinations(ARMS, 2):
            rng = np.random.default_rng(SEED)
            ga, gb, r, lo, hi, p, n = bootstrap_task(rel[a], rel[b], rng)
            rows_all.append([HL, ck, a, b, ga, gb, r, lo, hi, p, n, N_BOOT, SEED])
            rng = np.random.default_rng(SEED)
            ga_p, gb_p, r_p, lo_p, hi_p, p_p, n_p = bootstrap_task(rel_periodic[a], rel_periodic[b], rng)
            rows_per.append([HL, ck, a, b, ga_p, gb_p, r_p, lo_p, hi_p, p_p, n_p, N_BOOT, SEED])
            rng = np.random.default_rng(SEED)
            ga_c, gb_c, r_c, lo_c, hi_c, p_c, n_c, n_ds = bootstrap_cluster(rel[a], rel[b], rng)
            rows_clu.append([HL, ck, a, b, ga_c, gb_c, r_c, lo_c, hi_c, p_c, n_c, n_ds, N_BOOT, SEED])
            rng = np.random.default_rng(SEED)
            ga_h, gb_h, r_h, lo_h, hi_h, p_h, n_h = bootstrap_task(rel_medlong[a], rel_medlong[b], rng)
            rows_hor.append([HL, ck, a, b, ga_h, gb_h, r_h, lo_h, hi_h, p_h, n_h, N_BOOT, SEED])
            rng = np.random.default_rng(SEED)
            ga_pc, gb_pc, r_pc, lo_pc, hi_pc, p_pc, n_pc, n_pds = bootstrap_cluster(rel_periodic[a], rel_periodic[b], rng)
            rows_per_clu.append([HL, ck, a, b, ga_pc, gb_pc, r_pc, lo_pc, hi_pc, p_pc, n_pc, n_pds, N_BOOT, SEED])
            rng = np.random.default_rng(SEED)
            ga_hc, gb_hc, r_hc, lo_hc, hi_hc, p_hc, n_hc, n_hds = bootstrap_cluster(rel_medlong[a], rel_medlong[b], rng)
            rows_hor_clu.append([HL, ck, a, b, ga_hc, gb_hc, r_hc, lo_hc, hi_hc, p_hc, n_hc, n_hds, N_BOOT, SEED])
        # arm-5 rows at n_boot = 200 000 for the Bonferroni-resolution question:
        for other in ("arm1", "arm3", "arm4"):
            rng = np.random.default_rng(SEED)
            ga5, gb5, r5, lo5, hi5, p5, n5 = bootstrap_task(rel["arm5"], rel[other], rng, n_boot=N_BOOT_ARM5)
            p2 = 2 * min(p5, 1 - p5)
            se_1 = float(np.sqrt(p5 * (1 - p5) / N_BOOT_ARM5))
            rows_arm5_200k.append([HL, ck, "arm5", other, ga5, gb5, r5, lo5, hi5, p5, p2, 2 * se_1, n5, N_BOOT_ARM5, SEED])

    def write(path, header, rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows: w.writerow(r)

    write(OUT_ALL, ["head","ckpt","arm_a","arm_b","gm_a","gm_b","ratio_a_over_b","ci_lo","ci_hi","p_a_beats_b","n","n_boot","seed"], rows_all)
    write(OUT_PER, ["head","ckpt","arm_a","arm_b","gm_a","gm_b","ratio_a_over_b","ci_lo","ci_hi","p_a_beats_b","n_periodic","n_boot","seed"], rows_per)
    write(OUT_CLU, ["head","ckpt","arm_a","arm_b","gm_a","gm_b","ratio_a_over_b","ci_lo","ci_hi","p_a_beats_b","n","n_datasets","n_boot","seed"], rows_clu)
    write(OUT_HOR, ["head","ckpt","arm_a","arm_b","gm_a","gm_b","ratio_a_over_b","ci_lo","ci_hi","p_a_beats_b","n_medlong","n_boot","seed"], rows_hor)
    write(OUT_PER_CLU, ["head","ckpt","arm_a","arm_b","gm_a","gm_b","ratio_a_over_b","ci_lo","ci_hi","p_a_beats_b","n","n_datasets","n_boot","seed"], rows_per_clu)
    write(OUT_HOR_CLU, ["head","ckpt","arm_a","arm_b","gm_a","gm_b","ratio_a_over_b","ci_lo","ci_hi","p_a_beats_b","n","n_datasets","n_boot","seed"], rows_hor_clu)
    write(OUT_ARM5_200K, ["head","ckpt","arm_a","arm_b","gm_a","gm_b","ratio_a_over_b","ci_lo","ci_hi","one_sided_p","two_sided_p","mc_se_two_sided","n","n_boot","seed"], rows_arm5_200k)
    print(f"wrote {OUT_ALL.name}, {OUT_PER.name}, {OUT_CLU.name}, {OUT_HOR.name}, "
          f"{OUT_PER_CLU.name}, {OUT_HOR_CLU.name}, {OUT_ARM5_200K.name}")


if __name__ == "__main__":
    main()
