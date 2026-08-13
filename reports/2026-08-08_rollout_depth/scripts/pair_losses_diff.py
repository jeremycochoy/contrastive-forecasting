#!/usr/bin/env python3
"""#373 — compare the TRAINING LOG of the two cells in a same-arm pair.

`pair_identity.py` compares the two backbones after training. This compares
what happened during it. It is the stronger statement: two files can hold
equal weights because one was copied over the other, but two training logs
agree row by row only if the two runs took the same trajectory.

For every column of the losses CSV it reports how many rows differ. A pair
whose EMA regime is the only difference should show exactly one differing
column, `ema_tau`, and nothing else.

Usage: pair_losses_diff.py            # writes results/pair_losses_diff.tsv
"""
import csv
import os
import sys

R2 = "/home/jupyter/cf373_r2"
B3D = (f"{R2}/B3/sync/bb_small_arm5_combab_lalign_lrep_enc3l3_b64_200k"
       "_sigreg_ema_qk_aon_cpc_tau090_cf373k3")
B1D = (f"{R2}/B1/sync/bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k"
       "_sigreg_ema_qk_aon_cpc_tau090_cf373k3")

# pair, arm, leg label, cell-A csv, cell-B csv
PAIRS = [
    ("A1/B3", "arm5_combab_alignS", "steps 1..40k",
     f"{R2}/A1/sync/arm5_combab_alignS/leg_40k/"
     "cf393_arm5_combab_alignS_cf373k3_losses.csv",
     f"{B3D}/bb_small_arm5_combab_lalign_lrep_enc3l3_b64_200k"
     "_sigreg_ema_qk_aon_cpc_tau090_cf373k3_losses.csv"),
    ("A1/B3", "arm5_combab_alignS", "steps 40k..100k",
     f"{R2}/A1/sync/arm5_combab_alignS/leg_100k/"
     "cf393_arm5_combab_alignS_cf373k3_losses.csv",
     f"{B3D}/bb_small_arm5_combab_lalign_lrep_enc3l3_b64_200k"
     "_sigreg_ema_qk_aon_cpc_tau090_cf373k3_r2_losses.csv"),
    ("A4/B1", "arm6_v2_combab_alignS", "steps 40k..100k",
     f"{R2}/A4/sync/arm6_v2_combab_alignS/leg_100k/"
     "cf393_arm6_v2_combab_alignS_cf373k3_losses.csv",
     f"{B1D}/bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k"
     "_sigreg_ema_qk_aon_cpc_tau090_cf373k3_r2_losses.csv"),
]


def rows_by_step(path):
    """Index a losses CSV by its step column, so two legs align on step and
    not on line number. A resumed leg starts at 40001, a fresh one at 1."""
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        return hdr, {r[0]: r for r in rd if r}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results")
    out = os.path.join(res, "pair_losses_diff.tsv")
    lines = ["pair\tarm\tleg\tsteps_compared\tcols_differing\tdiffering_columns"]

    for pair, arm, leg, pa, pb in PAIRS:
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"{pair} {leg}: SKIP, a leg is not on this machine")
            continue
        ha, ra = rows_by_step(pa)
        hb, rb = rows_by_step(pb)
        if ha != hb:
            print(f"{pair} {leg}: SKIP, headers differ")
            continue
        common = sorted(set(ra) & set(rb), key=int)
        diff = {}
        for s in common:
            a, b = ra[s], rb[s]
            if a == b:
                continue
            for j, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    diff[ha[j]] = diff.get(ha[j], 0) + 1
        names = ",".join(f"{c}:{n}" for c, n in
                         sorted(diff.items(), key=lambda kv: -kv[1])) or "NONE"
        lines.append(f"{pair}\t{arm}\t{leg}\t{len(common)}\t{len(diff)}\t{names}")
        print(f"{pair} {arm} {leg}: {len(common)} steps compared, "
              f"{len(diff)} column(s) differ — {names}")

    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
