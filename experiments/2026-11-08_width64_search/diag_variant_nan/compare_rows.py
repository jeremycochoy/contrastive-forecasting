"""Compare the replay's losses CSV with pass 2 of the run (its r2 file), row by row, every column.

Usage: python3 compare_rows.py  -> prints the rows compared and the first difference (none = the same run)
"""
import os
import csv
HP64 = os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64")
NAME = "g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10pw2_rollfix"
RUN = f"{HP64}/{NAME}/ckpt/{NAME}_r2_losses.csv"
REPLAY = f"{HP64}/diag_variant_nan/ckpt/{NAME}_losses.csv"
run = {r["step"]: r for r in csv.DictReader(open(RUN))}
rows = list(csv.DictReader(open(REPLAY)))
diff = [(r["step"], k, r[k], run[r["step"]][k]) for r in rows for k in r
        if k in run.get(r["step"], {}) and r[k] != run[r["step"]][k]]
print(f"replay rows {len(rows)} (steps {rows[0]['step'] if rows else '-'} to {rows[-1]['step'] if rows else '-'}); "
      f"cells that differ: {len(diff)}", diff[:5])
