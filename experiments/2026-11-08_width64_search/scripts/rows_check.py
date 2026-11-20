"""Health of the elisa runs of the width-64 work from their chained losses files (<name>*_losses.csv, a later file
wins a step): last step, the means of the last 400 rows, all finite, the ema_tau gap to the formula of the run's
shape, rep_w, and the largest 100-step mean of sigreg_e over the last 3,000 rows. The script skips the lines with
NUL bytes and the half-written rows. A resume replaces the rows of the dead branch from its first step."""
import os
import csv
import math
from pathlib import Path

HP = Path(os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64"))
P = "g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere"
RUNS = {  # codename -> (run name, log EMA shape)
    "1/d run": (f"{P}_align10harm_rollfix", False),  # the variant + harmonic, no SIGReg on f (17:54 Nov 14)
    "1/d run + SIGReg-F": (f"{P}_sigf_align10harm_rollfix", False),  # resumed from its 20k save (15:19 Nov 17)
}
COLS = ("loss", "l_align", "sigreg_e", "sigreg_h", "sigreg_f", "auc")


def tau(step, log):
    frac = math.log1p(step) / math.log1p(166000) if log else step / 166000
    return 0.99995 + 0.00005 * frac


def chained_rows(name):
    out = {}
    for f in sorted((HP / name / "ckpt").glob(f"{name}*_losses.csv")):
        # a line with NUL bytes: the end of a file cut by the reboot of elisa (21:56 UTC Nov 15)
        rows = list(csv.DictReader(line for line in open(f) if "\0" not in line))
        rows = [r for r in rows if None not in r.values() and r.get("step")]  # a half-written last row (a pulled copy)
        if rows:  # a resume replaces every step from its first one: the rows of the dead branch go
            first = int(float(rows[0]["step"]))
            out = {s: r for s, r in out.items() if s < first}
        for r in rows:
            out[int(float(r["step"]))] = r
    return [out[s] for s in sorted(out)]


def mean(rows, col):
    vals = [float(r[col]) for r in rows if r.get(col) not in ("", None)]
    return sum(vals) / len(vals) if vals else None


for code, (name, log) in RUNS.items():
    rows = chained_rows(name)[-3000:]
    if not rows:
        print(f"{code:18} no rows")
        continue
    last = rows[-400:]
    finite = all(math.isfinite(float(v)) for r in rows for k, v in r.items() if k != "step" and v not in ("", None))
    gap = max(abs(float(r["ema_tau"]) - tau(float(r["step"]), log)) for r in rows)
    peak = max(mean(rows[i:i + 100], "sigreg_e") for i in range(0, len(rows) - 99, 100))
    parts = " ".join(f"{c} {m:.4f}" for c in COLS if (m := mean(last, c)) is not None)
    print(f"{code:18} step {last[-1]['step']}  {parts}  finite {finite}  tau gap {gap:.1e}  "
          f"rep_w {sorted({r['rep_w'] for r in last})}  max100 sigreg_e {peak:.3f}")
