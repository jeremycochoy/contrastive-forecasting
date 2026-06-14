#!/usr/bin/env python3
"""Merge sharded GIFT-Eval outputs into the canonical eval dir (#341, cloned
from #339).

Concatenates <out>__shard*/all_results.csv (deduped by config), verifies all
97 configs are present, recomputes Relative = MASE / seasonal-naive MASE (SN
taken from the #328 reference summary — identical data, identical SN model),
and writes all_results.csv + summary.txt in the standard format analyze.py
parses. usage: merge_shards.py <out_tag> <head_layers>
"""
import csv
import glob
import math
import os
import sys

EXP = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity"
out_tag, hl = sys.argv[1], sys.argv[2]
main_out = f"{EXP}/results/gift_eval_full_{out_tag}_{hl}L"

sn = {}
ref = ("/home/jupyter/workspaces/contrastive-forecasting/experiments/"
       "2026-06-03_crossfade_triplet/results/"
       "gift_eval_full_allt08_xftrip_nobn_enc3_qk_aon_b1024_2L/summary.txt")
for line in open(ref):
    p = line.split()
    if len(p) == 4 and "/" in p[0]:
        sn[p[0]] = float(p[2])
assert len(sn) == 97

rows, header = {}, None
sources = sorted(glob.glob(f"{main_out}__shard*/all_results.csv") + glob.glob(f"{main_out}__mopup*/all_results.csv"))
if os.path.exists(f"{main_out}/all_results.csv"):
    sources.insert(0, f"{main_out}/all_results.csv")
for path in sources:
    with open(path) as f:
        r = csv.reader(f)
        h = next(r)
        header = header or h
        for row in r:
            rows.setdefault(row[0], row)
missing = sorted(set(sn) - set(rows))
if missing:
    print(f"INCOMPLETE: {len(rows)}/97 — missing {missing[:5]}{'...' if len(missing) > 5 else ''}")
    sys.exit(1)

mase_idx = header.index("eval_metrics/MASE[0.5]")
os.makedirs(main_out, exist_ok=True)
with open(f"{main_out}/all_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for c in sorted(rows):
        w.writerow(rows[c])

rels = []
with open(f"{main_out}/summary.txt", "w") as f:
    f.write("=" * 90 + "\n")
    f.write(" " * 32 + "GIFT-Eval Official Results\n")
    f.write("=" * 90 + "\n")
    f.write(f"{'Config':<48}{'MASE':>8} {'SN_MASE':>8} {'Relative':>10}\n")
    f.write("-" * 90 + "\n")
    for c in sorted(rows):
        m = float(rows[c][mase_idx])
        rel = m / sn[c]
        rels.append(rel)
        f.write(f"{c:<48}{m:>8.4f} {sn[c]:>8.4f} {rel:>10.4f}\n")
    gm = math.exp(sum(math.log(r) for r in rels) / len(rels))
    f.write("=" * 90 + "\n")
    f.write(f"Aggregate GM-Relative MASE ({len(rels)} configs): {gm:.4f}\n")
    f.write("=" * 90 + "\n")
print(f"merged {len(rows)} configs -> {main_out}  GM={gm:.4f}")
