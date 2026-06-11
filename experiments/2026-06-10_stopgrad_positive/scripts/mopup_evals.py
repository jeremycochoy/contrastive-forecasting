#!/usr/bin/env python3
"""Mop-up missing configs of a sharded GIFT-Eval cell (#339).

merge_shards.py-style scan finds which of the 97 display configs are missing
from the union of <out> and <out>__shard*/all_results.csv, maps them to RAW
config names (derived in-process from the eval script — it filters on raw
names, which are partially uppercase), and runs N mop-up shard processes restricted
to exactly those raw names. usage: mopup_evals.py <head_run_name> <bb_file>
<out_tag> <head_layers> <n_shards> <gpu_list_csv>
"""
import csv
import glob
import os
import re
import subprocess
import sys

EXP = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-10_stopgrad_positive"
WT = "/tmp/cf-sgpos"
EVAL_SH = f"{WT}/experiments/2026-06-10_stopgrad_positive/scripts/eval_on_elisa.sh"
sys.path.insert(0, f"{WT}/experiments/2026-06-10_stopgrad_positive/scripts")
from shard_evals import display_to_raw  # noqa: E402

qn, bbf, out_tag, hl, n_shards, gpus = sys.argv[1:7]
n_shards = int(n_shards)
gpus = gpus.split(",")
d2r = display_to_raw()

main_out = f"{EXP}/results/gift_eval_full_{out_tag}_{hl}L"
done = set()
for path in glob.glob(f"{main_out}__shard*/all_results.csv") + [f"{main_out}/all_results.csv"]:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            done.add(row[0])
missing = sorted(set(d2r) - done)
print(f"{out_tag} {hl}L: {len(done)} done, {len(missing)} missing")
if not missing:
    sys.exit(0)

bins = [missing[i::n_shards] for i in range(n_shards)]
procs = []
for k, b in enumerate(bins):
    if not b:
        continue
    shard_out = f"{main_out}__mopup{k}"
    os.makedirs(shard_out, exist_ok=True)
    regex = "^(" + "|".join(re.escape(d2r[c]) for c in b) + ")$"
    env = dict(os.environ)
    env["EVAL_OUT_OVERRIDE"] = shard_out
    env["EVAL_CONFIG_FILTER"] = regex
    log = f"{EXP}/results/mopup_{out_tag}_{hl}L_{k}.out"
    procs.append(subprocess.Popen(["bash", EVAL_SH, qn, bbf, out_tag, hl, gpus[k % len(gpus)]],
                                  env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT))
    print(f"mopup {k}: {len(b)} cfgs, gpu {gpus[k % len(gpus)]}")
rc = [p.wait() for p in procs]
print("mopup rcs:", rc)
sys.exit(max(rc) if rc else 0)
