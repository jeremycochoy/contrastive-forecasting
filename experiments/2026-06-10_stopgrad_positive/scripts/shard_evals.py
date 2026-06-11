#!/usr/bin/env python3
"""Shard a GIFT-Eval full-97 run into N parallel processes (#339 wall-clock).

Per-config computation is independent (the eval script's --config-filter +
--resume make this exact): partition the 97 configs into N time-balanced bins
(greedy, descending reference per-task seconds), launch one eval process per
bin into <out>__shard<k>/, seeding each shard with any existing partial
all_results.csv so done configs are skipped.

usage: shard_evals.py <head_run_name> <bb_file> <out_tag> <head_layers> <n_shards> <gpu_list_csv>
"""
import csv
import os
import re
import subprocess
import sys

EXP = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-10_stopgrad_positive"
REF_LOG = f"{EXP}/results/reference_eval_times_2L.txt"  # generated from #328 log
WT = "/tmp/cf-sgpos"
EVAL_SH = f"{WT}/experiments/2026-06-10_stopgrad_positive/scripts/eval_on_elisa.sh"

qn, bbf, out_tag, hl, n_shards, gpus = sys.argv[1:7]
n_shards = int(n_shards)
gpus = gpus.split(",")

# Reference per-config seconds (fallback 60s for any missing name).
times = {}
ref_log = f"{EXP}/results/reference/gift_eval_full_allt08_xftrip_nobn_enc3_qk_aon_b1024_2L/summary.txt"
seen = []
for line in open(ref_log):
    p = line.split()
    if len(p) == 4 and "/" in p[0]:
        seen.append(p[0])
ref_times_log = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-03_crossfade_triplet/results/run_eval_full_allt08_xftrip_nobn_enc3_qk_aon_b1024_2L.log"
for line in open(ref_times_log):
    m = re.search(r"\]\s+(\S+)\s+MASE=.*\((\d+\.\d+)s\)", line)
    if m:
        times[m.group(1)] = float(m.group(2))
configs = [(c, times.get(c, 60.0)) for c in seen]
assert len(configs) == 97, f"expected 97 configs, got {len(configs)}"

# Greedy balance: biggest task to the lightest bin.
bins = [{"cfgs": [], "t": 0.0} for _ in range(n_shards)]
for c, t in sorted(configs, key=lambda x: -x[1]):
    b = min(bins, key=lambda b: b["t"])
    b["cfgs"].append(c)
    b["t"] += t

main_out = f"{EXP}/results/gift_eval_full_{out_tag}_{hl}L"
partial = os.path.join(main_out, "all_results.csv")
procs = []
for k, b in enumerate(bins):
    shard_out = f"{main_out}__shard{k}"
    os.makedirs(shard_out, exist_ok=True)
    # Seed shard with the partial csv so --resume skips already-done configs.
    if os.path.exists(partial) and not os.path.exists(f"{shard_out}/all_results.csv"):
        import shutil
        shutil.copy(partial, f"{shard_out}/all_results.csv")
    regex = "^(" + "|".join(re.escape(c) for c in b["cfgs"]) + ")$"
    env = dict(os.environ)
    env["EVAL_OUT_OVERRIDE"] = shard_out
    env["EVAL_CONFIG_FILTER"] = regex
    gpu = gpus[k % len(gpus)]
    log = f"{EXP}/results/shard_{out_tag}_{hl}L_{k}.out"
    cmd = ["bash", EVAL_SH, qn, bbf, out_tag, hl, gpu]
    procs.append(subprocess.Popen(cmd, env=env, stdout=open(log, "w"),
                                  stderr=subprocess.STDOUT))
    print(f"shard {k}: {len(b['cfgs'])} cfgs, est {b['t']/60:.0f} min, gpu {gpu} -> {shard_out}")
rc = [p.wait() for p in procs]
print("shard rcs:", rc)
sys.exit(max(rc) if rc else 0)
