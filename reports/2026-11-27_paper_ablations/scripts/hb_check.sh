#!/bin/bash
# The health check of the ablation boxes of #482: the chk function of
# hp64/tools/hb_check_boxes.sh (#425), unchanged, on each box of $CF/boxes.tsv.
# One line per box in boxes.tsv, tab-separated:
#   <label> <ssh target, for example "-p 41065 root@141.0.85.212"> <run> <short> <box PIDs>
# For each box: the PIDs, the last train step and speed, the NaN and Traceback
# count, the HF retries, the GPU, the disk, the last line of the follower, and
# the newest head wave and eval.
#   bash hb_check.sh
CF="${CF:-/home/jupyter/cf_runs/cf-482}"
O="-o ConnectTimeout=20 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
chk() { local hp=$1 run=$2 flog=$3; shift 3; echo "== ${run:0:12}...${run: -22}"
  timeout 60 ssh -n $O $hp "ps -o pid=,etime=,args= -p $(echo $* | tr ' ' ,) | cut -c1-45; L=/workspace/hp64/$run/logs; grep -E '^\[ *[0-9]+\]' \$L/train.log | tail -n 1 | grep -o -E '^\[ *[0-9]+\]|[0-9.]+ sps' | paste -sd' '; echo \"nan/tb: \$(cat \$L/*.log | grep -c -E 'nan|NaN|Traceback') retries: \$(grep -c 'failed, try' \$L/train.log) deepest: \$(grep -o 'try [0-9]*/20' \$L/train.log | sort -t' ' -k2 -n | tail -n 1) gpu: \$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) disk: \$(df -h /workspace | tail -n 1 | awk '{print \$4}')\"; echo \"   $flog: \$(tail -n 1 /workspace/hp64/ff/results/$flog | cut -c16-105)\"; w=\$(ls -td /workspace/hp64/ff/results/waves/*/ 2>/dev/null | head -n 1); [ -n \"\$w\" ] && echo \"   wave \$(basename \$w | cut -c1-22): \$(grep -E '^\[shared\]' \$w/train.log | tail -n 1 | cut -c1-55)\"; for d in \$(ls -td /workspace/hp64/ff/heads/eval/*/ 2>/dev/null | head -n 1); do echo \"   eval \$(basename \$d | grep -o '_bb[0-9]*k'): \$(tail -n 1 \$d/eval_local.log 2>/dev/null | cut -c1-35)\"; done"; }
[ -s "$CF/boxes.tsv" ] || { echo "no boxes in $CF/boxes.tsv"; exit 0; }
while IFS=$'\t' read -r label target run short pids; do
  [ -n "$label" ] && [ "${label:0:1}" != "#" ] || continue
  echo "## $label"
  chk "$target" "$run" "follow_${short}_box.log" $pids
done < "$CF/boxes.tsv"
date -u +%T
