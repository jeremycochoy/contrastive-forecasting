#!/bin/bash
# #412 pass 4 — one status block for the five legs of the two decay arms.
#
# Pass 4 asks ONE question: does a decay arm at 5.6e-4 get BETTER with a longer
# stop? Every arm of this card so far gets worse. So each leg needs a backbone
# AND a score, and the block prints both.
#
# Columns: the leg, the last step its losses CSV holds, the backbone, the
# score, and whether a trainer or a head runs for it now.
#
# Usage:  bash scripts/pass4_status.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
. "$HERE/pass4_lib.sh"

LEGS="${CF412_PASS4_LEGS:-k3_r100_09_lr56_dec:40000 k3_r100_09_lr56_dec:100000 k3_r100_09_lr56_dec:200000 k3_r100_09_lr56_dec10k:40000 k3_r100_09_lr56_dec10k:100000}"

# The last step one leg's losses CSV holds, or 0. A leg that runs writes into
# its own `leg_<N>k` directory, so this reads the leg, not the arm.
cf412_leg_step(){  # <arm> <stop>
  local f
  f="$(ls -t "$(cf412_leg_dir "$1" "$2")"/*_losses.csv 2>/dev/null | head -1)"
  [ -n "$f" ] || { echo 0; return 0; }
  tail -1 "$f" | cut -d, -f1 | grep -E '^[0-9]+$' || echo 0
}

# The last AUC one leg's losses CSV holds. The guard ranks nothing on this
# axis. It says one thing only: did the run keep the contrastive task?
cf412_leg_auc(){  # <arm> <stop>
  local f
  f="$(ls -t "$(cf412_leg_dir "$1" "$2")"/*_losses.csv 2>/dev/null | head -1)"
  [ -n "$f" ] || { echo "-"; return 0; }
  python3 - "$f" <<'PY' 2>/dev/null || echo "-"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
col = next((c for c in (rows[0] if rows else {}) if c.strip() == "auc"), None)
vals = [r[col] for r in rows if col and r.get(col) not in (None, "")]
print(f"{float(vals[-1]):.4f}" if vals else "-")
PY
}


printf '%-34s %-8s %8s %4s %8s %6s %s\n' \
  LEG STOP STEP BB SCORE AUC RUNNING
for leg in $LEGS; do
  arm="${leg%%:*}"; stop="${leg##*:}"
  bb="$(cf412_bb_ckpt "$arm" "$stop")"
  sf="$(cf412_score_file "$arm" "$stop")"
  score="-"; [ -s "$sf" ] && score="$(tr -d ' \n' <"$sf")"
  running=""
  cf412_trainer_pid "$arm" "$stop" >/dev/null && running="trainer"
  cf412_head_pid "$arm" "$stop" >/dev/null && running="${running:+$running+}head"
  [ -f "$(cf412_collapse_file "$arm")" ] && running="${running:-}COLLAPSED"
  printf '%-34s %-8s %8s %4s %8s %6s %s\n' \
    "$arm" "$stop" "$(cf412_leg_step "$arm" "$stop")" \
    "$([ -n "$bb" ] && [ -f "$bb" ] && echo yes || echo no)" \
    "$score" "$(cf412_leg_auc "$arm" "$stop")" "${running:--}"
done

echo
echo "lanes:  $(cf412_lanes_running) running"
echo "heads:  $(cf412_heads_running) running"
echo "sweep:  $(pgrep -fc 'head_sweep\.sh') running"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader 2>/dev/null | sed 's/^/gpu:    /'
