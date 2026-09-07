#!/bin/bash
# #412 — one status line of the whole card, for a reader and for a log.
#
# It reads the durable artefacts, never a lane's stdout: the losses CSV under
# the checkpoint root gives the step, `results/score_*.txt` gives the score,
# and `nvidia-smi` gives the card. A lane that died still reports its last
# step here.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

printf '===== %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')"
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader \
  | sed 's/^/gpu /'
for arm in $CF412_ARMS; do
  # The FURTHEST leg, not the 40,000-step one. An arm that climbs writes into
  # `leg_100k` while its `leg_40k` CSV sits finished, and a status that read
  # the first would report a climb that has not moved for hours.
  csv="$(cf412_furthest_losses_csv "$arm")"
  step="-"; auc="-"; rw="-"
  if [ -n "$csv" ] && [ -s "$csv" ]; then
    read -r step auc rw <<<"$(awk -F',' '
      NR==1 { for (i=1;i<=NF;i++) c[$i]=i; next }
      { s=$c["step"]; a=$c["auc"]; w=$c["rep_w"] }
      END { print s, a, w }' "$csv")"
  fi
  # One score for each stop this arm holds, so a climb shows its whole track.
  score=""
  for stop in $CF412_STOPS; do
    v="$(cat "$(cf412_score_file "$arm" "$stop")" 2>/dev/null | tr -d ' \n')"
    [ -n "$v" ] && score="$score $(( stop / 1000 ))k=$v"
  done
  note=""
  [ -f "$(cf412_collapse_file "$arm")" ] && note=" COLLAPSED"
  printf '%-16s step %-7s auc %-8.8s rep_w %-5s score %s%s\n' \
    "$arm" "${step:--}" "${auc:--}" "${rw:--}" "${score:- -}" "$note"
done
# The bracket keeps this script's own command line out of every count, and
# `pgrep -c` exits 1 on a count of zero, so the status is not read.
n(){ local c; c="$(pgrep -c -f "$1" 2>/dev/null)"; printf '%s' "${c:-0}"; }
# A lane runs `scripts/phase1.sh` OR `run_snapshot/phase1.sh`, the frozen copy.
# A pattern that names `scripts/` alone counts 2 lanes where 3 drive work, and
# a reader takes the low count for a lane that died. A bare `phase1.sh`
# over-counts instead: an agent session carries the string in its own command
# line. So the lane count matches the WHOLE command line, through `pgrep -x`.
nx(){ local c; c="$(pgrep -c -x -f "$1" 2>/dev/null)"; printf '%s' "${c:-0}"; }
printf 'lanes %s   trainers %s   heads %s   evals %s\n' \
  "$(nx 'bash [^ ]*phase1\.sh')" "$(n '[f]req-embedding/scripts/train.py')" \
  "$(n '[t]rain_forecasting_head.py')" "$(n '[e]val_gift_eval_official.py')"
