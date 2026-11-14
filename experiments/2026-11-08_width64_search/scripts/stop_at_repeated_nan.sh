#!/bin/bash
# Watches a run after a NaN and its restart from its last save (written for the variant, red without SIGReg-F,
# after its NaN at step 12325 and its restart from the 10k save).
# rc 0: its new branch passed PASS_STEP with finite rows (the NaN did not come back);
# rc 2: a second NaN before the next save (NEXT_SAVE): the next start would replay the same steps from 10k, so it
#       writes the stop file (the supervisor checks it before each start and ends; nothing is killed);
# rc 4: a second NaN after the next save (the next start resumes from a later save: no stop file);
# rc 3: the supervisor ended, or the polls are used.
# Usage: stop_at_repeated_nan.sh <run dir> <run name> <supervisor pid> [PASS_STEP] [NEXT_SAVE] [BRANCH]
# BRANCH: the resume branch to watch (_r3: only its files count; default every branch _r<n>).
V=$1; N=$2; SUP=$3; PASS_STEP=${4:-12800}; NEXT_SAVE=${5:-15000}; BR=${6:-_r[0-9]*}
echo "$(date -u +%T) start: pass at $PASS_STEP, stop file for a NaN before $NEXT_SAVE"
for poll in $(seq 1 ${POLLS:-230}); do
  emerg=$(ls "$V/ckpt/${N}"${BR}_EMERGENCY_*.pth 2>/dev/null | grep -v _optimizer | head -1)
  if [ -n "$emerg" ]; then
    step=${emerg##*_EMERGENCY_}; step=${step%.pth}
    if [ "$step" -lt "$NEXT_SAVE" ]; then
      echo "$(date -u +%FT%TZ) second NaN at step $step, before the $NEXT_SAVE save: the next start would replay the same steps; stop file written by stop_variant_at_second_nan.sh (the user decides the fix)" > "$V/stop"
      echo "$(date -u +%T) poll $poll: second NaN at $step: stop file written"; exit 2
    fi
    echo "$(date -u +%T) poll $poll: second NaN at $step, after the $NEXT_SAVE save: no stop file"; exit 4
  fi
  last=$(ls "$V/ckpt/${N}"${BR}_losses.csv 2>/dev/null | tail -1)
  if [ -n "$last" ]; then
    if python3 - "$last" "$PASS_STEP" <<'EOF'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1])))
ok = rows and int(float(rows[-1]["step"])) >= int(sys.argv[2]) and all(
    math.isfinite(float(v)) for r in rows[-3000:] for k, v in r.items() if k != "step" and v not in ("", None))
sys.exit(0 if ok else 1)
EOF
    then echo "$(date -u +%T) poll $poll: passed $PASS_STEP with finite rows"; exit 0; fi
  fi
  kill -0 "$SUP" 2>/dev/null || { echo "$(date -u +%T) poll $poll: supervisor $SUP ended"; exit 3; }
  sleep 30
done
echo "$(date -u +%T) poll budget used"; exit 3
