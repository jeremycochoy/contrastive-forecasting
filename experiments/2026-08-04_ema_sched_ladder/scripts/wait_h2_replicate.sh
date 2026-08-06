#!/bin/bash
# #393 — wait for the last replicate cell, pull it home, then let the release
# loop give the box back.
#
# arm6_v2_combab_alignT seed 20260724 (student and teacher) is the last of the
# 36 (cell, head, seed) replicate measurements, and it is the one the study
# most needs: that cell's recorded branch rests on a student move of +0.0026
# against an in-study spread of 0.002 to 0.015, so without its third seed the
# branch cannot be called signal or noise.
#
# seed-h2 is the only rented box left. Waiting on collect_results.sh's 30-min
# tick to notice the score would keep it alive for up to half an hour past the
# work, at $0.4681/h. This polls the box itself, pulls the evidence the moment
# both scores exist, and rebuilds the spread table. release_seed_boxes.sh then
# sees 4/4 home on its next pass and destroys the contract.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
RES="$EXP/results"
RUNS="${RUNS:-/home/jupyter/checkpoints_backup/cf-393}"
CELL=arm6_v2_combab_alignT
SEED=20260724
HOST=ssh1.vast.ai
PORT=17540
POLL="${POLL:-300}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

say(){ echo "[$(date -u '+%m-%d %H:%M:%SZ')] [wait-h2] $*"; }

# Rows finished across the two heads' four shards, as a progress number that
# means something: 97 configs total per head, 194 for the pair.
progress(){
  timeout 60 ssh -n "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" \
    "cat /root/cf393_runs/$CELL/eval/bb100k_{student,teacher}_s$SEED/gift/shard_*/all_results.csv 2>/dev/null \
     | grep -cv '^dataset' " 2>/dev/null
}

remote_scores(){
  timeout 60 ssh -n "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" \
    "ls /root/cf393_runs/$CELL/eval/score_bb100k_{student,teacher}_s$SEED.txt 2>/dev/null | wc -l" 2>/dev/null
}

pull(){
  timeout 300 ssh -n "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" \
    "cd /root/cf393_runs && tar czf /tmp/cf393_seed_evidence.tgz \
     \$(find $CELL -path '*_s20260*' \( -name 'score_bb*.txt' -o -name 'summary.txt' \
       -o -name '*_encoder_source.txt' -o -name 'all_results.csv' \) -type f) \
     \$(find $CELL -maxdepth 2 -name 'score_bb*_s20260*.txt' -type f) 2>/dev/null" 2>/dev/null
  timeout 300 scp -q "${SSH_OPTS[@]}" -P "$PORT" \
      "root@$HOST:/tmp/cf393_seed_evidence.tgz" /tmp/cf393_seed_evidence_h2.tgz 2>/dev/null \
    && tar xzf /tmp/cf393_seed_evidence_h2.tgz -C "$RUNS" \
    && ( cd "$RUNS" && find . \( -name 'score_bb*.txt' -o -name 'summary.txt' \
          -o -name '*_encoder_source.txt' -o -name 'all_results.csv' \) -type f \
          -exec cp -a --parents {} "$RES/eval/" \; ) 2>/dev/null \
    && return 0
  return 1
}

last=-1
while :; do
  n="$(remote_scores)"
  case "$n" in ''|*[!0-9]*) say "box unreachable — retrying"; sleep "$POLL"; continue ;; esac
  if [ "$n" -ge 2 ]; then
    say "both scores on the box — pulling evidence"
    if pull; then
      st="$(tr -d '[:space:]' < "$RUNS/$CELL/eval/score_bb100k_student_s$SEED.txt" 2>/dev/null)"
      te="$(tr -d '[:space:]' < "$RUNS/$CELL/eval/score_bb100k_teacher_s$SEED.txt" 2>/dev/null)"
      say "HOME $CELL s$SEED student=$st teacher=$te"
      python3 "$HERE/seed_spread.py" --runs "$RUNS" >> "$RES/seed_spread.log" 2>&1 \
        && say "seed_spread rebuilt: $(awk -F, 'NR>1 && $7>=3 {d++} NR>1{t++} END{printf "%d/%d rows at 3 seeds", d+0, t+0}' "$RES/seed_spread.csv")"
      say "DONE — release_seed_boxes.sh will give seed-h2 back on its next pass"
      exit 0
    fi
    say "pull failed — retrying"
  else
    p="$(progress)"
    [ "${p:-x}" != "$last" ] && say "$n/2 scored, ${p:-?}/194 configs evaluated"
    last="${p:-x}"
  fi
  sleep "$POLL"
done
