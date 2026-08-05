#!/bin/bash
# #393 — assemble one view of the ladder out of three machines.
#
# Usage:  WT=<checkout> bash scripts/collect_results.sh
#
# Each machine runs its own `ladder.py`, so each keeps its own
# `results/ladder.csv` and `results/decisions.csv` covering only the cells
# it was given. The cell sets are disjoint, so the union is the experiment.
#
# The per-machine files are copied in verbatim, never merged in place. A
# live driver appends to elisa's `ladder.csv` between any two lines of this
# script, and rewriting a file that another process is appending to is how
# a row goes missing. `ladder_all.csv` and `decisions_all.csv` are derived
# files, written fresh each run; the inputs are left alone.
#
# Also pulls the small text artefacts that are the actual evidence behind
# each number — the per-stop score files, the GIFT-Eval summaries, and the
# encoder-source marker next to every head. Checkpoints are not pulled:
# they are 5 MB each and live on the durable roots and in the sync trees.
set -uo pipefail

WT="${WT:?WT must be the absolute path of the checkout}"
EXP="$WT/experiments/2026-08-04_ema_sched_ladder"
RES="$EXP/results"
ELISA_RUNS="${RUNS:-/home/jupyter/checkpoints_backup/cf-393}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=15)
# <name> <host> <port>, one per vast.ai box. Empty is fine: elisa alone.
REMOTES=("${REMOTES[@]:-vastA ssh2.vast.ai 11448 vastB ssh4.vast.ai 13146}")
REMOTE_EXP=/root/cf/experiments/2026-08-04_ema_sched_ladder
REMOTE_RUNS=/root/cf393_runs

mkdir -p "$RES/per_machine" "$RES/eval"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [collect] $*"; }

# --- elisa ---------------------------------------------------------------
for f in ladder decisions; do
  [ -f "$RES/$f.csv" ] && cp -a "$RES/$f.csv" "$RES/per_machine/${f}_elisa.csv"
done
# score files, GIFT-Eval summaries, encoder markers — the evidence, not the
# checkpoints.
if [ -d "$ELISA_RUNS" ]; then
  ( cd "$ELISA_RUNS" && find . \( -name 'score_bb*.txt' -o -name 'summary.txt' \
      -o -name '*_encoder_source.txt' -o -name 'all_results.csv' \) -type f \
      -exec cp -a --parents {} "$RES/eval/" \; ) 2>/dev/null
fi

# --- each vast.ai box ----------------------------------------------------
set -- ${REMOTES[@]}
while [ $# -ge 3 ]; do
  name="$1"; host="$2"; port="$3"; shift 3
  say "$name ($host:$port)"
  for f in ladder decisions; do
    scp -q "${SSH_OPTS[@]}" -P "$port" \
        "root@$host:$REMOTE_EXP/results/$f.csv" \
        "$RES/per_machine/${f}_${name}.csv" 2>/dev/null \
      || say "  no $f.csv yet on $name"
  done
  # One tarball rather than one scp per file: a stop produces a summary, a
  # marker and a score file, and by the end there are hundreds of them.
  if ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" \
       "cd $REMOTE_RUNS 2>/dev/null && tar czf /tmp/cf393_evidence.tgz \
        \$(find . \( -name 'score_bb*.txt' -o -name 'summary.txt' \
          -o -name '*_encoder_source.txt' -o -name 'all_results.csv' \) -type f) \
        2>/dev/null" 2>/dev/null; then
    scp -q "${SSH_OPTS[@]}" -P "$port" \
        "root@$host:/tmp/cf393_evidence.tgz" "/tmp/cf393_evidence_${name}.tgz" \
      && tar xzf "/tmp/cf393_evidence_${name}.tgz" -C "$RES/eval" \
      && say "  evidence unpacked"
  fi
  # Per-cell run logs, so a reader can see the training itself, not just
  # the scores it produced.
  scp -q "${SSH_OPTS[@]}" -P "$port" \
      "root@$host:$REMOTE_EXP/results/*.log" "$RES/" 2>/dev/null
done

# --- the union -----------------------------------------------------------
# Header from whichever file has one; rows from all of them, deduplicated.
# Sorting is by cell then stop then head, so the file reads as the ladder
# rather than as the order three machines happened to finish in.
merge(){  # <basename>
  local out="$RES/${1}_all.csv" hdr="" first=1
  : > "$out.tmp"
  for f in "$RES/per_machine/${1}_"*.csv; do
    [ -f "$f" ] || continue
    [ $first -eq 1 ] && { hdr=$(head -n 1 "$f"); first=0; }
    [ "$(head -n 1 "$f")" = "$hdr" ] || {
      echo "ABORT: $f has a different header than the others" >&2; return 4; }
    tail -n +2 "$f" >> "$out.tmp"
  done
  [ -n "$hdr" ] || { rm -f "$out.tmp"; say "no ${1} rows yet"; return 0; }
  { printf '%s\n' "$hdr"; sort -u -t, -k1,1 -k4,4n "$out.tmp"; } > "$out"
  rm -f "$out.tmp"
  say "$(basename "$out"): $(( $(wc -l < "$out") - 1 )) rows"
}
merge ladder
merge decisions
say "done"
