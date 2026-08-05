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
# Names match the second column of results/cell_claims.txt, so a per-machine
# file can be read against the cell it holds.
REMOTES=("${REMOTES[@]:-vastA ssh2.vast.ai 11448 vastB ssh4.vast.ai 13146 vastC ssh6.vast.ai 18762 vastD ssh7.vast.ai 18862 vastE ssh5.vast.ai 18856 vastF ssh1.vast.ai 18914 vastG ssh7.vast.ai 13258}")
REMOTE_EXP=/root/cf/experiments/2026-08-04_ema_sched_ladder
REMOTE_RUNS=/root/cf393_runs

mkdir -p "$RES/per_machine" "$RES/eval"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [collect] $*"; }

# A per-machine CSV is the only record of that box's scores once the box is
# released, and `ladder_all.csv` is rebuilt from these files every run — so a
# row that vanishes here vanishes from the results table, silently. Raw scp
# writes straight to the destination, so a drop mid-transfer leaves a
# truncated file where the good copy was (CLAUDE.md, after the same thing
# cost a checkpoint). Fetch to a temp path, and adopt it only if it is a
# plausible successor: these files are append-only on the box, so a copy with
# a different header or with fewer rows than the one it would replace is a
# broken transfer, not progress.
pull_csv(){  # <host> <port> <remote> <local> <label>
  local host="$1" port="$2" remote="$3" local_="$4" label="$5"
  local tmp="$local_.tmp" n_new n_old
  rm -f "$tmp"
  if ! scp -q "${SSH_OPTS[@]}" -P "$port" "root@$host:$remote" "$tmp" 2>/dev/null; then
    rm -f "$tmp"; say "  no $label yet"; return 1
  fi
  n_new=$(wc -l < "$tmp" 2>/dev/null || echo 0)
  if [ "$n_new" -lt 1 ] || ! head -n 1 "$tmp" | grep -q '^cell,'; then
    rm -f "$tmp"; say "  REJECTED $label: no header — transfer broken, keeping the old copy"
    return 1
  fi
  if [ -f "$local_" ]; then
    n_old=$(wc -l < "$local_" 2>/dev/null || echo 0)
    if [ "$n_new" -lt "$n_old" ]; then
      rm -f "$tmp"
      say "  REJECTED $label: $n_new lines against $n_old on disk — keeping the old copy"
      return 1
    fi
  fi
  mv -f "$tmp" "$local_"
}

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
    pull_csv "$host" "$port" "$REMOTE_EXP/results/$f.csv" \
             "$RES/per_machine/${f}_${name}.csv" "$name/$f"
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

# --- the eval output itself ----------------------------------------------
# A machine's ladder.csv is not the only record of a score, and it is the
# slowest and the least durable one. ladder.py appends a stop's two rows
# only once BOTH heads have returned, and the copy above needs the box to
# still be rented. Every score is computed here, so read them off elisa's
# own eval directories as one more per-machine source. merge_pooled.sh
# unions the sources and collapses identical rows, so this adds the ones
# no ladder.csv has yet and duplicates nothing.
python3 "$EXP/scripts/scores_from_evals.py" \
        --out "$RES/per_machine/ladder_evaldirs.csv" 2>&1 | sed 's/^/  /'

# --- the union -----------------------------------------------------------
# Pooling lives in its own script so it can be tested without ssh in the way;
# scripts/test_merge_pooled.sh is the guard. It pools on a key declared by
# column name, and deduplicates on the whole line so no row is ever discarded
# for a field the key does not mention.
RES="$RES" bash "$EXP/scripts/merge_pooled.sh" ladder decisions

# --- which branch stopped which cell -------------------------------------
# The pooled decisions file carries more than one row per (cell, stop) — a
# park written by the spend order, and the rule re-derived once the stop's
# second head landed — and merge_pooled.sh sorts it, so row order is no
# guide to which came last. Resolve it here, every cycle, rather than
# leaving the report to pick a row. scripts/test_stop_reason.sh is the
# guard.
python3 "$EXP/scripts/stop_reason.py" \
        --decisions "$RES/decisions_all.csv" \
        --ladder "$RES/ladder_all.csv" \
        --out "$RES/stop_reason.csv" 2>&1 | sed 's/^/  /'

# --- the result, as a picture --------------------------------------------
# Redrawn every cycle from the pooled table, so the committed plot never
# shows fewer stops than the CSV beside it.
python3 "$EXP/scripts/plot_ladder.py" --ladder "$RES/ladder_all.csv" \
        --out "$EXP/plots/ladder.png" 2>&1 \
  | grep -vE 'UserWarning|warnings\.warn|Axes3D' | sed 's/^/  /'
say "done"
