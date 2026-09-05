#!/bin/bash
# #409 — `results/RUN_STATE.md`, the one file a session reads first.
#
# WHY THIS FILE EXISTS AS A SCRIPT. `launch.sh` used to hold the writer inline
# and call it every 30 minutes. The launcher died in the Hub outage of 08-23 at
# 19:11, so the file froze with three arms of the six that later scored, and it
# called `dec_m099_fix_r2` an error where `results/auc_verdicts.tsv` calls it
# held. A shared results directory then carried two answers to one question.
#
# So the writer lives here. `launch.sh` sources it, and anyone can refresh the
# file from the artefacts on disk after the run:
#
#   bash scripts/run_state.sh "done — 7 backbones spent, 6 scored"
#
# Every number it prints comes from a file: `results/scores.csv`,
# `results/auc_verdicts.tsv`, the losses CSVs and the checkpoints. It states
# nothing of its own.
set -uo pipefail

CF409_STATE_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Sourced from `launch.sh`, which has already sourced `study.sh`.
[ -n "${CF409_ROOT:-}" ] || . "$CF409_STATE_HERE/study.sh"

# The last step each arm's losses CSVs reach, over every file that arm wrote.
# A leg that started again APPENDS to the CSV it opened, so the last ROW is not
# the last STEP: `dec_m080_r200` holds 59,900 rows over 40,000 steps.
cf409_reached(){  # <arm>
  local stop csv
  for stop in $CF409_STOPS; do
    while read -r csv; do
      [ -n "$csv" ] && [ -f "$csv" ] && cat "$csv"
    done < <(cf409_losses_csvs "${1:?arm}" "$stop")
  done | awk -F, 'NR>1 && $1+0>m { m=$1+0 } END { print m+0 }'
}

cf409_run_state(){  # <state file> <note>
  local state="${1:?state file}" note="${2:-}" a
  local arms="${ARMS:-$CF409_ARMS}"
  # The first stop of the study. The `reaches` and the `score` columns
  # both read it, so an arm carried to a second stop cannot put that
  # stop's score beside the first stop's momentum.
  local stop="${CF409_STOPS%% *}"
  { echo "# #409 run state — the L_rep weight decay at k = 32"
    echo
    echo "- updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- note: $note"
    echo "- cell: \`$CF409_CELL\`, k = $CF409_K, reduce \`$CF409_REDUCE\`," \
         "target \`$CF409_ALIGN_TARGET\`"
    echo "- decay: $CF409_REP_W_START to $CF409_REP_W_END at the arm's" \
         "ramp, which is column 5 of \`scripts/arms.tsv\`."
    echo "- axes: the EMA schedule and the decay ramp. No control arm: the" \
         "sweep scored these schedules with no decay, in" \
         "\`reports/2026-08-19_ema_momentum_k32/\`."
    echo "- arms: $arms"
    echo "- cards: ${GPUS:--}, launcher pid ${CF409_LAUNCHER_PID:--}"
    echo "- root: \`$CF409_ROOT\`"
    echo "- artefacts: elisa holds them all, and no sync loop runs." \
         "See \`notes/artefacts.md\`."
    echo
    echo "## The arms, and what each one reached"
    echo
    echo '```'
    printf '%-18s %-22s %-8s %-7s %-9s %-8s %s\n' \
      arm schedule reaches ramp seed reached score
    for a in $arms; do
      printf '%-18s %-22s %-8s %-7s %-9s %-8s %s\n' "$a" \
        "$(cf409_ema_label "$a")" \
        "$(cf409_momentum_at "$a" "$stop")" \
        "$(cf409_decay_ramp_of "$a")" \
        "$(cf409_seed "$a")" \
        "$(cf409_reached "$a")" \
        "$(awk -F, -v a="$a" -v s="$stop" '$1==a && $11==s{print $NF}' \
             "$CF409_RESULTS/scores.csv" 2>/dev/null | tail -1)"
    done
    echo '```'
    echo
    echo "\`reached\` is the last step that arm's losses CSVs hold." \
         "0 means the arm never wrote a step." \
         "\`score\` is the score at the $stop-step stop." \
         "The Scores section below holds every stop."
    echo
    echo "## Scores"
    echo
    echo '```'
    cat "$CF409_RESULTS/scores.csv" 2>/dev/null || echo "(none yet)"
    echo '```'
    echo
    echo "## Contrastive AUC"
    echo
    echo '```'
    cat "$CF409_RESULTS/auc_verdicts.tsv" 2>/dev/null || echo "(none yet)"
    echo '```'
    echo
    echo "One row per losses CSV, not per arm. A leg re-fired after a crash" \
         "resumes under a \`_rN\` name and writes a second file, and the AUC" \
         "gate reads a file."
    echo
    echo "## Backbones on disk"
    echo
    echo '```'
    ls -1 "$CF409_ROOT"/*/*/leg_*k/*k.pth 2>/dev/null \
      | grep -v optimizer | sed "s#$CF409_ROOT/##" || echo "(none yet)"
    echo '```'
  } >"$state.tmp" && mv -f "$state.tmp" "$state"
}

# Run it directly to refresh the file from the artefacts on disk.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  mkdir -p "$CF409_RESULTS"
  cf409_run_state "$CF409_RESULTS/RUN_STATE.md" "${1:-refreshed by hand}"
  echo "$CF409_RESULTS/RUN_STATE.md"
fi
