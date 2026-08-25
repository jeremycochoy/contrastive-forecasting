#!/bin/bash
# #409 — does this card's protocol match the study its reference comes from?
#
# WHY THIS CHECK EXISTS. This card runs NO control arm. Its headline is a gap
# against 1.1491, and that number comes from
# `reports/2026-08-19_ema_momentum_k32/` (#404), not from any arm here. A gap
# between two studies is only a result if the two measure the same thing.
#
# So this compares the two, item by item, and writes
# `results/reference_match.tsv`. Every row states WHERE it read each side, so a
# reader can check it without running anything.
#
# THE ONE THING IT CANNOT CHECK. #404's checkpoint root,
# `/home/jupyter/checkpoints_backup/cf-404`, is deleted. Its head seed and its
# align target therefore rest on its scripts and its report, not on a surviving
# qhead file. Those two rows say `script` and not `artefact` in the `evidence`
# column. This card's own side of both rows IS an artefact: its qhead files
# carry `_s20260722` in their names.
#
# WHAT THE TABLE SAID ON 2026-08-25. Eleven items of eleven match: the cell,
# the rollout depth, the depth reduction, the align target, the backbone stop,
# the head steps, the head seed, the head encoder, the head runner, the
# 97-config eval and the score tag.
#
# The head runner row is the strongest. It is not two settings that agree, it
# is ONE file that both studies call. The align target has the same shape: it
# rides the cell in `run_leg_k.sh`, and both studies name that cell.
#
# Usage:  bash scripts/reference_match.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

SWEEP="$(cd "$CF409_STUDY/../2026-08-19_ema_momentum_k32" && pwd)" || {
  echo "ABORT: no sweep directory beside this study" >&2; exit 2; }
OUT="${1:-$CF409_RESULTS/reference_match.tsv}"

# One value of the sweep's study.sh, read by sourcing it.
sweep_value(){  # <name>
  bash -c '. "'"$SWEEP"'/scripts/study.sh" >/dev/null 2>&1; printf "%s" "${'"$1"'}"'
}

rows=0
mismatch=0
emit(){  # <item> <this card> <the sweep> <evidence> <where>
  local verdict="match"
  [ "$2" = "$3" ] || { verdict="DIFFERS"; mismatch=$((mismatch + 1)); }
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$verdict" "$4" "$5"
  rows=$((rows + 1))
}

# The eval line both studies log when they hand the 97 configs to the CPUs.
eval_line(){  # <log glob>
  grep -hoE 'eval start \([^)]*\)' $1 2>/dev/null | sort -u | head -1
}
# The tag both studies build for a scored run: <arm>_bb<stop>_h<head>_<enc>.
tag_shape(){  # <log glob>
  grep -hoE '_bb[0-9]+k_h[0-9]+k_[a-z]+' $1 2>/dev/null | sort -u | head -1
}

{
  printf 'item\tthis_card\tthe_sweep\tverdict\tevidence\twhere\n'

  emit "cell" "$CF409_CELL" "$(sweep_value CF404_CELL)" \
    "script" "study.sh of each"
  emit "rollout depth k" "$CF409_K" "$(sweep_value CF404_K)" \
    "script" "study.sh of each"
  emit "depth reduction" "$CF409_REDUCE" "$(sweep_value CF404_REDUCE)" \
    "script" "study.sh of each"
  emit "backbone stop" "${CF409_STOPS%% *}" "$(sweep_value CF404_STOPS)" \
    "script" "study.sh of each"
  emit "head steps" "$CF409_HEAD_STEPS" "$(sweep_value CF404_HEAD_STEPS)" \
    "script" "study.sh of each"
  emit "head encoder" "$CF409_ENC" "$(sweep_value CF404_ENC)" \
    "script" "study.sh of each"

  # The head runner. Both studies call ONE script, in #373's directory. That is
  # the strongest row here: it is not two settings that agree, it is one file.
  emit "head runner" \
    "$(basename "$CF409_PARENT")/scripts/head_eval_bb.sh" \
    "$(basename "$(sweep_value CF404_PARENT)")/scripts/head_eval_bb.sh" \
    "script" "head_eval.sh of each names it"

  # The align target rides the CELL, in #373's shared runner, not either
  # study's own flags. Both name `arm6_v2_combab_alignT`, whose ALIGN_ARGS is
  # `--align-target teacher`.
  emit "align target" "$CF409_ALIGN_TARGET" \
    "$(grep -A 6 'arm6_v2_combab_alignT)' \
        "$CF409_PARENT/scripts/run_leg_k.sh" \
        | grep -oE 'align-target [a-z]+' | head -1 | cut -d' ' -f2)" \
    "script" "run_leg_k.sh, the cell both studies name"

  # The head seed. This card sets it. The sweep passes `${HEAD_SEED:-20260722}`
  # and its report's Protocol section states the same value.
  emit "head seed" "$CF409_HEAD_SEED" \
    "$(grep -hoE 'HEAD_SEED:-[0-9]+' "$SWEEP"/scripts/*.sh \
        | sed -E 's/.*:-//' | sort -u | head -1)" \
    "script" "the sweep's checkpoints are deleted"

  # What each study's own eval log says it handed to the CPUs.
  emit "eval" "$(eval_line "$CF409_RESULTS/head_*_bb40k.out")" \
    "$(eval_line "$SWEEP/results/eval_*_bb40k.out")" \
    "artefact" "the eval log of each"
  emit "score tag" "$(tag_shape "$CF409_RESULTS/head_*_bb40k.out")" \
    "$(tag_shape "$SWEEP/results/eval_*_bb40k.out")" \
    "artefact" "the eval log of each"
} >"$OUT.$$" && mv -f "$OUT.$$" "$OUT"

echo "$OUT: $rows row(s), $mismatch mismatch(es)"
[ "$mismatch" -eq 0 ] || {
  echo "WARN: the reference is NOT measured the same way — see $OUT" >&2
  exit 1; }
exit 0
