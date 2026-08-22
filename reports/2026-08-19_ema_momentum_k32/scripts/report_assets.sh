#!/bin/bash
# #404 — put the raw artefacts of the study into the repository.
#
# The checkpoints stay out of git: a backbone is 5 MB, an optimizer 5.8 MB,
# and the study holds twelve of each. They live in the sync tree and in the
# checkpoint store. Everything a reader has to audit comes here.
#
#   results/eval/<tag>/     the eval's own 97-config CSV, its summary, its log
#                           and the backbone it scored. #373's layout.
#   curves/box_a/           each arm's losses CSV, downsampled: every step
#                           below 1000, then every 20th. The raw file is
#                           42 MB per arm and carries 33 depth columns.
#   results/                the per-arm trainer log, the attention amplitude
#                           and the latent drift, as written.
#
# Usage:  bash scripts/report_assets.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
. "$HERE/study.sh"
cf404_use_root "$CF404_SYNC_ROOT"

SRC_RESULTS="${SRC_RESULTS:-$CF404_SYNC_DIR/results}"
CURVES="$CF404_STUDY/curves/$CF404_BOX_LABEL"
EVERY="${EVERY:-20}"
HEAD_ROWS="${HEAD_ROWS:-1000}"
STOP="${STOP:-$CF404_STOPS}"
mkdir -p "$CURVES" "$CF404_RESULTS/eval"

say(){ echo "[#404 assets] $*"; }

for arm in $CF404_ARMS; do
  tag="$(cf404_tag "$arm" "$STOP" "$CF404_HEAD_STEPS")"
  leg="$(cf404_leg_dir "$arm" "$STOP")"
  name="$(cf404_run_name "$arm")"

  # ---- the eval's own output --------------------------------------------
  src="$(cf404_eval_dir "$arm" "$tag")"
  dst="$CF404_RESULTS/eval/$tag"
  mkdir -p "$dst"
  for f in gift/all_results.csv gift/summary.txt eval_local.log stop.log; do
    [ -f "$src/$f" ] || continue
    cp -f "$src/$f" "$dst/$(basename "$f")"
    say "$tag <- $(basename "$f") $(wc -c <"$dst/$(basename "$f")") B"
  done
  # Which backbone the eval scored. #373 commits this beside the CSV so a
  # reader does not have to trust the tag.
  bb="$(cf404_bb_ckpt "$arm" "$STOP")"
  [ -n "$bb" ] && printf '%s\n%s bytes\n' "$(basename "$bb")" "$(wc -c <"$bb")" \
    >"$dst/backbone.txt"

  # ---- the trainer's own tables -----------------------------------------
  for f in "${name}_attn_amplitude.csv" "${name}_latent_drift.csv"; do
    [ -f "$leg/$f" ] && cp -f "$leg/$f" "$CF404_RESULTS/$f"
  done
  [ -f "$SRC_RESULTS/run_${name}.log" ] && \
    cp -f "$SRC_RESULTS/run_${name}.log" "$CF404_RESULTS/run_${name}.log"

  # ---- the loss curve, downsampled --------------------------------------
  csv="$leg/${name}_losses.csv"
  [ -f "$csv" ] || { say "$arm has no losses CSV"; continue; }
  awk -F, -v every="$EVERY" -v head_rows="$HEAD_ROWS" '
    NR == 1 { print; for (i = 1; i <= NF; i++) if ($i == "step") sc = i; next }
    { s = (sc ? $sc : NR - 1) + 0
      if (s < head_rows || s % every == 0) print }
  ' "$csv" >"$CURVES/${name}_losses.csv"
  say "$arm curve $(wc -l <"$CURVES/${name}_losses.csv") rows," \
      "$(wc -c <"$CURVES/${name}_losses.csv") B (raw $(wc -c <"$csv") B)"
done

say "results/eval $(du -sh "$CF404_RESULTS/eval" | cut -f1), curves $(du -sh "$CURVES" | cut -f1)"
