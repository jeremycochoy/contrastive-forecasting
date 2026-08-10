#!/bin/bash
# #373 — rebuild every figure and table from whatever results exist.
#
# Usage: bash make_report_assets.sh [git checkout root]
#
# This is the study's single rebuild entry point. It re-derives everything
# from the per-config eval CSVs, the losses CSVs and the checkpoints on
# disk, so running it twice gives the same answer and running it while a
# queue is still going gives the answer for whatever has finished.
#
# Idempotent and partial-tolerant: a run that has not finished is skipped
# with a line saying so, rather than taking the whole rebuild down.
#
# It holds no paths of its own. `runs.py` says which runs exist and
# `find_artefacts.py` says where each one's artefacts landed, because the
# study trained on two kinds of machine and the two wrote to two trees.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_ROOT="${1:-/tmp/contrastive-forecasting-373}"
DST="$GIT_ROOT/reports/2026-08-08_rollout_depth"
RES="$DST/results"; PLOTS="$DST/plots"
BATCH="$GIT_ROOT/reports/2026-07-21_split_pred_rep_small/plots/_latent_movement_batch.pt"
mkdir -p "$RES" "$PLOTS"
export PYTHONPATH="${PYTHONPATH:-}:$GIT_ROOT"
say(){ echo "[assets] $*"; }
run(){ python3 "$@" 2>&1 | grep -Ev "UserWarning|warnings.warn" | sed 's/^/  /'; }

# ---- 1. splits: horizon term and domain, per finished eval -----------------
stops=()
for d in "$RES"/eval/*/; do
  [ -f "$d/all_results.csv" ] || continue
  n=$(( $(wc -l <"$d/all_results.csv") - 1 ))
  if [ "$n" -ne 97 ]; then
    say "skip $(basename "$d"): $n configs, want 97"; continue
  fi
  stops+=(--stop "$(basename "$d")=$d/all_results.csv")
done
if [ "${#stops[@]}" -gt 0 ]; then
  run "$HERE/split_scores.py" "${stops[@]}" --out "$RES/splits.csv"
else
  say "no finished eval yet — no splits, no score figures"
fi

# ---- 2. paired dataset-cluster bootstrap -----------------------------------
# Both arms of a pair are evaluated on the same 97 configs, so the delta is
# paired per config; the resampling unit is the DATASET, because
# `<ds>/short`, `/medium` and `/long` are three configs of one series and are
# not independent draws.
#
# `runs.py` emits every depth pair. The three rows after it are not depth
# pairs and are listed here on purpose: two measure the backbone seed at a
# fixed depth, one measures the re-weighting control.
rm -f "$RES/bootstrap.csv"
boot(){ # <label> <baseline tag> <compared tag>
  local a="$RES/eval/$2/all_results.csv" b="$RES/eval/$3/all_results.csv"
  [ -s "$a" ] && [ -s "$b" ] || return 0
  run "$HERE/paired_bootstrap.py" --k0 "$a" --k3 "$b" --label "$1" \
      --out "$RES/bootstrap.csv"
}
while IFS=$'\t' read -r label base cmp; do
  [ -n "${label:-}" ] && boot "$label" "$base" "$cmp"
done < <(python3 "$HERE/find_artefacts.py" --what pairs --results "$RES")
[ -s "$RES/bootstrap.csv" ] && say "bootstrap -> $RES/bootstrap.csv"

# ---- 3. score figures ------------------------------------------------------
if [ -f "$RES/splits.csv" ]; then
  run "$HERE/plot_depth_response.py" --splits "$RES/splits.csv" \
      --out "$PLOTS/depth_response.png"
  run "$HERE/plot_b5_backbones.py" --splits "$RES/splits.csv" \
      --out "$PLOTS/b5_backbones.png"
  run "$HERE/plot_a3_depth.py" --splits "$RES/splits.csv" \
      --out "$PLOTS/a3_depth.png"
  run "$HERE/plot_encoder_delta.py" --splits "$RES/splits.csv" \
      --out "$PLOTS/encoder_delta.png"
  for head in student teacher; do
    run "$HERE/plot_horizon_split.py" --splits "$RES/splits.csv" \
        --head "$head" --out "$PLOTS/horizon_split_${head}.png"
    run "$HERE/plot_domain_radar.py" --splits "$RES/splits.csv" \
        --head "$head" --out "$PLOTS/domain_radar_${head}.png"
  done
fi
run "$HERE/plot_reproduction.py" --results "$RES" \
    --out "$PLOTS/reproduction.png"

# ---- 4. training curves ----------------------------------------------------
mapfile -t curves < <(python3 "$HERE/find_artefacts.py" --what curves)
if [ "${#curves[@]}" -gt 0 ]; then
  run "$HERE/plot_cos_err_depth.py" "${curves[@]}" \
      --out "$PLOTS/cos_err_depth.png"
  run "$HERE/plot_train_curves.py" "${curves[@]}" --out-dir "$PLOTS"
else
  say "no losses CSV found — no training-curve figures"
fi

# ---- 5. rollout fidelity ---------------------------------------------------
# Both of the next two steps load checkpoints onto a GPU. elisa's two cards
# are shared with other sessions, so "out of memory" is the normal failure
# here, not a bug. The models are d_model=64: the CPU run is slower but it
# finishes, and it gives the same numbers. Try the card, fall back.
gpu_or_cpu(){ # <script> <args...>
  local out; out="$(python3 "$@" 2>&1)"
  if grep -qiE "CUDA error|out of memory|AcceleratorError" <<<"$out"; then
    say "GPU busy; re-running $(basename "$1") on the CPU"
    out="$(python3 "$@" --device cpu 2>&1)"
  fi
  grep -Ev "UserWarning|warnings.warn" <<<"$out" | sed "s/^/  /"
}
mapfile -t ckpts < <(python3 "$HERE/find_artefacts.py" --what ckpt)
if [ "${#ckpts[@]}" -gt 0 ] && [ -f "$BATCH" ]; then
  gpu_or_cpu "$HERE/rollout_fidelity.py" "${ckpts[@]}" --batch "$BATCH" \
      --out "$RES/rollout_fidelity.csv"
  [ -f "$RES/rollout_fidelity.csv" ] && \
    run "$HERE/plot_rollout_fidelity.py" --csv "$RES/rollout_fidelity.csv" \
        --out "$PLOTS/rollout_fidelity.png"
else
  say "no bb40k checkpoint or no diagnostic batch — no fidelity figure"
fi

# ---- 6. latent movement, between each run's 20k and 40k checkpoints --------
mapfile -t mv < <(python3 "$HERE/find_artefacts.py" --what ckptdir)
if [ "${#mv[@]}" -gt 0 ] && [ -f "$BATCH" ]; then
  gpu_or_cpu "$HERE/plot_latent_movement.py" "${mv[@]}" --batch "$BATCH" \
      --out-csv "$RES/latent_movement.csv" \
      --out "$PLOTS/latent_movement.png"
else
  say "fewer than two periodic checkpoints per run — no latent-movement figure"
fi

# ---- 7. step time of the real runs, and who else held the card -------------
# A median over a run's timing windows is a cost of the depth only if the
# card was that run's alone. elisa ran two of this study's backbones at a
# time and trained heads beside them, so most runs fail that test. The
# driver logs say which; the step-time table then publishes a number only
# where the run passes.
queues=()
for q in "$DST"/sync/*/queue.log; do
  [ -f "$q" ] && queues+=(--queue "$(basename "$(dirname "$q")")=$q")
done
run "$HERE/run_provenance.py" --driver "$RES/gaps_driver.log" \
    "${queues[@]}" --out "$RES/run_provenance.csv"
mapfile -t logs < <(python3 "$HERE/find_artefacts.py" --what logs --results "$RES")
if [ "${#logs[@]}" -gt 0 ]; then
  run "$HERE/steptime_provenance.py" "${logs[@]}" \
      --provenance "$RES/run_provenance.csv" --out "$RES/steptime_solo.csv"
else
  say "no trainer log found — no per-run step-time table"
fi

# ---- 8. where each k = 3 lands on the published k = 0 trajectory -----------
run "$HERE/plot_ladder.py" --results "$RES" --out "$PLOTS/ladder.png"

# ---- 9. the tables ---------------------------------------------------------
run "$HERE/tables.py" --results "$RES" --out "$RES/scores.md" \
    --inject "$DST/rollout_depth.md"

say "-> $PLOTS"
ls -1 "$PLOTS" 2>/dev/null | sed 's/^/  /'
