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
# study trained on two kinds of machine and the two wrote to two trees. It
# reads the eval outputs and trainer logs under `results/`, and every
# backbone's losses CSV under `sync/<box>/` for the runs pulled off a rented
# box and `curves/elisa/` for the runs that trained on elisa. The curves are
# downsampled to every step below 1000 and every 20th after it.
#
# Two figures need more than the repository holds. `rollout_fidelity.png`
# and `latent_movement.png` load backbone checkpoints, which are 80 MB each
# and stay out of git. Their `results/*.csv` are committed, so the numbers
# are auditable and only the re-derivation needs the checkpoint store.
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

# ---- 1b. the same splits on the PUBLISHED k = 0 side -----------------------
# The per-domain and per-horizon questions need a k = 0 polygon, and the three
# parent reports print only the 97-config aggregate. `parent_splits.py` splits
# their own per-config CSVs the same way, and accepts one only after it
# reproduces the number that parent printed.
run "$HERE/parent_splits.py" --results "$RES"

# ---- 2. paired dataset-cluster bootstrap -----------------------------------
# Both arms of a pair are evaluated on the same 97 configs, so the delta is
# paired per config; the resampling unit is the DATASET, because
# `<ds>/short`, `/medium` and `/long` are three configs of one series and are
# not independent draws.
#
# `find_artefacts.py --what pairs` names every comparison: the depth pairs
# from the registry, the retraining pairs of any cell trained more than
# once — each labelled by what it changes, the seed or the machine or both —
# and the re-weighting control.
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
      --bootstrap "$RES/bootstrap.csv" --out "$PLOTS/depth_response.png"
  run "$HERE/plot_b5_backbones.py" --splits "$RES/splits.csv" \
      --out "$PLOTS/b5_backbones.png"
  run "$HERE/plot_a3_depth.py" --splits "$RES/splits.csv" \
      --out "$PLOTS/a3_depth.png"
  # The two review-gap controls. Each needs three finished evals, so each
  # exits with a message of its own until they are all in hand.
  run "$HERE/plot_b1_alignx4.py" --splits "$RES/splits.csv" \
      --bootstrap "$RES/bootstrap.csv" --out "$PLOTS/b1_alignx4.png"
  run "$HERE/plot_encoder_delta.py" --splits "$RES/splits.csv" \
      --results "$RES" --out "$PLOTS/encoder_delta.png"
  for head in student teacher; do
    run "$HERE/plot_horizon_split.py" --splits "$RES/splits.csv" \
        --head "$head" --out "$PLOTS/horizon_split_${head}.png"
    run "$HERE/plot_domain_radar.py" --splits "$RES/splits.csv" \
        --splits-k0 "$RES/splits_k0.csv" \
        --head "$head" --out "$PLOTS/domain_radar_${head}.png"
  done
  # The per-family table the radar's numbers come from, and the report's
  # DOMAIN block.
  run "$HERE/domain_table.py" --results "$RES" --inject "$DST/rollout_depth.md"
fi
run "$HERE/plot_reproduction.py" --results "$RES" \
    --out "$PLOTS/reproduction.png"
run "$HERE/plot_a3_reseed.py" --results "$RES" \
    --out "$PLOTS/a3_reseed.png"
# Every student/teacher gap in the grid, largest first. The reseed section
# reads its top three rows, so the table it reads is rebuilt beside it.
run "$HERE/gap6_head_gap.py" --results "$RES" --out "$RES/head_gap.tsv"

# ---- 4. training curves ----------------------------------------------------
mapfile -t curves < <(python3 "$HERE/find_artefacts.py" --what curves)
if [ "${#curves[@]}" -gt 0 ]; then
  run "$HERE/plot_cos_err_depth.py" "${curves[@]}" \
      --out "$PLOTS/cos_err_depth.png"
  run "$HERE/plot_train_curves.py" "${curves[@]}" --out-dir "$PLOTS"
  # The sign the report reads off cos_err_depth.png, as a number, over four
  # end-of-run windows. An arm whose sign depends on the window has none.
  run "$HERE/depth0_gap.py" "${curves[@]}" --out "$RES/depth0_gap.csv"
  # The card's collapse watch, as numbers. It also reports which of the
  # quantities the card names no run logged.
  run "$HERE/collapse_watch.py" "${curves[@]}" \
      --out "$RES/collapse_watch.csv"
else
  say "no losses CSV found — no training-curve figures"
fi

# ---- 4b. the EMA schedule, as each group logged it -------------------------
# This one walks the committed curve trees itself rather than taking the
# five diagnostic arms `find_artefacts.py` resolves: the figure is about the
# regime all 14 cells trained under, so it draws every backbone leg.
run "$HERE/plot_alpha_schedule.py" --curves "$DST/curves" --curves "$DST/sync" \
    --out "$PLOTS/alpha_schedule.png"

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
elif [ -s "$RES/latent_movement.csv" ]; then
  say "no checkpoint store — redrawing latent movement from its committed CSV"
  run "$HERE/plot_latent_movement.py" --from-csv \
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

# ---- 7b. what the backbone seed pins, on the cells trained twice ----------
# The B5 retrain at a fixed seed is only a machine test if the seed really
# does pin the data order. The mixer's per-window count says whether it does.
mapfile -t retrains < <(python3 "$HERE/find_artefacts.py" --what retrainlogs \
                          --results "$RES")
if [ "${#retrains[@]}" -gt 0 ]; then
  run "$HERE/early_loss.py" "${retrains[@]}" --steps 4 \
      --out "$RES/early_loss.csv"
else
  say "no cell trained twice — no early-loss table"
fi

# ---- 8. the two lead figures ----------------------------------------------
# Figure 1: every cell's best score against the frontier the project held
# before this study. Figure 2: the card's `ladder.png`, GM-Relative MASE
# against backbone train step, all 14 cells, both heads. Both draw the same
# grey rule, from `published.best_published()`.
run "$HERE/plot_frontier.py" --results "$RES" --out "$PLOTS/frontier.png"
run "$HERE/plot_ladder.py" --results "$RES" --out "$PLOTS/ladder.png"
# Annex: the five retrained arms on the trajectories their parents published.
run "$HERE/plot_k0_overlay.py" --results "$RES" --out "$PLOTS/k0_overlay.png"

# ---- 8b. the stop contrast: what the second 100,000 steps buys -------------
# `stop_bootstrap.sh` pairs each cell's bb200k eval against its OWN bb100k
# eval — same head recipe, same 97 configs — under the same dataset-cluster
# bootstrap section 2 uses. The two figures the report embeds for this
# contrast draw off it, so both are rebuilt here rather than only on the
# round's publish tick. They need no checkpoint, so the rebuild redraws
# every figure the report embeds.
bash "$HERE/stop_bootstrap.sh" "$RES/stop_bootstrap.csv" 2>&1 | sed 's/^/  /'
run "$HERE/plot_stop_delta.py" --results "$RES" --out "$PLOTS/stop_delta.png"
run "$HERE/r2_plot_ladder.py" --results "$RES" --out "$PLOTS/stop_ladder.png"

# ---- 9. the tables, and the screen figure the intervals feed ---------------
run "$HERE/published_bootstrap.py" --results "$RES"
# After published_bootstrap.py, because the screen draws its intervals.
# The card's `schedule_vs_fixed` equivalent: k = 3 minus k = 0, per cell, at
# every stop, on both heads.
run "$HERE/plot_k3_minus_k0.py" --results "$RES" \
    --out "$PLOTS/k3_minus_k0.png"
run "$HERE/tables.py" --results "$RES" --out "$RES/scores.md" \
    --inject "$DST/rollout_depth.md"

say "-> $PLOTS"
ls -1 "$PLOTS" 2>/dev/null | sed 's/^/  /'
