#!/bin/bash
# #373 — build this study's launchers from the three published ones.
#
# The 14 cells at k = 3 run the same command lines that trained their k = 0
# baselines, plus one flag. This script copies each parent launcher and
# applies exactly four edits. Run it, then `diff` a copy against its parent:
# the diff IS the study's deviation from the baseline protocol.
#
# Why copy instead of edit in place. Both parent launchers ASSIGN
# EXTRA_ARGS inside their per-cell `case` block and never read the
# environment, so `export EXTRA_ARGS=...` is silently overwritten and the
# flag never reaches the trainer — 14 runs labelled k = 3 that train at
# k = 0. Editing the published launchers is worse: they trained the k = 0
# baselines this study compares against, and two shape tests pin them.
#
# The four edits, per launcher:
#
#   1. K="${K:-3}" beside SEED.
#   2. --train-rollout-depth "$K" in the SHARED flag block, on the line
#      before "${EXTRA_ARGS[@]}". The shared block runs for every cell, so
#      no per-cell case arm changes and no cell's EXTRA_ARGS carries it.
#   3. OUT -> this study's directory, so results and logs never land in a
#      parent experiment's tree.
#   4. Run name gains a `_cf373k<K>` suffix, so no k = 3 checkpoint can
#      overwrite a published k = 0 one, and a k = 0 rerun here (the
#      baseline validity gate) is distinct from both.
#
# Usage: bash make_launchers.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
STUDY='$WT/reports/2026-08-08_rollout_depth'

die(){ echo "ABORT: $*" >&2; exit 2; }

# Assert a file holds a pattern exactly N times, so a parent launcher that
# moved under us fails here rather than producing a launcher that runs at
# k = 0.
want(){ # <file> <count> <fixed string>
  local f="$1" n="$2" pat="$3"
  # -e, because every pattern here can begin with a dash.
  local got; got=$(grep -cF -e "$pat" "$f" || true)
  [ "$got" = "$n" ] || die "$(basename "$f"): want $n match(es) of '$pat', got $got"
}

patch_launcher(){ # <src> <dst> <out-old>
  local src="$1" dst="$2" out_old="$3"
  [ -f "$src" ] || die "no parent launcher at $src"
  cp "$src" "$dst"

  want "$dst" 1 'SEED=20260520'
  want "$dst" 1 '"${EXTRA_ARGS[@]}" \'
  want "$dst" 1 "OUT=\"$out_old\""

  # 1. K beside SEED.
  sed -i 's|^SEED=20260520$|SEED=20260520\n# #373: rollout depth. Every cell of this study takes it from the SHARED\n# flag block below, never from EXTRA_ARGS.\nK="${K:-3}"|' "$dst"

  # 2. the flag, in the shared block.
  sed -i 's|^\(\s*\)"\${EXTRA_ARGS\[@\]}" \\$|\1--train-rollout-depth "$K" \\\n\1"${EXTRA_ARGS[@]}" \\|' "$dst"

  # 3. this study's output tree.
  sed -i "s|^OUT=\"$out_old\"$|OUT=\"$STUDY\"|" "$dst"

  # 4. --log-every becomes an env override, default unchanged at 200.
  #    The per-cell flag check reads the FIRST CSV row of a 1-step run: at
  #    step 1 a k = 0 and a k = 3 run of the same cell start from the same
  #    weights and see the same batch, so `loss_tau_ref` (pinned to depth 0)
  #    must match and `loss` must not. At the default 200 the weights have
  #    already diverged and neither column proves anything.
  want "$dst" 1 '--log-every 200'
  # Both parents carry it mid-line, after --run-name. `--log-attn-amplitude
  # -every 200` is a different flag and is left alone.
  sed -i 's|--log-every 200|--log-every "$LOG_EVERY"|' "$dst"
  sed -i 's|^K="\${K:-3}"$|K="${K:-3}"\nLOG_EVERY="${LOG_EVERY:-200}"|' "$dst"

  want "$dst" 1 '--train-rollout-depth "$K" \'
  want "$dst" 1 'K="${K:-3}"'
  want "$dst" 1 'LOG_EVERY="${LOG_EVERY:-200}"'
  want "$dst" 1 '--log-every "$LOG_EVERY"'
  want "$dst" 0 '--log-every 200'
  want "$dst" 1 "OUT=\"$STUDY\""
  chmod +x "$dst"
  echo "built $(basename "$dst") from $src"
}

# ---- 1. group A, the four _sched cells -------------------------------------
LEG_SRC="$ROOT/experiments/2026-08-04_ema_sched_ladder/scripts"
patch_launcher "$LEG_SRC/run_leg.sh" "$HERE/run_leg_k.sh" \
  '$WT/experiments/2026-08-04_ema_sched_ladder'
# run_leg.sh names its runs cf393_<cell>. Edit 4.
sed -i 's|^NAME="cf393_\${CELL}"$|NAME="cf393_${CELL}_cf373k${K}"|' "$HERE/run_leg_k.sh"
want "$HERE/run_leg_k.sh" 1 'NAME="cf393_${CELL}_cf373k${K}"'
# It sources two siblings, and resolves them from its own dirname.
cp "$LEG_SRC/leg_paths.sh" "$LEG_SRC/gpu_gate.sh" "$HERE/"
# Its durable root default is #393's. Give this study its own.
sed -i 's|^RUNS_DEFAULT=/home/jupyter/checkpoints_backup/cf-393$|RUNS_DEFAULT=/home/jupyter/checkpoints_backup/cf-373|' "$HERE/leg_paths.sh"
want "$HERE/leg_paths.sh" 1 'RUNS_DEFAULT=/home/jupyter/checkpoints_backup/cf-373'

# ---- 2. group B, the six fix09 cells on #379's launcher ---------------------
patch_launcher "$ROOT/experiments/2026-07-21_split_pred_rep_small/scripts/run_arm.sh" \
  "$HERE/run_arm_k.sh" '$WT/experiments/2026-07-21_split_pred_rep_small'

# ---- 3. group B, the four fix09 cells on #390's launcher --------------------
patch_launcher "$ROOT/experiments/2026-08-01_lalign_teacher/scripts/run_arm.sh" \
  "$HERE/run_arm_lalign_k.sh" '$WT/experiments/2026-08-01_lalign_teacher'

# Edit 4 for the two run_arm.sh copies. Both assign NAME per case arm, so
# the suffix goes on once, after the case block closes and before the
# trainer call — beside the K they already carry.
#
# Edit 5, same two files: checkpoints move OUT of the checkout. Both parents
# write them to `$OUT/runs`, which under edit 3 would be a git worktree —
# `git worktree remove --force` deletes every untracked file in one
# (CLAUDE.md checkpoint safety rule 4, an 80 MB backbone, Apr 2026). The
# durable root is the one run_leg_k.sh's leg_paths.sh already refuses to
# leave. RES stays under the study directory: logs are small and committed.
for f in "$HERE/run_arm_k.sh" "$HERE/run_arm_lalign_k.sh"; do
  want "$f" 1 'RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"'
  sed -i 's|^K="\${K:-3}"$|K="${K:-3}"\n# Artefacts of this study never share a name with a published k = 0 one.\nNAME="${NAME}_cf373k${K}"\n# Checkpoints live on the durable root, never inside the checkout.\nRUNS="${CF373_RUNS:-/home/jupyter/checkpoints_backup/cf-373}/$NAME"\nmkdir -p "$RUNS"|' "$f"
  sed -i 's|^RUNS="\$OUT/runs"; RES="\$OUT/results"; mkdir -p "\$RUNS" "\$RES"$|RES="$OUT/results"; mkdir -p "$RES"|' "$f"
  want "$f" 1 'NAME="${NAME}_cf373k${K}"'
  want "$f" 1 'RUNS="${CF373_RUNS:-/home/jupyter/checkpoints_backup/cf-373}/$NAME"'
  want "$f" 0 'RUNS="$OUT/runs"'
done

echo "OK — 3 launchers + 2 helpers under $HERE"

# ---- 4. the eval half, copied verbatim ---------------------------------------
# eval_local.sh takes every path as an argument, so it needs no edit: it is
# the 97-config GIFT-Eval under the official B4 strategy, sharded over
# elisa's cores, with the same seasonal-naive denominator every stop of the
# parent reports used. Re-deriving GM-Relative MASE here would be a second
# implementation of the study's one metric. eval_slot.sh caps how many run
# at once; shard_configs.py is its cost-balanced 97-way split.
cp -f "$LEG_SRC/eval_local.sh" "$LEG_SRC/eval_slot.sh" \
      "$LEG_SRC/shard_configs.py" "$HERE/"
echo "staged eval_local.sh, eval_slot.sh, shard_configs.py"

# shard_configs.py splits the 97 configs by MEASURED cost (they span 0.4 s to
# 1537 s), and it reads that table from this study's own results/. Without
# it every eval dies at shard 0 before the first config.
mkdir -p "$HERE/../results"
cp -f "$LEG_SRC/../results/config_costs.csv" "$HERE/../results/"
echo "staged config_costs.csv"
