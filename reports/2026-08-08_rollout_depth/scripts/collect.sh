#!/bin/bash
# #373 — gather the study's artefacts into the git checkout for committing.
#
# Three places hold them and none of them is the checkout:
#
#   $RUN_WT/reports/.../results   the run worktree — launcher logs, verify
#                                 CSVs, step-time tables, score files. The
#                                 launchers refuse a WT under /tmp, so the
#                                 training checkout is not the git one.
#   ~/cf373_sync/<box>/sync       what the sync loops pulled off each box —
#                                 trainer logs and losses CSVs.
#   $CF373_ROOT                   the durable root — checkpoints, heads,
#                                 GIFT-Eval outputs, and the losses CSV of
#                                 every run that trained on elisa, which has
#                                 no sync loop of its own.
#
# Checkpoints stay where they are: an 80 MB backbone does not belong in git.
# Everything else that is small and is evidence comes across.
#
# Usage: bash collect.sh [git checkout root]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_ROOT="${1:-/tmp/contrastive-forecasting-373}"
DST="$GIT_ROOT/reports/2026-08-08_rollout_depth"
RUN_WT="${RUN_WT:-/home/jupyter/wt-cf-373-train}"
SYNC_BASE="${CF373_SYNC_BASE:-$HOME/cf373_sync}"
CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"
mkdir -p "$DST/results" "$DST/plots"

say(){ echo "[collect] $*"; }
die(){ echo "[collect] STOP: $*" >&2; exit 1; }

# 1. The run worktree's results.
#
# Only the RUNS' OWN OUTPUT comes across. Everything the rebuild generates is
# excluded, because the run worktree carries a stale fork of each one and
# rsync happily copies the older file over the newer. Aug 10 2026: 82 lines
# of `execution_log.md` lost that way. `execution_log.md` was then excluded
# by name, which left the other eight to be reverted the same way.
#
# The list is here in full, and a guard below re-derives it from
# `make_report_assets.sh` and refuses to run if the two disagree. A rebuild
# output that nobody adds here would otherwise be silently reverted, and
# nothing about the reverted file would say so.
GENERATED=(
  bootstrap.csv        # 2. paired dataset-cluster bootstrap
  depth0_gap.csv       # 4. the depth-0 diagnostic, as a number
  early_loss.csv       # 7b. what the backbone seed pins
  head_gap.tsv         # 3. every student/teacher gap, largest first
  latent_movement.csv  # 6. movement between the 20k and 40k checkpoints
  rollout_fidelity.csv # 5. cos against the true h, per depth
  run_provenance.csv   # 7. who else held the card
  scores.md            # 9. every table the report carries
  splits.csv           # 1. horizon term and domain, per finished eval
  steptime_solo.csv    # 7. median fwd+bwd where the run was alone
  execution_log.md     # written by hand in the checkout; no run writes it
)

# The guard. `make_report_assets.sh` names each of its outputs as a literal
# `$RES/<name>`, so the set is greppable. Read-only references resolve to the
# same names, and a new one is a mismatch a person should look at rather than
# something to guess about.
derived="$(grep -oE '\$RES/[A-Za-z0-9_]+\.(csv|md|tsv)' "$HERE/make_report_assets.sh" \
           | sed 's|\$RES/||' | sort -u)"
[ -n "$derived" ] || die "the rebuild script named no \$RES/<file>.csv|md|tsv. \
the grep in this guard no longer matches it. Fix the guard, do not delete it."
listed="$(printf '%s\n' "${GENERATED[@]}" | grep -v '^execution_log.md$' | sort -u)"
if [ "$derived" != "$listed" ]; then
  die "GENERATED is out of date with make_report_assets.sh:
$(diff <(echo "$listed") <(echo "$derived") | sed 's/^/    /')
  (< only in GENERATED, > only in the rebuild script)"
fi

if [ -d "$RUN_WT/reports/2026-08-08_rollout_depth/results" ]; then
  ex=(--exclude='*.pth')
  for g in "${GENERATED[@]}"; do ex+=(--exclude="$g"); done
  rsync -a "${ex[@]}" \
    "$RUN_WT/reports/2026-08-08_rollout_depth/results/" "$DST/results/"
  say "run worktree results (${#GENERATED[@]} generated file(s) held back)"
fi

# 2. Per-box: the trainer log and the losses CSV of every run that box
#    carried. Named by box so two boxes cannot collide on `run_<name>.log`.
for d in "$SYNC_BASE"/*; do
  [ -d "$d" ] || continue
  lbl="$(basename "$d")"
  mkdir -p "$DST/sync/$lbl"
  # Logs and small CSVs go across whole. The per-step losses CSV does not:
  # it is one row per step, ~11 MB per 40k run, and a report commits a curve
  # rather than a log. scripts/downsample_curve.py is the house tool for it,
  # and it counts DISTINCT STEPS rather than step values, so the reduction
  # is the same whatever cadence the writer used.
  find "$d" \( -name '*.log' -o -name '*_latent_drift.csv' \) -type f 2>/dev/null \
  | while read -r f; do
      cp -f "$f" "$DST/sync/$lbl/$(basename "$f")"
    done
  find "$d" \( -name '*_losses.csv' -o -name '*_attn_amplitude.csv' \) -type f 2>/dev/null \
  | while read -r f; do
      out="$DST/sync/$lbl/$(basename "$f")"
      python3 "$GIT_ROOT/scripts/downsample_curve.py" "$f" "$out" \
        --stride 20 --dense-until 1000 >/dev/null 2>&1 \
        || cp -f "$f" "$out"
    done
  n=$(ls -1 "$DST/sync/$lbl" 2>/dev/null | wc -l)
  say "box $lbl: $n file(s)"
done

# 2b. Every backbone whose losses CSV no box directory carries. Each elisa
#     run is one: elisa has no sync loop, it wrote to the durable root, and
#     so five of the study's curves lived only on this machine. That left
#     BOTH SIDES of B1, the study's one sound comparison, out of git, and
#     the training-curve figures rebuilt from files a clone cannot read.
#
#     Same downsampling as the box runs above, so every curve in a figure is
#     at one resolution. `find_artefacts.py` decides what is missing by
#     looking in the committed tree first, so a run already under `sync/` is
#     not copied twice and re-running this is a no-op.
while IFS=$'\t' read -r run mach src; do
  [ -n "${run:-}" ] && [ -f "${src:-}" ] || continue
  mkdir -p "$DST/curves/$mach"
  out="$DST/curves/$mach/$(basename "$src")"
  python3 "$GIT_ROOT/scripts/downsample_curve.py" "$src" "$out" \
    --stride 20 --dense-until 1000 >/dev/null 2>&1 || cp -f "$src" "$out"
  say "curve $mach/$run"
done < <(python3 "$HERE/find_artefacts.py" --what missingcurves)

# 3. GIFT-Eval outputs: the per-config CSV and the summary, per stop. These
#    are the study's numbers; the head checkpoints beside them are not.
if [ -d "$CF373_ROOT/eval" ]; then
  for d in "$CF373_ROOT"/eval/*/; do
    [ -d "$d" ] || continue
    tag="$(basename "$d")"
    mkdir -p "$DST/results/eval/$tag"
    for f in gift/all_results.csv gift/summary.txt stop.log eval_local.log; do
      [ -f "$d/$f" ] && cp -f "$d/$f" "$DST/results/eval/$tag/$(basename "$f")"
    done
  done
  say "eval outputs: $(ls -1 "$DST/results/eval" 2>/dev/null | wc -l) stop(s)"
fi

say "-> $DST"
