#!/bin/bash
# The committed numbers do not move: old pick == new resolver, everywhere.
#
# `eval_arm.sh` and `pipeline.sh` used to select a backbone with
#
#     ls -t "$RUNS/${NAME}"_${K}k.pth "$RUNS/${NAME}"_r*_${K}k.pth | head -1
#
# and now call `scripts/resolve_eval_checkpoint.sh`. The report's cells were
# measured under the old line, so replacing it is only safe if the two agree
# on every (run name, step) pair this experiment actually has on disk — and if
# the resolver aborts on none of them.
#
# This enumerates those pairs from `runs/` and checks both properties. Usage:
#
#   bash experiments/2026-08-01_lalign_teacher/scripts/verify_ckpt_resolution_parity.sh \
#     [runs-dir] > experiments/2026-08-01_lalign_teacher/results/ckpt_resolution_parity.txt
#
# Exits 0 only when every pair agrees and none aborts.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RUNS="${1:-$ROOT/experiments/2026-08-01_lalign_teacher/runs}"
RESOLVER="$ROOT/scripts/resolve_eval_checkpoint.sh"

[ -d "$RUNS" ]      || { echo "ABORT: no runs dir at $RUNS" >&2; exit 2; }
[ -f "$RESOLVER" ]  || { echo "ABORT: no resolver at $RESOLVER" >&2; exit 2; }

# Every (base run name, step) pair with a backbone snapshot on disk. A
# snapshot is `<name>[_r<N>]_<K>k.pth`; the `_optimizer.pth` sidecar and the
# `_EMERGENCY_<step>.pth` NaN dump are not backbones the eval selects.
pairs(){
  local f base
  for f in "$RUNS"/*k.pth; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    case "$base" in *_optimizer.pth|*_EMERGENCY_*) continue ;; esac
    # strip `_<K>k.pth`, then a trailing `_r<N>` if the file is a resume
    printf '%s %s\n' \
      "$(echo "$base" | sed -E 's/(_r[0-9]+)?_[0-9]+k\.pth$//')" \
      "$(echo "$base" | sed -E 's/^.*_([0-9]+)k\.pth$/\1/')"
  done | sort -u
}

n=0; agree=0; differ=0; aborted=0
while read -r name k; do
  [ -n "$name" ] || continue
  n=$((n + 1))
  old=$(ls -t "$RUNS/${name}"_${k}k.pth "$RUNS/${name}"_r*_${k}k.pth 2>/dev/null | head -1)
  new=$(bash "$RESOLVER" "$RUNS" "$name" "$k" 2>/dev/null); rc=$?
  if [ "$rc" -ne 0 ]; then
    aborted=$((aborted + 1))
    echo "ABORT rc=$rc  $name @ ${k}k  (old picked: ${old:-<none>})"
  elif [ "$old" = "$new" ]; then
    agree=$((agree + 1))
  else
    differ=$((differ + 1))
    echo "DIFFER        $name @ ${k}k"
    echo "  old: $old"
    echo "  new: $new"
  fi
done < <(pairs)

echo
echo "runs dir : $RUNS"
echo "pairs    : $n"
echo "agree    : $agree"
echo "differ   : $differ"
echo "aborted  : $aborted"
if [ "$n" -gt 0 ] && [ "$differ" -eq 0 ] && [ "$aborted" -eq 0 ]; then
  echo "VERDICT  : PARITY — the resolver returns the file the old line returned, on every pair."
  exit 0
fi
echo "VERDICT  : NOT PARITY"
exit 1
