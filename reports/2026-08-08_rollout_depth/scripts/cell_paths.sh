#!/bin/bash
# #373 — where one (cell, k) run's artefacts live.
#
# The 14 cells run three launchers, and the two groups lay their
# checkpoints out differently because their parents did:
#
#   group A, run_leg_k.sh      <root>/<arg>/leg_<N>k/cf393_<arg>_cf373k<K>_<N>k.pth
#   group B, run_arm*_k.sh     <root>/<name>/<name>_<N>k.pth
#
# Everything downstream — the head, the GIFT-Eval, the sync loop — needs one
# answer, so it comes from here. Sourced, never run.
#
# Nothing in this file guesses. `bb_ckpt` returns a path that exists or
# nothing, and the run name comes from the launcher's own case block, read
# out of the launcher file rather than copied into a second table that can
# drift.

CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"

cf373_row(){ # <cell id> -> "id slug launcher arg" on stdout
  local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  awk -F'\t' -v c="$1" '$1==c {print $1"\t"$2"\t"$3"\t"$4; exit}' "$here/cells.tsv"
}

# The run name the launcher would build for this cell at this k.
# Group A's is a template. Group B's lives in the launcher's case block: ask
# the launcher, do not re-derive it.
cf373_run_name(){ # <cell id> <k>
  local row launcher arg here name
  row="$(cf373_row "$1")" || return 2
  [ -n "$row" ] || { echo "no such cell: $1" >&2; return 2; }
  launcher=$(cut -f3 <<<"$row"); arg=$(cut -f4 <<<"$row")
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  case "$launcher" in
    run_leg_k.sh) printf 'cf393_%s_cf373k%s\n' "$arg" "$2" ;;
    *)
      # The NAME= line of that arm, with the study's suffix appended the
      # same way the launcher appends it.
      name="$(awk -v a="  $arg)" '
        $0 == a { grab = 1; next }
        grab && /^ *NAME=/ { sub(/^ *NAME="/, ""); sub(/"$/, ""); print; exit }
      ' "$here/$launcher")"
      [ -n "$name" ] || { echo "no NAME for arm '$arg' in $launcher" >&2; return 2; }
      printf '%s_cf373k%s\n' "$name" "$2" ;;
  esac
}

# The directory the launcher saves that run into.
cf373_runs_dir(){ # <cell id> <k> <stop steps>
  local row launcher arg name
  row="$(cf373_row "$1")" || return 2
  launcher=$(cut -f3 <<<"$row"); arg=$(cut -f4 <<<"$row")
  name="$(cf373_run_name "$1" "$2")" || return 2
  case "$launcher" in
    run_leg_k.sh) printf '%s/%s/leg_%dk\n' "$CF373_ROOT" "$arg" "$(( $3 / 1000 ))" ;;
    *)            printf '%s/%s\n' "$CF373_ROOT" "$name" ;;
  esac
}

# The checkpoint a stop produced, or nothing. The `*` tolerates train.py's
# `_rN` infix on a re-fired run.
cf373_bb_ckpt(){ # <cell id> <k> <stop steps>
  local dir name
  dir="$(cf373_runs_dir "$1" "$2" "$3")" || return 2
  name="$(cf373_run_name "$1" "$2")" || return 2
  ls "$dir/$name"*_"$(( $3 / 1000 ))"k.pth 2>/dev/null | grep -v optimizer | head -1
}

# The FURTHEST checkpoint strictly below a stop, chosen by the step in its
# name and required to carry its optimizer sidecar. An extend resumes this,
# never `stop - 100000`: B1 and B2 already hold a 140k, and resuming their
# 100k would retrain 40k steps each that are already on disk.
#
# Two things must not decide this. `ls -t` reads mtime, and a checkpoint set
# copied between machines comes back in copy order, not training order. A
# fixed arithmetic step assumes every cell stopped on the same grid, and
# these did not.
#
# Group B keeps one directory per run. Group A keeps one per leg, so the
# search walks the arm's sibling legs.
cf373_bb_below(){ # <cell id> <k> <stop steps>
  local row launcher arg name dirs d f n best="" best_n=-1
  row="$(cf373_row "$1")" || return 2
  [ -n "$row" ] || return 2
  launcher=$(cut -f3 <<<"$row"); arg=$(cut -f4 <<<"$row")
  name="$(cf373_run_name "$1" "$2")" || return 2
  case "$launcher" in
    run_leg_k.sh) dirs="$(ls -d "$CF373_ROOT/$arg"/leg_*k 2>/dev/null)" ;;
    *)            dirs="$CF373_ROOT/$name" ;;
  esac
  for d in $dirs; do
    for f in "$d/$name"*k.pth; do
      case "$f" in *optimizer*|*'*'*) continue ;; esac
      [ -f "$f" ] || continue
      [ -f "${f%.pth}_optimizer.pth" ] || continue
      n="${f%k.pth}"; n="${n##*_}"
      case "$n" in ''|*[!0-9]*) continue ;; esac
      [ "$(( n * 1000 ))" -lt "$3" ] || continue
      [ "$n" -gt "$best_n" ] && { best_n="$n"; best="$f"; }
    done
  done
  [ -n "$best" ] && printf '%s\n' "$best"
}
