#!/bin/bash
# #393 — where a leg's artefacts live, how the next leg finds them, and how
# the stop's one number is read back.
#
# Sourced by run_leg.sh and eval_stop.sh. Four jobs.
#
# 1. Resolve the DURABLE ROOT. Backbone checkpoints, quantile heads and
#    GIFT-Eval outputs each cost GPU hours. `git worktree remove --force`
#    deletes every untracked file under the checkout (CLAUDE.md checkpoint
#    safety rule 4, Apr-2026 incident, an 80 MB backbone), and /tmp does
#    not survive a reboot. Both are refused, for every artefact class.
#
# 2. Give each leg its OWN save dir, `<cell>/leg_<target>k`. This is not
#    tidiness. train.py branches `--run-name` to `<name>_r2` when the save
#    dir it is handed already holds `<name>_*.pth` (safe_run_name), so a
#    second leg writing into the first leg's dir produces
#    `<name>_r2_100k.pth` — not where the ladder or the eval look for the
#    stop's checkpoint, and the leg fails after burning its GPU time. A
#    fresh dir per leg is also CLAUDE.md checkpoint safety rule 2: never
#    reuse a save path when resuming.
#
# 3. Resolve checkpoints BY STEP, never by mtime. Splitting a cell across
#    machines with --max-stop means copying a checkpoint set, and a copy
#    stamps every file with a fresh mtime in copy order, so `ls -t` returns
#    whichever was copied last. The `_<N>k` field in the filename is the
#    only one that means what it says. The glob tolerates the `_rN` infix
#    so a re-fired leg's checkpoints are still found.
#
# 4. Read the stop's score out of the eval's summary, pinned to one file
#    and one metric — see score_from_summary.

RUNS_DEFAULT=/home/jupyter/checkpoints_backup/cf-373

runs_root(){
  local root="${RUNS:-$RUNS_DEFAULT}"
  case "$root" in
    /tmp|/tmp/*|"${WT:-/nonexistent}"|"${WT:-/nonexistent}"/*)
      echo "ABORT: RUNS=$root is under /tmp or inside the checkout." >&2
      echo "  Point it at a durable path, e.g. $RUNS_DEFAULT." >&2
      return 2 ;;
  esac
  printf '%s\n' "$root"
}

# The save dir for the leg that targets <total steps>.
leg_dir(){  # <cell runs dir> <target steps>
  printf '%s/leg_%dk\n' "$1" "$(( $2 / 1000 ))"
}

# The checkpoint a stop produced, or nothing.
ckpt_at_step(){  # <cell runs dir> <run name> <step in thousands>
  ls "$1"/leg_"$3"k/"$2"*_"$3"k.pth 2>/dev/null | head -1
}

# The cell's furthest checkpoint across all its legs, or nothing.
newest_ckpt(){  # <cell runs dir> <run name>
  ls "$1"/leg_*/"$2"*_[0-9]*k.pth 2>/dev/null \
    | sed -E 's|.*_([0-9]+)k\.pth$|\1 &|' \
    | sort -k1,1n | tail -1 | cut -d' ' -f2-
}

# EVERY step checkpoint under a cell's runs dir, whatever run name it
# carries. `newest_ckpt` above is pinned to ONE run name, so a checkpoint
# written under any other name reads there as "nothing on disk" and the leg
# starts fresh at step 0. This is the second reading, and a leg refuses to
# start fresh while it returns anything.
step_ckpts(){  # <cell runs dir>
  ls "$1"/leg_*/*_[0-9]*k.pth 2>/dev/null | grep -v optimizer
}

# The step a checkpoint path carries, in thousands.
ckpt_step_k(){  # <checkpoint path>
  printf '%s\n' "$1" | sed -E 's|.*_([0-9]+)k\.pth$|\1|'
}

# The one number the ladder records, out of one eval's summary.
#
# eval_gift_eval_official.py writes exactly one summary.txt per
# --output-dir, holding exactly one `Aggregate GM-Relative MASE (N
# configs): X` line. Pinning both the file and the metric name is the
# point: `grep Aggregate` over a glob returns whichever line the glob
# happened to order first, so the day a second aggregate metric (GM-MAPE_SN,
# GM-CRPS_SN) or a per-config summary appears, the ladder would record that
# one as GM-Relative MASE and the extend rule would run on it. More than
# one match is a protocol change, not something to pick a winner from.
score_from_summary(){  # <summary.txt>
  local agg n
  [ -f "$1" ] || { echo "ABORT: no summary at $1" >&2; return 4; }
  agg=$(grep -h "Aggregate GM-Relative MASE" "$1")
  n=$(printf '%s' "$agg" | grep -c .)
  [ "$n" -eq 1 ] || {
    echo "ABORT: $1 holds $n 'Aggregate GM-Relative MASE' lines, want 1" >&2
    printf '%s\n' "$agg" >&2
    return 4; }
  # "Aggregate GM-Relative MASE (97 configs): 1.1556" -> "1.1556". A format
  # change leaves something unparseable here, which the driver turns into a
  # hard stop rather than a wrong number.
  printf '%s\n' "$agg" | sed -E 's/.*\): *//'
}
