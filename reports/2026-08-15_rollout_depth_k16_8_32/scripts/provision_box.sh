#!/bin/bash
# #401 — get ONE vast.ai box with TWO GPUs, or give up cleanly.
#
# The provisioner is #373's `provision_box.sh`, unchanged. It already handles
# the two routine vast.ai failures — an offer that churns away between the
# search and the create, and an instance that exists and BILLS while its sshd
# is not up yet — and it destroys only instances it created itself. A second
# copy of that logic here would be a second thing to keep correct.
#
# This wrapper supplies the search filters this card asks for, through
# #373's `VAST_SEARCH_ARGS`:
#
#   --num-gpus 2          one machine, two cards, one arm on each. Two
#                         1-GPU machines would cost two bootstraps, two sync
#                         loops and two failure domains for the same work.
#   --gpu-model RTX_4090,RTX_5090
#                         the two classes the card names. The step times the
#                         run plan uses were measured on a 4090.
#   --min-reliability 0.99
#                         host reliability above 99%. A 30-hour k = 32 leg
#                         is the longest single run of this study.
#   (no --prosumer)       datacenter hosts only. `vastrun-search` lists
#                         datacenter machines unless --prosumer is passed.
#
# The CPU filter is #373's and it is not cosmetic. This backbone is
# d_model = 64 at batch 64, so the kernels are small and the step rate
# follows how fast one CPU core launches them. #373 measured 5.6-6.7 steps/s
# on a Ryzen 7 7800X3D, 3.3 on an EPYC 7B13 and 1.1 on an EPYC 7452, same
# card, same arm. `vastrun-search` has no CPU filter, so #373 filters its
# `--hardware` columns by regular expression.
#
# Here that filter is a PREFERENCE, in two passes. The 2-GPU pool at this
# reliability is small — 14 offers on 2026-08-17, of which 2 printed — and a
# hard CPU gate over a pool that size returns nothing and the study waits for
# a machine instead of training. So: pass 1 asks for the fast parts, pass 2
# drops the CPU filter and says so. A slower CPU costs step time. No box
# costs the whole card.
#
# vastrun-kit only, never raw `vastai`. The vast.ai account is SHARED with
# other agent sessions: this script destroys an instance only when its own
# `vastrun-provision` created it, and only by the id that call printed.
#
# Prints `<instance id> <ssh host> <ssh port>` on success.
#
# Usage:  bash scripts/provision_box.sh [label] [max attempts]
#         MAX_BID=1.20 bash scripts/provision_box.sh cf401-box-a
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

LABEL="${1:-cf401-mean-box}"
TRIES="${2:-8}"

PROVISION="$CF401_PARENT/scripts/provision_box.sh"
[ -f "$PROVISION" ] || {
  echo "ABORT: no provisioner at $PROVISION" >&2; exit 2; }

# Two cards, on demand, for about 30 hours. The card's budget is about $80,
# so the ceiling is set well under that and the caller can raise it.
MAX_BID="${MAX_BID:-1.20}"
GPU_MODELS="${GPU_MODELS:-RTX_4090,RTX_5090}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.99}"
NUM_GPUS="${NUM_GPUS:-2}"

SEARCH_ARGS="${VAST_SEARCH_ARGS:---num-gpus $NUM_GPUS --gpu-model $GPU_MODELS --min-reliability $MIN_RELIABILITY --max-bid $MAX_BID}"
# The fast parts, over the `--hardware` columns: Ryzen and Threadripper
# desktop cores, Milan and Genoa EPYC (7x3, 7V13, 7B13, 9xx4), Xeon Gold.
# Pass 2 drops this.
CPU_RE="${VAST_CPU_RE:-Ryzen|Threadripper|Xeon.*Gold|EPYC 9|EPYC 7[VB]|EPYC 7[0-9][0-9]3}"
# Attempts for each pass. Pass 1 keeps a few, so a preferred box is taken
# when the listing holds one, and pass 2 starts soon after when it does not.
TRIES_PREF="${TRIES_PREF:-3}"

if [ -n "${CF401_DRY_RUN:-}" ]; then
  echo "provision label=$LABEL tries=$TRIES"
  echo "  provisioner=$PROVISION"
  echo "  VAST_SEARCH_ARGS=$SEARCH_ARGS"
  echo "  pass 1 VAST_CPU_RE=$CPU_RE ($TRIES_PREF attempt(s))"
  echo "  pass 2 VAST_CPU_RE=.  (any CPU, $TRIES attempt(s))"
  echo "  (datacenter hosts only: prosumer machines are not asked for)"
  exit 0
fi

echo "[#401] pass 1: $SEARCH_ARGS  (cpu ~ /$CPU_RE/)" >&2
VAST_SEARCH_ARGS="$SEARCH_ARGS" VAST_CPU_RE="$CPU_RE" \
  VAST_SEARCH_LIMIT="${VAST_SEARCH_LIMIT:-40}" \
  bash "$PROVISION" "$LABEL" "$TRIES_PREF" && exit 0

echo "[#401] pass 1 found no fast-CPU offer. Pass 2 takes ANY CPU on the" >&2
echo "       same cards. Expect a lower step rate; measure it with" >&2
echo "       scripts/smoke_depth.sh before the 200k legs." >&2
VAST_SEARCH_ARGS="$SEARCH_ARGS" VAST_CPU_RE="." \
  VAST_SEARCH_LIMIT="${VAST_SEARCH_LIMIT:-40}" \
  bash "$PROVISION" "$LABEL" "$TRIES"
