#!/bin/bash
# #373 round 3 — the success tail.
#
# Two loops already cover the failure side: q_super.sh restarts a dead
# dispatcher, q_guard.sh stops everything at the credit floor and posts the
# blocking comment. Nothing covered the side where the round simply FINISHES.
# Two things have to happen there and neither can wait for a session:
#
#   1. The meter. Backbones and heads run on the rented box; the 97-config
#      GIFT-Eval runs on elisa's cores. So the box is dead weight from the
#      moment the last head lands, about 1.5 h before the last eval, and at
#      $0.81/h that is real money against a $15 credit.
#
#   2. The round's own comment on PR #400.
#
# The destroy is gated, not timed. `verify_box_work` reads every job the
# queue calls done and checks the artefact it produced is on THIS disk, by
# name and by size: a backbone and its optimizer sidecar over 4 MB, a head
# final over 300 KB. One miss and the box lives and the script says so.
# That is the standing rule — never destroy an instance whose work is not
# already off it.
#
# Usage: BOX_ID=<id> bash q_finish.sh [poll seconds]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
Q="$HERE/q_queue.tsv"
STATE="$RES/queue"
BOX_ID="${BOX_ID:?BOX_ID}"
BOX_LABEL="${BOX_LABEL:-cf373-dual}"
VDIR="${VDIR:-/home/jupyter/wt-cf-373-run2}"
PR="${PR:-400}"
K="${K:-3}"
HEAD_SEED="${HEAD_SEED:-20260722}"
R3="${CF373_R3:-/home/jupyter/cf373_r3/sync}"
POLL="${1:-180}"
# The sync loop ticks every 900 s. One full tick has to pass after the last
# head writes before the box may go.
SYNC_GRACE="${SYNC_GRACE:-1000}"
BB_MIN="${BB_MIN:-4000000}"
HD_MIN="${HD_MIN:-300000}"
export PATH="$HOME/.local/bin:$PATH"

# shellcheck source=/dev/null
source "$HERE/cell_paths.sh"

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [finish] $*" | tee -a "$RES/q_finish.log"; }

ids(){ awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$Q"; }
row(){ awk -F'\t' -v i="$1" '$1==i {print; exit}' "$Q"; }
st(){ cat "$STATE/$1.state" 2>/dev/null || echo queued; }
counts(){ local d=0 r=0 q=0 f=0 s
          for id in $(ids); do s="$(st "$id")"
            case "$s" in done) d=$((d+1));; running) r=$((r+1));;
                         failed) f=$((f+1));; *) q=$((q+1));; esac; done
          echo "done=$d running=$r queued=$q failed=$f"; }
# Open jobs whose id matches an ERE. `open` = neither done nor failed.
open_matching(){ local s n=0
  for id in $(ids); do
    [[ "$id" =~ $1 ]] || continue
    s="$(st "$id")"
    case "$s" in done|failed) ;; *) n=$((n+1));; esac
  done; echo "$n"; }

big_enough(){ [ -f "$1" ] && [ "$(stat -c%s "$1" 2>/dev/null || echo 0)" -ge "$2" ]; }

head_final(){ # <cell> <stop> <enc>  -> the path the head writes, on this disk
  printf '%s/eval/%s_k%s_bb%dk_%s/qhead_%s_k%s_bb%dk_%s_s%s_final.pth\n' \
    "$R3" "$1" "$K" "$(( $2 / 1000 ))" "$3" \
    "$1" "$K" "$(( $2 / 1000 ))" "$3" "$HEAD_SEED"
}

# Every artefact the box produced, on this disk, by name and by size.
verify_box_work(){
  local bad=0 id type cell stop enc f opt
  for id in $(ids); do
    [ "$(st "$id")" = done ] || continue
    IFS=$'\t' read -r _ type cell stop enc _ <<<"$(row "$id")"
    case "$type" in
      bb)
        f="$(CF373_ROOT="$R3" cf373_bb_ckpt "$cell" "$K" "$stop")"
        if [ -z "$f" ] || ! big_enough "$f" "$BB_MIN"; then
          log "UNVERIFIED $id: no backbone >= $BB_MIN B under $R3"; bad=1; continue
        fi
        opt="${f%.pth}_optimizer.pth"
        if big_enough "$opt" "$BB_MIN"; then
          log "verified $id -> $(basename "$f") + optimizer"
        else
          log "UNVERIFIED $id: optimizer missing or small ($opt)"; bad=1
        fi
        ;;
      head)
        f="$(head_final "$cell" "$stop" "$enc")"
        if big_enough "$f" "$HD_MIN"; then
          log "verified $id -> $(basename "$f")"
        else
          log "UNVERIFIED $id: head final missing or small ($f)"; bad=1
        fi
        ;;
    esac
  done
  return "$bad"
}

# The gate is the one thing here that must not be wrong, so it is runnable on
# its own: VERIFY_ONLY=1 prints a line per done job and exits on the verdict.
if [ "${VERIFY_ONLY:-0}" = 1 ]; then
  verify_box_work && { log "VERIFY ok — $(counts)"; exit 0; }
  log "VERIFY failed"; exit 1
fi

log "start. box $BOX_ID ($BOX_LABEL), PR #$PR, poll ${POLL}s. $(counts)"

# The A1/B3 reproduction trains four heads on box card 1. They are not queue
# jobs, so `open_matching` cannot see them, and a queue that goes terminal
# early — one failed backbone is enough — would take the box out from under a
# repro head still training. The box may not go until all four finals are on
# this disk, or until the deadline says they never will be.
REPRO_TAGS="A1rep_k3_bb40k_student B3rep_k3_bb40k_student
            A1rep_k3_bb100k_student B3rep_k3_bb100k_student"
REPRO_DEADLINE="${REPRO_DEADLINE:-21600}"
repro_open(){ local n=0 t
  for t in $REPRO_TAGS; do
    big_enough "$RES/eval/$t/qhead_${t}_s${HEAD_SEED}_final.pth" "$HD_MIN" \
      || n=$((n+1))
  done; echo "$n"; }

# ---------------------------------------------------------------- the meter
waited=0
while [ "$(open_matching '^(bb|hd)_')" -gt 0 ] || [ "$(repro_open)" -gt 0 ]; do
  if [ "$(open_matching '^(bb|hd)_')" -eq 0 ] && [ "$waited" -ge "$REPRO_DEADLINE" ]; then
    log "the queue is terminal and $(repro_open) repro head(s) never landed after ${waited}s — going on"
    break
  fi
  sleep "$POLL"; waited=$(( waited + POLL ))
done
log "every backbone and head terminal, repro heads open: $(repro_open). $(counts)"

if [ -f "$RES/BLOCKED_BUDGET" ]; then
  log "the guard already stopped the box; nothing to tear down"
else
  log "waiting ${SYNC_GRACE}s for one full sync tick before the destroy"
  sleep "$SYNC_GRACE"
  (cd "$VDIR" && timeout 120 vastrun-status 2>/dev/null) > "$RES/box_final_status.txt"
  if verify_box_work; then
    log "every backbone and head verified on this disk by name and size"
    if (cd "$VDIR" && timeout 300 vastrun-destroy "$BOX_ID" "$BOX_LABEL" --force) \
         >>"$RES/q_finish.log" 2>&1; then
      log "box $BOX_ID destroyed; the meter stops"
    else
      log "DESTROY FAILED for $BOX_ID — it is still running, stop it by hand"
    fi
  else
    log "BOX KEPT: work is not verified on this disk. Nothing was destroyed."
    printf 'q_finish refused to destroy box %s at %s: an artefact a done job\nproduced is not under %s. See results/q_finish.log.\n' \
      "$BOX_ID" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$R3" > "$RES/FINISH_BLOCKED"
  fi
fi

# ---------------------------------------------------------------- the round
while [ "$(open_matching '.')" -gt 0 ]; do sleep "$POLL"; done
log "every job terminal. $(counts)"

# One last publish pass. r3_publish.sh collects, rebuilds the tables and the
# ladder, commits and pushes, then stands down on a drained queue — so this
# call returns rather than looping.
if pgrep -f 'bash scripts/r3_publish.sh' >/dev/null; then
  log "the 20-min publisher is alive; giving it one tick to take the last score"
  sleep 1260
else
  log "running one final publish pass"
  (cd "$STUDY" && CF373_R3="$R3" timeout 1800 bash scripts/r3_publish.sh 30) \
    >>"$RES/q_finish.log" 2>&1
fi

credit="$(cd "$VDIR" && timeout 120 vastrun-balance 2>/dev/null \
          | awk '/Credit/{print $2}')"
nfail="$(for id in $(ids); do st "$id"; done | grep -c failed)"

{
  printf '«Agent ExperimentRunner claude-opus-5 writing»\n\n'
  printf '## Round 3 complete — every deliverable is scored\n\n'
  printf '**Experiment directory:** `reports/2026-08-08_rollout_depth/`\n'
  printf '(results `results/`, plots `plots/`, scripts `scripts/`, run tree `/home/jupyter/cf373_r3`)\n\n'
  printf '### Coverage, 14 cells x 3 stops x 2 heads\n\n'
  cat "$RES/coverage.md" 2>/dev/null || printf '(no coverage table on disk)\n'
  printf '\n### Runs completed\n\n```\nqueue     %s   %s\n' "$(ids | wc -l | tr -d ' ') jobs" "$(counts)"
  printf 'backbone  9 legs: B8 to 100k, eight cells 100k -> 200k\n'
  printf 'heads     30,000 steps, seed %s, --grad-clip 1.0\n' "$HEAD_SEED"
  printf 'evals     97 GIFT-Eval configs, strategy B4, horizon 16\n```\n\n'
  printf '### Spend\n\nCredit **%s** at %s. Box %s is stopped: the meter ran only while a\n' \
    "${credit:-?}" "$(date -u '+%Y-%m-%dT%H:%MZ')" "$BOX_ID"
  printf 'backbone or a head held a card. Every eval ran on elisa cores and cost nothing.\n\n'
  [ "$nfail" -gt 0 ] && printf '**%s job(s) failed — see `results/q_run.log`.**\n\n' "$nfail"
  printf 'Posted by `scripts/q_finish.sh`, which the queue outlives its sessions.\n'
} > "$RES/final_comment.md"

if (cd "$VDIR" && timeout 180 gh pr comment "$PR" --body-file "$RES/final_comment.md") \
     >>"$RES/q_finish.log" 2>&1; then
  log "completion comment posted on PR #$PR"
else
  log "could not post on PR #$PR — $RES/final_comment.md holds the text"
fi
log "done"
