#!/bin/bash
# Push the four-file resume bundle to a fresh vast.ai instance before
# relaunching a training run.
#
# Reason for existing: when the original instance dies and a fresh one
# is provisioned, the operator must push *all four artifact classes*,
# not just the model:
#   1. <run_name>.pth             — the resume target (FINAL or best_loss)
#   2. <run_name>_optimizer.pth   — companion (RNG, AdamW state, step counter)
#   3. <run_name>_losses.csv      — historical CSV (trainer appends with mode="a")
#   4. run_<arm>.log              — historical log (trainer's `tee -a` appends)
#
# Forgetting any one of these silently destroys the local pre-resume
# history once the sync_loop ticks once on the fresh remote (the new
# remote starts a short file, sync_loop pulls it, the long good local
# copy gets rotated to .prev, the next cycle overwrites .prev with
# another short pull). The May 1–2 2026 learnable-τ incident lost
# steps 0–17100 of the loss CSV that way; the May 2 #6 eval-only
# relaunch on machine7 lost ~300 per-step τ samples by forgetting the
# run.log step alone.
#
# This script bundles all four classes into one push, atomically (each
# scp lands at <remote>.tmp first, then mv on the remote rotates it
# into place — partial transfers cannot be seen by a launching
# trainer).
#
# Usage:
#   push_resume_bundle.sh [--head-only] [--dry-run] \
#       <local_sync_arm_dir> <remote_user>@<remote_host> <ssh_port> \
#       <bb_run_name> <head_run_name> [arm_label]
#
# Args:
#   <local_sync_arm_dir>   — local arm dir; must contain checkpoints/
#                           and run.log
#   <remote_user>@<remote_host> — vast.ai SSH target (e.g. root@ssh6.vast.ai)
#   <ssh_port>             — vast.ai SSH proxy port
#   <bb_run_name>          — backbone run-name (e.g.
#                           tiny_realonly_full4096_learnable_tau).
#                           Skipped under --head-only.
#   <head_run_name>        — head run-name (e.g.
#                           R1q_realonly_full4096_learnable_tau)
#   [arm_label]            — optional. Used to name the remote run log
#                           as /workspace/app/run_<arm_label>.log. If
#                           omitted, derived from bb_run_name by
#                           stripping the leading "tiny_" prefix; under
#                           --head-only, omitted defaults to the
#                           basename of <local_sync_arm_dir>.
#
# Flags:
#   --head-only — skip the backbone files. For eval-only relaunches
#                 where the backbone+head are already on the new remote
#                 from the previous training round but the run.log /
#                 head losses CSV need restoring before the next
#                 launch.
#   --dry-run   — print what would be pushed (✓/✗ lines) without
#                 invoking scp. Suitable for syntactic + path-existence
#                 testing.
#
# Exit codes:
#   0 — every expected file was pushed (or would be, under --dry-run)
#   1 — at least one expected local file was missing, OR a transfer
#       failed (a clear ✗ line is printed for each missing file before
#       exit)

set -u

###############################################################################
# Argument parsing
###############################################################################

HEAD_ONLY=0
DRY_RUN=0

while [[ ${1:-} == --* ]]; do
    case "$1" in
        --head-only) HEAD_ONLY=1 ;;
        --dry-run)   DRY_RUN=1 ;;
        --help|-h)
            sed -n '2,/^set -u/p' "$0" | sed -e '/^set -u/d' -e 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown flag: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [[ $# -lt 5 ]]; then
    echo "usage: push_resume_bundle.sh [--head-only] [--dry-run]" >&2
    echo "         <local_sync_arm_dir> <user@host> <ssh_port> <bb_run_name> <head_run_name> [arm_label]" >&2
    exit 1
fi

LOCAL_ARM_DIR="$1"
SSH_TARGET="$2"   # e.g. root@ssh6.vast.ai
SSH_PORT="$3"
BB_RUN_NAME="$4"  # ignored under --head-only
HEAD_RUN_NAME="$5"
ARM_LABEL="${6:-}"

# Normalize: strip trailing slash off LOCAL_ARM_DIR so basename works
LOCAL_ARM_DIR="${LOCAL_ARM_DIR%/}"

# Derive arm label if not given:
#  * non-head-only: strip leading "tiny_" (or "R1q_") from BB_RUN_NAME
#  * head-only:     fallback to basename of LOCAL_ARM_DIR
if [[ -z "$ARM_LABEL" ]]; then
    if [[ $HEAD_ONLY -eq 1 ]]; then
        ARM_LABEL="$(basename "$LOCAL_ARM_DIR")"
    else
        ARM_LABEL="${BB_RUN_NAME#tiny_}"
        ARM_LABEL="${ARM_LABEL#R1q_}"
    fi
fi

###############################################################################
# Local-side preflight: verify every file we plan to push exists
###############################################################################

LOCAL_CKPT="${LOCAL_ARM_DIR}/checkpoints"
LOCAL_LOG="${LOCAL_ARM_DIR}/run.log"

REMOTE_CKPT_DIR="/workspace/app/checkpoints"
REMOTE_LOG="/workspace/app/run_${ARM_LABEL}.log"

if [[ ! -d "$LOCAL_CKPT" ]]; then
    echo "ERROR: local checkpoints dir does not exist: $LOCAL_CKPT" >&2
    exit 1
fi

# A "file class" is a quadruple (label, local_path, remote_path, required).
# We collect them all first so we can print one ✗ per missing file before
# exiting, instead of bailing on the first one.

declare -a CLASS_LABELS
declare -a CLASS_LOCAL
declare -a CLASS_REMOTE
declare -a CLASS_MISSING_OK

# Pick the resume .pth: prefer <name>_FINAL.pth, fall back to <name>_best_loss.pth
pick_resume_pth() {
    local run_name="$1"
    if [[ -f "${LOCAL_CKPT}/${run_name}_FINAL.pth" ]]; then
        echo "${LOCAL_CKPT}/${run_name}_FINAL.pth"
        return 0
    fi
    if [[ -f "${LOCAL_CKPT}/${run_name}_best_loss.pth" ]]; then
        echo "${LOCAL_CKPT}/${run_name}_best_loss.pth"
        return 0
    fi
    if [[ -f "${LOCAL_CKPT}/${run_name}_best.pth" ]]; then
        echo "${LOCAL_CKPT}/${run_name}_best.pth"
        return 0
    fi
    return 1
}

pick_optimizer_pth() {
    # Match the optimizer that pairs with the resume .pth choice.
    local run_name="$1"
    if [[ -f "${LOCAL_CKPT}/${run_name}_FINAL.pth" \
       && -f "${LOCAL_CKPT}/${run_name}_FINAL_optimizer.pth" ]]; then
        echo "${LOCAL_CKPT}/${run_name}_FINAL_optimizer.pth"
        return 0
    fi
    if [[ -f "${LOCAL_CKPT}/${run_name}_best_loss.pth" \
       && -f "${LOCAL_CKPT}/${run_name}_best_loss_optimizer.pth" ]]; then
        echo "${LOCAL_CKPT}/${run_name}_best_loss_optimizer.pth"
        return 0
    fi
    if [[ -f "${LOCAL_CKPT}/${run_name}_best.pth" \
       && -f "${LOCAL_CKPT}/${run_name}_best_optimizer.pth" ]]; then
        echo "${LOCAL_CKPT}/${run_name}_best_optimizer.pth"
        return 0
    fi
    return 1
}

# Build the file-class list.
if [[ $HEAD_ONLY -eq 0 ]]; then
    # Backbone bundle
    if BB_PTH=$(pick_resume_pth "$BB_RUN_NAME"); then
        CLASS_LABELS+=("backbone .pth")
        CLASS_LOCAL+=("$BB_PTH")
        CLASS_REMOTE+=("${REMOTE_CKPT_DIR}/$(basename "$BB_PTH")")
        CLASS_MISSING_OK+=(0)
    else
        CLASS_LABELS+=("backbone .pth")
        CLASS_LOCAL+=("${LOCAL_CKPT}/${BB_RUN_NAME}_FINAL.pth (or _best_loss.pth)")
        CLASS_REMOTE+=("(would push)")
        CLASS_MISSING_OK+=(0)
    fi

    if BB_OPT=$(pick_optimizer_pth "$BB_RUN_NAME"); then
        CLASS_LABELS+=("backbone optimizer")
        CLASS_LOCAL+=("$BB_OPT")
        CLASS_REMOTE+=("${REMOTE_CKPT_DIR}/$(basename "$BB_OPT")")
        CLASS_MISSING_OK+=(0)
    else
        CLASS_LABELS+=("backbone optimizer")
        CLASS_LOCAL+=("${LOCAL_CKPT}/${BB_RUN_NAME}_FINAL_optimizer.pth (or _best_loss_optimizer.pth)")
        CLASS_REMOTE+=("(would push)")
        CLASS_MISSING_OK+=(0)
    fi

    CLASS_LABELS+=("backbone losses CSV")
    CLASS_LOCAL+=("${LOCAL_CKPT}/${BB_RUN_NAME}_losses.csv")
    CLASS_REMOTE+=("${REMOTE_CKPT_DIR}/${BB_RUN_NAME}_losses.csv")
    CLASS_MISSING_OK+=(0)
fi

# Head bundle (always, in both modes)
if HEAD_PTH=$(pick_resume_pth "$HEAD_RUN_NAME"); then
    CLASS_LABELS+=("head .pth")
    CLASS_LOCAL+=("$HEAD_PTH")
    CLASS_REMOTE+=("${REMOTE_CKPT_DIR}/$(basename "$HEAD_PTH")")
    CLASS_MISSING_OK+=(0)
else
    CLASS_LABELS+=("head .pth")
    CLASS_LOCAL+=("${LOCAL_CKPT}/${HEAD_RUN_NAME}_FINAL.pth (or _best.pth)")
    CLASS_REMOTE+=("(would push)")
    CLASS_MISSING_OK+=(0)
fi

if HEAD_OPT=$(pick_optimizer_pth "$HEAD_RUN_NAME"); then
    CLASS_LABELS+=("head optimizer")
    CLASS_LOCAL+=("$HEAD_OPT")
    CLASS_REMOTE+=("${REMOTE_CKPT_DIR}/$(basename "$HEAD_OPT")")
    CLASS_MISSING_OK+=(0)
else
    # Head optimizer is required only when we have a head .pth to resume from.
    CLASS_LABELS+=("head optimizer")
    CLASS_LOCAL+=("${LOCAL_CKPT}/${HEAD_RUN_NAME}_FINAL_optimizer.pth (or _best_optimizer.pth)")
    CLASS_REMOTE+=("(would push)")
    CLASS_MISSING_OK+=(0)
fi

CLASS_LABELS+=("head losses CSV")
CLASS_LOCAL+=("${LOCAL_CKPT}/${HEAD_RUN_NAME}_losses.csv")
CLASS_REMOTE+=("${REMOTE_CKPT_DIR}/${HEAD_RUN_NAME}_losses.csv")
CLASS_MISSING_OK+=(0)

# Run log (always — in both modes; in --head-only this is THE critical one
# that today's incident missed)
CLASS_LABELS+=("run log")
CLASS_LOCAL+=("$LOCAL_LOG")
CLASS_REMOTE+=("$REMOTE_LOG")
CLASS_MISSING_OK+=(0)

###############################################################################
# Header summary
###############################################################################

echo "=== push_resume_bundle.sh ==="
echo "  local_arm_dir : $LOCAL_ARM_DIR"
echo "  ssh_target    : $SSH_TARGET"
echo "  ssh_port      : $SSH_PORT"
if [[ $HEAD_ONLY -eq 0 ]]; then
    echo "  bb_run_name   : $BB_RUN_NAME"
fi
echo "  head_run_name : $HEAD_RUN_NAME"
echo "  arm_label     : $ARM_LABEL"
echo "  remote_log    : $REMOTE_LOG"
echo "  mode          : $([[ $HEAD_ONLY -eq 1 ]] && echo head-only || echo full)$([[ $DRY_RUN -eq 1 ]] && echo ' [dry-run]' || echo '')"
echo ""

###############################################################################
# Push each class atomically
###############################################################################

# atomic_push_one: scp local→<remote>.tmp, then ssh "mv <remote>.tmp <remote>".
# - The remote-side mv is atomic (same filesystem), so a launching
#   trainer never sees a partial file.
# - On scp failure or zero-byte transfer, the remote .tmp is removed
#   and the existing remote file (if any) is preserved untouched.
# - Returns 0 on success, non-zero on failure (caller logs ✗).
atomic_push_one() {
    local local_path="$1"
    local remote_path="$2"
    local remote_tmp="${remote_path}.tmp"

    if [[ $DRY_RUN -eq 1 ]]; then
        local sz
        sz=$(stat -f%z "$local_path" 2>/dev/null || stat -c%s "$local_path" 2>/dev/null || echo "?")
        echo "  ✓ [dry-run] ${local_path} → ${SSH_TARGET}:${remote_path} (${sz} bytes)"
        return 0
    fi

    # The scp itself
    if ! scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -P "$SSH_PORT" "$local_path" "${SSH_TARGET}:${remote_tmp}"; then
        echo "  ✗ scp FAILED ${local_path} → ${SSH_TARGET}:${remote_tmp}" >&2
        # Best-effort cleanup; ignore failure (the .tmp may not have landed)
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -p "$SSH_PORT" "$SSH_TARGET" "rm -f $(printf '%q' "$remote_tmp")" 2>/dev/null || true
        return 1
    fi

    # Verify the .tmp is non-zero on the remote, then mv into place.
    if ! ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -p "$SSH_PORT" "$SSH_TARGET" \
            "test -s $(printf '%q' "$remote_tmp") && mv -f $(printf '%q' "$remote_tmp") $(printf '%q' "$remote_path")"; then
        echo "  ✗ remote-side verify+mv FAILED for ${remote_path}" >&2
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -p "$SSH_PORT" "$SSH_TARGET" "rm -f $(printf '%q' "$remote_tmp")" 2>/dev/null || true
        return 1
    fi

    local sz
    sz=$(stat -f%z "$local_path" 2>/dev/null || stat -c%s "$local_path" 2>/dev/null || echo "?")
    echo "  ✓ ${local_path} → ${SSH_TARGET}:${remote_path} (${sz} bytes)"
    return 0
}

# Pre-flight: collect missing files first, exit before any push if any are missing.
MISSING_COUNT=0
for ((i=0; i<${#CLASS_LABELS[@]}; i++)); do
    label="${CLASS_LABELS[$i]}"
    local_path="${CLASS_LOCAL[$i]}"
    if [[ "${CLASS_REMOTE[$i]}" == "(would push)" ]]; then
        # The pick_*_pth helper failed to find a valid local file.
        echo "  ✗ MISSING ${label}: expected ${local_path}" >&2
        MISSING_COUNT=$((MISSING_COUNT + 1))
    elif [[ ! -e "$local_path" ]]; then
        echo "  ✗ MISSING ${label}: ${local_path} does not exist locally" >&2
        MISSING_COUNT=$((MISSING_COUNT + 1))
    elif [[ ! -s "$local_path" ]]; then
        echo "  ✗ EMPTY ${label}: ${local_path} is zero bytes" >&2
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

if [[ $MISSING_COUNT -gt 0 ]]; then
    echo "" >&2
    echo "ERROR: ${MISSING_COUNT} missing/empty file(s); aborting before any push." >&2
    echo "       Verify the local arm dir is correct and re-run." >&2
    exit 1
fi

# All present — now push each.
PUSH_FAILED=0
for ((i=0; i<${#CLASS_LABELS[@]}; i++)); do
    label="${CLASS_LABELS[$i]}"
    local_path="${CLASS_LOCAL[$i]}"
    remote_path="${CLASS_REMOTE[$i]}"
    echo "[push] ${label}"
    if ! atomic_push_one "$local_path" "$remote_path"; then
        PUSH_FAILED=$((PUSH_FAILED + 1))
    fi
done

echo ""
if [[ $PUSH_FAILED -gt 0 ]]; then
    echo "ERROR: ${PUSH_FAILED} push(es) failed. Re-run after diagnosing." >&2
    exit 1
fi

echo "=== push_resume_bundle.sh DONE ($([[ $HEAD_ONLY -eq 1 ]] && echo 'head-only' || echo 'full')$([[ $DRY_RUN -eq 1 ]] && echo ', dry-run' || echo '')) ==="
exit 0
