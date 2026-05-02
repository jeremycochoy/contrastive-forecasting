#!/bin/bash
# Regression test for the append-only length guard added to atomic_scp in
# every sync_*/sync_loop.sh.
#
# Failure mode under guard (see docs/SYNC_PROTOCOL_REVIEW.md §2.1):
# fresh remote serves a short CSV/log; the previous unguarded atomic_scp
# would rotate the long good local copy to .prev and a second cycle would
# overwrite .prev. The patched function:
#   - if remote_lines >= local_lines: behaves as before (rotate to .prev,
#     accept).
#   - if remote_lines <  local_lines: leaves the local file & .prev intact
#     and moves the new file to <LOCAL_BASE>/archive/<stamp>_<bn>.regression
#     and emits a "⚠️ APPEND REGRESSION" line to the log.
#
# This test mocks scp by overriding it with a `cp` shim; no network access.
# Run as: bash tests/test_sync_loop_length_guard.sh
# Exits 0 on success, non-zero on any assertion failure.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_UNDER_TEST="${REPO_ROOT}/sync_realonly_4096_smaller_learnable_tau/sync_loop.sh"

if [ ! -f "${SCRIPT_UNDER_TEST}" ]; then
    echo "FAIL: cannot find ${SCRIPT_UNDER_TEST}" >&2
    exit 2
fi

TMPROOT=$(mktemp -d -t sync_loop_guard_test.XXXXXX)
trap 'rm -rf "${TMPROOT}"' EXIT

# --- Per-test environment that atomic_scp expects from the surrounding script. ---
export LOCAL_BASE="${TMPROOT}/local_base"
export LOG_FILE="${LOCAL_BASE}/sync.log"
export REMOTE="root@example.invalid"
export SSH_PORT="22"
export SSH_OPTS=""
mkdir -p "${LOCAL_BASE}/checkpoints"
: > "${LOG_FILE}"

REMOTE_DIR="${TMPROOT}/remote/checkpoints"
mkdir -p "${REMOTE_DIR}"

# --- scp shim. ---
# The patched atomic_scp invokes:
#   scp ${SSH_OPTS} -P "${SSH_PORT}" -q "${REMOTE}:${remote_path}" "${tmp}"
# We override that with a function that ignores all flags, takes the last
# two positional args (the source spec and the local destination), strips
# "user@host:" off the source, and copies from REMOTE_DIR by basename.
# This matches the test's mock-remote layout (REMOTE_DIR keyed by file
# basename) without requiring a real ssh/scp.
scp() {
    local src dst bn
    dst="${@: -1}"
    src="${@: -2:1}"
    src="${src#*:}"   # strip user@host: prefix
    bn="$(basename "${src}")"
    if [ -f "${REMOTE_DIR}/${bn}" ]; then
        command cp "${REMOTE_DIR}/${bn}" "${dst}"
        return 0
    fi
    return 1
}
export -f scp

# --- Source atomic_scp (and only atomic_scp) from the patched script. ---
# The function spans from `atomic_scp() {` through the closing `}` on a
# line of its own. awk extracts it; we eval into the current shell.
ATOMIC_SCP_SRC=$(awk '/^atomic_scp\(\) \{/,/^\}$/' "${SCRIPT_UNDER_TEST}")
if [ -z "${ATOMIC_SCP_SRC}" ]; then
    echo "FAIL: could not extract atomic_scp from ${SCRIPT_UNDER_TEST}" >&2
    exit 2
fi
eval "${ATOMIC_SCP_SRC}"

# --- Helpers. ---
make_lines() {
    # make_lines <count> <prefix>
    local n="$1" prefix="${2:-line}"
    local i=1
    while [ "$i" -le "$n" ]; do
        echo "${prefix}_${i}"
        i=$((i + 1))
    done
}

assert_eq() {
    # assert_eq <actual> <expected> <message>
    if [ "$1" != "$2" ]; then
        echo "FAIL: ${3} — got '$1', expected '$2'" >&2
        exit 1
    fi
}

assert_file_exists() {
    # assert_file_exists <path> <message>
    if [ ! -f "$1" ]; then
        echo "FAIL: ${2} — file not found: $1" >&2
        exit 1
    fi
}

assert_no_file() {
    # assert_no_file <path> <message>
    if [ -e "$1" ]; then
        echo "FAIL: ${2} — file unexpectedly present: $1" >&2
        exit 1
    fi
}

count_lines() {
    wc -l < "$1" | tr -d ' '
}

REMOTE_CSV_PATH="/workspace/app/checkpoints/tiny_test_losses.csv"
LOCAL_CSV="${LOCAL_BASE}/checkpoints/tiny_test_losses.csv"

# ============================================================
# Phase 1 — healthy growth.
# Remote has 100 lines, local has 100 lines (header + 99 rows).
# Expect: rotation to .prev (remote >= local), local stays 100 lines.
# ============================================================
echo "--- Phase 1: 100 remote vs 100 local (equal lines, must accept and rotate) ---"
make_lines 100 csv > "${REMOTE_DIR}/tiny_test_losses.csv"
make_lines 100 csv > "${LOCAL_CSV}"

atomic_scp "${REMOTE_CSV_PATH}" "${LOCAL_CSV}" 1
rc=$?
assert_eq "${rc}" "0" "atomic_scp returned non-zero on equal-length pull"
assert_file_exists "${LOCAL_CSV}" "local CSV missing after equal-length pull"
local_lines_p1=$(count_lines "${LOCAL_CSV}")
assert_eq "${local_lines_p1}" "100" "local CSV line count changed after equal-length pull"
# .prev should now exist with the previous 100-line content (rotation took place).
assert_file_exists "${LOCAL_CSV}.prev" ".prev not created after equal-length pull"
prev_lines_p1=$(count_lines "${LOCAL_CSV}.prev")
assert_eq "${prev_lines_p1}" "100" ".prev line count wrong after equal-length pull"
# No archive should have been produced for an equal-length pull.
if [ -d "${LOCAL_BASE}/archive" ]; then
    if compgen -G "${LOCAL_BASE}/archive/*tiny_test_losses.csv*.regression" > /dev/null; then
        echo "FAIL: archive/*.regression unexpectedly created during healthy pull" >&2
        exit 1
    fi
fi

# ============================================================
# Phase 2 — fresh-instance regression.
# Remote shrinks to 5 lines (simulating a fresh remote that started a new
# CSV from scratch). Local is still 100 lines from Phase 1.
# Expect: local CSV stays 100 lines, .prev stays 100 lines, archive/
# contains a *.regression file with 5 lines, sync.log shows "APPEND
# REGRESSION".
# ============================================================
echo "--- Phase 2: 5 remote vs 100 local (regression must NOT overwrite local) ---"
make_lines 5 csv > "${REMOTE_DIR}/tiny_test_losses.csv"
# Sleep 1s so the archive timestamp is distinct (defensive; not strictly
# required since Phase 1 produced no archive).
sleep 1
log_size_before=$(wc -c < "${LOG_FILE}" | tr -d ' ')
atomic_scp "${REMOTE_CSV_PATH}" "${LOCAL_CSV}" 1
rc=$?
# Per spec: regression branch returns 0 (soft failure, surfaced via log).
assert_eq "${rc}" "0" "atomic_scp must return 0 on append regression (soft fail)"

# Local CSV must be unchanged.
assert_file_exists "${LOCAL_CSV}" "local CSV vanished during regression"
local_lines_p2=$(count_lines "${LOCAL_CSV}")
assert_eq "${local_lines_p2}" "100" "local CSV got truncated during regression"

# .prev must be unchanged (still 100 lines, not the 5-line shrink).
assert_file_exists "${LOCAL_CSV}.prev" ".prev vanished during regression"
prev_lines_p2=$(count_lines "${LOCAL_CSV}.prev")
assert_eq "${prev_lines_p2}" "100" ".prev got overwritten during regression"

# Archive must contain a *.regression with 5 lines.
shopt -s nullglob
archive_files=("${LOCAL_BASE}/archive/"*tiny_test_losses.csv*.regression)
shopt -u nullglob
if [ "${#archive_files[@]}" -lt 1 ]; then
    echo "FAIL: no archive/*tiny_test_losses.csv.regression file produced" >&2
    ls -la "${LOCAL_BASE}/archive/" 2>&1 >&2
    exit 1
fi
archive_lines=$(count_lines "${archive_files[0]}")
assert_eq "${archive_lines}" "5" "archive .regression has wrong line count"

# Log must contain the warning marker.
if ! grep -q "APPEND REGRESSION" "${LOG_FILE}"; then
    echo "FAIL: sync.log missing 'APPEND REGRESSION' line" >&2
    echo "----- sync.log -----" >&2
    cat "${LOG_FILE}" >&2
    exit 1
fi
log_size_after=$(wc -c < "${LOG_FILE}" | tr -d ' ')
if [ "${log_size_after}" -le "${log_size_before}" ]; then
    echo "FAIL: sync.log did not grow during regression cycle" >&2
    exit 1
fi

# ============================================================
# Phase 3 — recovery growth after regression.
# Remote eventually grows past local (e.g. 200 lines). The local CSV must
# be rotated cleanly and the new file accepted; the previous archive
# remains intact.
# ============================================================
echo "--- Phase 3: 200 remote vs 100 local (growth after regression must rotate cleanly) ---"
make_lines 200 csv > "${REMOTE_DIR}/tiny_test_losses.csv"
atomic_scp "${REMOTE_CSV_PATH}" "${LOCAL_CSV}" 1
rc=$?
assert_eq "${rc}" "0" "atomic_scp returned non-zero on healthy growth"
local_lines_p3=$(count_lines "${LOCAL_CSV}")
assert_eq "${local_lines_p3}" "200" "local CSV did not grow on healthy pull"
prev_lines_p3=$(count_lines "${LOCAL_CSV}.prev")
assert_eq "${prev_lines_p3}" "100" ".prev should now hold the previous 100-line copy"

# Archive from Phase 2 must still exist.
shopt -s nullglob
archive_files_p3=("${LOCAL_BASE}/archive/"*tiny_test_losses.csv*.regression)
shopt -u nullglob
if [ "${#archive_files_p3[@]}" -lt 1 ]; then
    echo "FAIL: archive/*regression vanished after recovery cycle" >&2
    exit 1
fi

echo ""
echo "PASS: append-regression length guard works in all 3 phases."
exit 0
