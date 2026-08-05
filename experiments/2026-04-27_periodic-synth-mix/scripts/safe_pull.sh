#!/bin/bash
# Atomic, rotation-aware single-file pull from a vast.ai instance.
#
# Reason for existing: a raw `scp` call (or any tool that writes directly
# to the destination path) leaves a partial file behind on connection
# drops. PR #45 lost a RevIN backbone this way: 80 MB expected, 2 MB
# arrived, scp left the 2 MB in place and overwrote the destination —
# unrecoverable.
#
# This wrapper:
#   1. scp to ${dst}.tmp
#   2. verify the .tmp is at least --min-bytes large (default 1 MB)
#   3. if ${dst} exists, rotate to ${dst}.prev (1-deep backup)
#   4. atomic mv ${dst}.tmp → ${dst}
# A failed transfer is detected (file too small or empty) and leaves
# the existing ${dst} intact.
#
# Usage:
#   safe_pull.sh <ssh_host> <ssh_port> <remote_path> <local_dst> [min_bytes]
#
# Connects as $SSH_USER, default `root` — vast.ai instances, which this
# was written for. Set SSH_USER=jupyter for elisa.

set -u

SSH_HOST=${1:?usage: safe_pull.sh <host> <port> <remote_path> <local_dst> [min_bytes]}
SSH_PORT=${2:?missing port}
REMOTE=${3:?missing remote_path}
LOCAL=${4:?missing local_dst}
MIN_BYTES=${5:-1048576}   # default: 1 MB
SSH_USER=${SSH_USER:-root}

mkdir -p "$(dirname "$LOCAL")"

scp -q -o StrictHostKeyChecking=no -P "$SSH_PORT" \
    "${SSH_USER}@${SSH_HOST}:${REMOTE}" "${LOCAL}.tmp"

if [[ ! -s "${LOCAL}.tmp" ]]; then
    rm -f "${LOCAL}.tmp"
    echo "✗ ${REMOTE}: empty or missing — destination preserved" >&2
    exit 1
fi
SZ=$(wc -c < "${LOCAL}.tmp")
if (( SZ < MIN_BYTES )); then
    rm -f "${LOCAL}.tmp"
    echo "✗ ${REMOTE}: ${SZ} bytes < min ${MIN_BYTES} — destination preserved" >&2
    exit 1
fi
# Rotate
if [[ -s "$LOCAL" ]]; then
    mv -f "$LOCAL" "${LOCAL}.prev"
fi
mv "${LOCAL}.tmp" "$LOCAL"
echo "✓ ${REMOTE} → ${LOCAL} (${SZ} bytes)"
