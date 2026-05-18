#!/bin/bash
# Wait for tau=0.10 100k extension to finish, then launch tau=0.20 100k.
# Polls every 60s; relies on the FINAL.pth marker created at the end of
# run_square_tau_0_10_resume_100k.sh.

set -e
MARKER=/home/jupyter/contrastive-forecasting/sync_loss_ext_square/checkpoints/loss_ext_square_tau_0_10_100k_FINAL.pth

echo "Waiting for ${MARKER}..."
while [ ! -f "${MARKER}" ]; do
    sleep 60
done

echo "tau=0.10 100k complete. Launching tau=0.20 100k on GPU 1."
date

CUDA_VISIBLE_DEVICES=1 bash /tmp/run_square_tau_0_20_resume_100k.sh
