#!/bin/bash
# #425 — the linear queue on elisa, on its two GPUs. This is the main path of
# the linear heads. queue.sh on the box (CF425_HEAD_ARCH=linear) is the
# fallback.
#
# It is queue.sh with the folders, the lanes and the limits of elisa:
#
#   * Every file is under ~/checkpoints_backup/cf-425-lin, so a restart of
#     elisa keeps it: the code that the queue reads (code/, from
#     deploy_elisa.sh), the heads and the files of each score (ckpt/recon),
#     and the scores, the logs, the locks and the eval slots (results/).
#     The path of each process of the queue shows cf-425-lin.
#   * Four lanes, two for each GPU: one trainer process keeps a RTX 4090 busy
#     72% of the time (probe.sh, 10-07), so two lanes use a GPU fully. Three
#     lanes take their waves from GiftEvalPretrain. The fourth lane trains
#     the old-data jobs in waves of 10 (the 10 jobs of ABC make one wave),
#     then it also takes GiftEvalPretrain waves.
#   * Waves of 6 jobs on GiftEvalPretrain. The frozen backbone sets the rate
#     of a job, so the size of a wave changes no rate, and elisa pays no
#     download. A small wave gives its scores early, loses little at a stop,
#     and gives its GPU back early.
#   * 4 CPU threads for each trainer. A score has 2 shards with one thread
#     each, and each GPU runs 2 scores at a time. So the queue takes no more
#     than 24 of the 32 cores.
#   * GPU memory, for each GPU of 24 GB: 2 waves of about 6 GB, and 2 scores
#     of 4.8 GB.
#
# The GPUs of elisa are shared. To give GPU <N> back at the end of its waves:
#   touch ~/checkpoints_backup/cf-425-lin/results/stop_gpu<N>
# To get it at once, write that file, and then stop the trainers of that GPU:
#   nvidia-smi --id=<N> --query-compute-apps=pid --format=csv,noheader
# gives their process numbers, and `ps -o cmd -p <number>` shows cf-425-lin.
# A wave that stops in this way costs its jobs no try.
#
# After a stop or a restart of elisa, remove the stop file and give the same
# command: a job with a score is done, a job with a head gets its score, and
# the other jobs train again from step 1.
#
# Usage, on elisa:
#   bash deploy_elisa.sh            # one time, and after each code change
#   S=~/checkpoints_backup/cf-425-lin/code/reports/2026-10-06_encoder_reconstruction/scripts
#   mkdir -p ~/checkpoints_backup/cf-425-lin/results
#   nohup setsid bash $S/queue_elisa.sh \
#     >>~/checkpoints_backup/cf-425-lin/results/queue.log 2>&1 </dev/null &
#   CF425_DRY_RUN=1 bash $S/queue_elisa.sh       # the inputs, and the work of a
#                                                # start now
#   HEAD_LOG_EVERY=50 bash $S/queue_elisa.sh probe 150   # probe.sh, 4 lanes
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${CF425_ELISA_BASE:-$HOME/checkpoints_backup/cf-425-lin}"
export CF425_HEAD_ARCH=linear
export CF425_CODE="${CF425_CODE:-$BASE/code}"
export CF425_CK="${CF425_CK:-$HOME/checkpoints_backup/cf-412/vast_lr100x}"
export CF425_ROOT="${CF425_ROOT:-$BASE/ckpt/recon}"
export CF425_RES="${CF425_RES:-$BASE/results}"
export GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}"
export CF425_LANES="${CF425_LANES:-gift_pretrain:0 gift_pretrain:0 gift_pretrain:1 old,gift_pretrain:1}"
export CF425_WAVE_SIZE="${CF425_WAVE_SIZE:-6}"
export CF425_OLD_WAVE_SIZE="${CF425_OLD_WAVE_SIZE:-10}"
export CF425_TRAIN_THREADS="${CF425_TRAIN_THREADS:-4}"
export CF425_EVAL_SLOTS="${CF425_EVAL_SLOTS:-2}"
export CF425_WAVE_VRAM_MIB="${CF425_WAVE_VRAM_MIB:-6500}"
export CF425_SCORE_VRAM_MIB="${CF425_SCORE_VRAM_MIB:-4800}"
export CF425_LANE_STAGGER="${CF425_LANE_STAGGER:-30}"
# The locks and the eval slots stay with the results, not in /tmp.
export CF425_GPU_LOCK="${CF425_GPU_LOCK:-$CF425_RES/locks/gpu_start.lock}"
export CF425_EVAL_SLOTDIR="${CF425_EVAL_SLOTDIR:-$CF425_RES/evalslots}"

if [ "${1:-}" = probe ]; then
  shift
  export CF425_PROBE_RES="${CF425_PROBE_RES:-$BASE/results/probe}"
  export CF425_PROBE_ROOT="${CF425_PROBE_ROOT:-$BASE/ckpt/probe}"
  export CF425_PROBE_WAVES="${CF425_PROBE_WAVES:-3}"
  exec bash "$HERE/probe.sh" "$@"
fi
exec bash "$HERE/queue.sh"
