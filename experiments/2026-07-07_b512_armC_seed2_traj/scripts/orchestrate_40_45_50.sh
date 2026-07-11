#!/bin/bash
# #371 — orchestrate head+eval cells at loci 40000/45000/50000 as their
# backbone trajectory checkpoints land. Launches 2L and 6L per locus in
# parallel on ONE GPU (light memory footprint: ~4 GB head-train + ~4 GB
# eval each; two cells stay under 12 GB). Loci run sequentially to avoid
# stacking 6 cells at once on the free GPU.
#
# GPU choice per locus:
#   step40000, step45000 — GPU 0 (GPU 1 is still training backbone)
#   step50000            — GPU 1 (backbone done, both GPUs free)
#
#   WT=... EXP=... SYNC=... bash orchestrate_40_45_50.sh
set -uo pipefail
: "${WT:?}"; : "${EXP:?}"; : "${SYNC:?}"
LOG="$EXP/results/orchestrate_40_45_50.log"
NAME="bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_lC_emb10_enc10_tau090_seed2"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch] $*" | tee -a "$LOG"; }

wait_bb(){ # STEP → 0 when checkpoint on disk
  local step="$1"
  for _ in $(seq 1 720); do  # 720 * 60s = 12h ceiling
    for cand in "$SYNC/${NAME}_step${step}.pth" "$SYNC/${NAME}_r"*"_step${step}.pth"; do
      [ -f "$cand" ] && return 0
    done
    sleep 60
  done
  log "TIMEOUT waiting for step${step}"; return 1
}

run_locus(){ # STEP GPU
  local step="$1" gpu="$2"
  wait_bb "$step" || return 1
  log "step${step} backbone landed — 2L + 6L on GPU ${gpu}"
  # Both depths in parallel via dl_one_cell.sh's atomic claim mechanism
  WT="$WT" EXP="$EXP" SYNC="$SYNC" bash "$EXP/scripts/dl_one_cell.sh" 2 "$gpu" "$step" \
    >>"$LOG" 2>&1 &
  WT="$WT" EXP="$EXP" SYNC="$SYNC" bash "$EXP/scripts/dl_one_cell.sh" 6 "$gpu" "$step" \
    >>"$LOG" 2>&1 &
  wait
  log "step${step} both depths complete"
}

run_locus 40000 0
run_locus 45000 0
run_locus 50000 1
log "orchestration complete"
