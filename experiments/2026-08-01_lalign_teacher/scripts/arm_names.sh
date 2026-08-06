#!/bin/bash
# #390 — the arm → backbone-run-name mapping, in one place.
#
# `run_arm.sh` spells each NAME out in its case block, because that block is
# compared token-by-token against #379's launcher. Everything downstream
# (monitor.sh, sync/sync_loop.sh) needs the same names, and a mapping copied
# by hand into three files is how a sync loop silently stops pulling an arm
# (CLAUDE.md § Remote Machine Monitoring: verify by `ls`, not by log).
#
# So derive them instead. The names follow one rule:
#
#   bb_small_<arm>_lalign_<family>_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher
#
# where <family> is `lrep` for the arm5 cells and `lrepmoco` for the arm6_v2
# cells (the ones that add `--moco-rep-keys`). The arm token is always
# followed by `_lalign`, so `arm5` and `arm5_tr1` stay distinct.
#
# `tests/test_390_launcher_shape.py` checks bb_name() against run_arm.sh's
# case block for all 10 arms, so drift fails the suite rather than a wave.
#
# Usage:  source arm_names.sh;  name=$(bb_name arm6_v2_tr1)

# The 10 cells of #390: the two L_align arms × the five #379 settings.
CF390_ARMS=(arm5 arm5_tr1 arm5_nse arm5_ncpc arm5_combab
            arm6_v2 arm6_v2_tr1 arm6_v2_nse arm6_v2_ncpc arm6_v2_combab)

# The trailing name token. `alignteacher` is the experiment; the student
# control (`run_arm_student.sh`, review item 3) writes its checkpoints under
# `alignstudent` and reads them back by exporting this. Defaulted, so every
# existing caller resolves the same names it always did.
CF390_NAME_SUFFIX="${CF390_NAME_SUFFIX:-alignteacher}"

bb_name() {  # arm -> backbone run name
  local arm="$1" family
  case "$arm" in
    arm6_v2|arm6_v2_*) family="lrepmoco" ;;
    arm5|arm5_*)       family="lrep" ;;
    *) echo "bb_name: unknown arm '$arm'" >&2; return 2 ;;
  esac
  echo "bb_small_${arm}_lalign_${family}_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_${CF390_NAME_SUFFIX}"
}

# The issue's three waves, in one place:
#
#   wave 1 |  40 000 backbone steps, save every 10 000
#   wave 2 | 100 000 backbone steps, save every 25 000
#   wave 3 | 200 000 backbone steps, save every 25 000 (the arm's true end)
#
# orchestrate.sh drives a wave from this table, monitor.sh takes the wave's
# step budget from it, and sync/sync_loop.sh derives from it the checkpoint
# steps it pulls. Same reason as the names above: a cadence retyped in three
# files is how a sync loop silently stops pulling the snapshot the next wave
# resumes from.
CF390_WAVE_TARGET_STEPS=(0 40000 100000 200000)   # index = wave number
CF390_WAVE_SAVE_EVERY=(0 10000 25000 25000)
CF390_EXTRA_SAVES=2500
CF390_FINAL_STEPS=200000

wave_checkpoint_steps_k() {  # the _<N>k.pth snapshots the three waves leave
  local w every target s start=0
  echo $(( CF390_EXTRA_SAVES / 1000 ))
  for w in 1 2 3; do
    every="${CF390_WAVE_SAVE_EVERY[$w]}"
    target="${CF390_WAVE_TARGET_STEPS[$w]}"
    # train.py counts save-every on the global step, so a wave resuming at
    # 40k with save-every 25 000 saves at 50k, not 65k.
    for (( s = every; s <= target; s += every )); do
      if [ "$s" -gt "$start" ]; then echo $(( s / 1000 )); fi
    done
    start="$target"
  done
}
