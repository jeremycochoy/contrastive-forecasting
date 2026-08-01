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

bb_name() {  # arm -> backbone run name
  local arm="$1" family
  case "$arm" in
    arm6_v2|arm6_v2_*) family="lrepmoco" ;;
    arm5|arm5_*)       family="lrep" ;;
    *) echo "bb_name: unknown arm '$arm'" >&2; return 2 ;;
  esac
  echo "bb_small_${arm}_lalign_${family}_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher"
}
