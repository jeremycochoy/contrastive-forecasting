#!/bin/bash
# #373 review gap 6 — A3's bb200k student head, drawn a second time.
#
# A3 at bb200k reads student 1.3998 against teacher 1.2913. The two heads sit
# 0.1085 apart on ONE backbone at ONE stop, 2.8x the +-0.0384 head-seed band,
# and every other group-A student/teacher gap in the study is 0.0141 or less.
# A3's student is also the only non-monotone trajectory in the ladder.
#
# So the study cannot tell a bad head draw from a real 200k regression. One
# more draw of the SAME head on the SAME backbone tells them apart, and it
# costs no backbone time: 30,000 head steps and one 97-config eval.
#
# Everything is head_eval_bb.sh's, unchanged: same trainer, same 30,000-step
# budget the bb100k and bb200k columns use, same --grad-clip 1.0, same 97
# configs under the official B4 strategy. Only --seed moves.
#
# The seed is 20260723, the second of the three head seeds `ema_sched_ladder`
# drew for the band this study quotes. A seed from that set keeps the second
# draw inside the family the band was measured over.
#
# Usage: gap6_a3_reseed.sh   (HEAD_SEED and BB_GPU override)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WT="${WT:-/home/jupyter/wt-cf-373-train}"
export BB_GPU="${BB_GPU:-1}"
export HEAD_SEED="${HEAD_SEED:-20260723}"

# A3 = arm6_v2_combab_alignT, group A, so run_leg_k.sh's layout. Round 3 wrote
# its 200k leg into the sync tree and the reap verified it there.
BB="${BB:-/home/jupyter/cf373_r3/sync/arm6_v2_combab_alignT/leg_200k/cf393_arm6_v2_combab_alignT_cf373k3_200k.pth}"
TAG="A3_k3_bb200k_student_s${HEAD_SEED}"

[ -f "$BB" ] || { echo "ABORT: no A3 bb200k backbone at $BB" >&2; exit 3; }

CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"
OUT="$CF373_ROOT/eval/$TAG"
mkdir -p "$OUT"

# The first draw read this file. Record which one this draw read, so the two
# rows are known to differ by the head seed and by nothing else.
md5sum "$BB" | tee "$OUT/backbone_md5.txt"

exec bash "$HERE/head_eval_bb.sh" "$TAG" "$BB" student 30000
