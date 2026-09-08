#!/bin/bash
# #412 pass 3 — the pre-registered gate. It writes `results/gate_pass3.txt`.
#
# GM-RELATIVE MASE DECIDES, AND NOTHING ELSE. The contrastive AUC does not
# rank arms. Over the ten scored arms of this card at the 40,000-step stop, a
# higher AUC goes with a WORSE score inside both reduction groups: Pearson
# +0.533 over the seven sum arms and +0.954 over the three mean arms. The best
# arm, `k3_r100_09_lr56` at 1.1820, holds the LOWEST AUC of the seven sum
# arms. So the AUC has ONE use here: the guard keeps its 0.55 threshold and
# stops a run that reached chance, because a dead run gives no score.
#
# THE REFERENCE  1.1820, `k3_r100_09_lr56` at 40,000 steps, the best of the
#                card.
# THE BAND       0.0649, the two-seed spread at 5.6e-4
#                (`k3_r100_09b_lr56` 1.2469 against 1.1820). Pass 1 measured
#                0.0568 at 1e-3. Pass 3 ranks at 5.6e-4, so it uses 0.0649.
# PROGRESS       score < 1.1820 - 0.0649 = 1.1171.
# WORSE          score > 1.1820 + 0.0649 = 1.2469.
# TIED           between the two, and this card does not rank it.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
REF=1.1820
BAND=0.0649
ARMS="${CF412_PASS3_ARMS:-k32_r100_09_sum k3_r100_09_lr45 k3_r100_09_lr70}"
OUT="$CF412_RESULTS/gate_pass3.txt"
{
  echo "--- the rule ---"
  echo "  reference   $REF   k3_r100_09_lr56 at 40,000 steps"
  echo "  band        $BAND   two seeds at 5.6e-4"
  printf "  progress    score < %.4f\n" "$(echo "$REF - $BAND" | bc -l)"
  printf "  worse       score > %.4f\n" "$(echo "$REF + $BAND" | bc -l)"
  echo "  the AUC says only whether the guard stopped a run."
  echo
  echo "--- the pass-3 arms ---"
  for a in $ARMS; do
    s="$CF412_RESULTS/score_${a}_bb40k_h30k_student.txt"
    if [ -f "$CF412_RESULTS/collapsed_$a.txt" ]; then
      printf "  %-18s GUARD STOPPED IT — no score\n" "$a"
    elif [ -s "$s" ]; then
      v="$(cat "$s")"
      awk -v a="$a" -v v="$v" -v r="$REF" -v b="$BAND" 'BEGIN{
        if (v < r - b)      printf "  %-18s %s   PROGRESS — beats %.4f by more than the band\n", a, v, r;
        else if (v > r + b) printf "  %-18s %s   WORSE — behind %.4f by more than the band\n", a, v, r;
        else                printf "  %-18s %s   TIED with %.4f — inside the band\n", a, v, r;
      }'
    else
      printf "  %-18s -        no score yet\n" "$a"
    fi
  done
} > "$OUT"
cat "$OUT"
