#!/bin/bash
# #412 pass 4 — the verdict, from the scores on disk.
#
# The card asks ONE question: does an arm get BETTER with a longer stop? Every
# arm of this card so far gets worse. So the gate compares each arm against
# ITSELF at 40,000 steps, and then against the no-decay reference at the same
# stop.
#
# It ranks on GM-Relative MASE alone. LOWER IS BETTER. The contrastive AUC does
# not rank arms on this card, so it enters here for one purpose: to say whether
# the guard stopped a run.
#
# The three references come from `k3_r100_09_lr56`, the same cell with no
# decay, and the band is the 5.6e-4 seed band of `k3_r100_09b_lr56`.
#
# Usage:  bash scripts/pass4_gate.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

ARMS="${CF412_PASS4_ARMS:-k3_r100_09_lr56_dec k3_r100_09_lr56_dec10k}"
BAND="${CF412_PASS4_BAND:-0.0649}"
OUT="${CF412_PASS4_GATE_OUT:-$CF412_RESULTS/gate_pass4.txt}"
mkdir -p "$CF412_RESULTS"

ref_of(){  # <stop>
  case "$1" in 40000) echo 1.1820 ;; 100000) echo 1.3170 ;;
               200000) echo 1.3189 ;; *) echo - ;; esac
}
score_of(){  # <arm> <stop>
  local f; f="$(cf412_score_file "$1" "$2")"
  [ -s "$f" ] && tr -d ' \n' <"$f" || echo -
}

{
echo "#412 pass 4 — the L_rep decay at 5.6e-4, over a long stop"
echo "generated $(date '+%Y-%m-%d %H:%M:%S')"
echo "GM-Relative MASE, lower is better. Seed band at 5.6e-4: $BAND."
echo "Reference: k3_r100_09_lr56, the same cell with no decay."
echo
printf '%-26s %-8s %9s %9s %9s %s\n' ARM STOP SCORE REF "DELTA" NOTE
for arm in $ARMS; do
  for stop in 40000 100000 200000; do
    s="$(score_of "$arm" "$stop")"
    [ "$s" = "-" ] && [ ! -f "$(cf412_leg_dir "$arm" "$stop")" ] && \
      { [ "$arm" = "k3_r100_09_lr56_dec10k" ] && [ "$stop" = 200000 ] && continue; }
    r="$(ref_of "$stop")"
    d="-"; note="-"
    if [ "$s" != "-" ]; then
      d="$(awk -v a="$s" -v b="$r" 'BEGIN{printf "%+.4f", a-b}')"
      note="$(awk -v a="$s" -v b="$r" -v w="$BAND" 'BEGIN{
        x=a-b; if (x<-w) print "beats the reference";
        else if (x>w) print "loses to the reference"; else print "inside the band"}')"
    elif [ -f "$(cf412_collapse_file "$arm")" ]; then
      note="the AUC guard stopped this arm"
    else
      note="not scored yet"
    fi
    printf '%-26s %-8s %9s %9s %9s %s\n' "$arm" "$stop" "$s" "$r" "$d" "$note"
  done
done

echo
echo "THE QUESTION: does a longer stop beat the 40,000-step stop of the SAME arm?"
for arm in $ARMS; do
  a40="$(score_of "$arm" 40000)"
  for stop in 100000 200000; do
    s="$(score_of "$arm" "$stop")"
    [ "$a40" = "-" ] || [ "$s" = "-" ] && continue
    awk -v arm="$arm" -v st="$stop" -v a="$a40" -v b="$s" -v w="$BAND" 'BEGIN{
      d=b-a
      if (d < -w) v="YES — it improves with the longer stop"
      else if (d > w) v="no — it gets worse, as every other arm does"
      else v="no verdict — the two land inside the band"
      printf "  %-26s %6dk %.4f against %.4f at 40k  (%+.4f)  %s\n",
             arm, st/1000, b, a, d, v }'
  done
done
echo
echo "The AUC enters here for one purpose only: the guard stops a run at 0.55."
for arm in $ARMS; do
  if [ -f "$(cf412_collapse_file "$arm")" ]; then
    echo "  $arm — STOPPED by the guard: $(cf412_collapse_file "$arm")"
  else
    echo "  $arm — the guard did not stop it."
  fi
done
} | tee "$OUT"
