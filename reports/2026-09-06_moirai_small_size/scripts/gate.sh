#!/bin/bash
# #412 — the gate of PR #413, applied to the 40,000-step scores.
#
# It reads `results/scores.csv` and `results/auc_verdicts.tsv` and prints the
# arms that climb to 200,000 steps. It changes nothing on disk, so a reader
# can run it, and the report can quote its output.
#
# ---- The rule, in the words of the plan --------------------------------------
#
#   * THE BAND is 0.0568 GM-Relative MASE, and this card measured it at 11.4M
#     parameters over the two `k3_r100_09` seeds. It replaces #409's 0.0471,
#     which is a 1.1M number. A wider measured spread replaces it in turn.
#   * THE AUC COMES FIRST. An arm whose rolling AUC median ended under 0.55
#     lost the contrastive task, and it does not climb.
#   * Climb `k3_r100_09` ALWAYS. Its reference, 1.0651, is a 200,000-step
#     number.
#   * Climb every other arm within the band of the 40,000-step leader.
#   * `k32_r200_08` gets an extra 0.0291, the mid-ramp deficit its 1.1M twin
#     showed at this stop.
#   * THE PAIR. If `k32_r100_09_dec` minus `k32_r100_09` lands inside the
#     band, repeat `k32_r100_09` at a second seed instead of climbing both.
#
# Usage:  bash scripts/gate.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

STOP="${CF412_GATE_STOP:-40000}"
BAND_FLOOR="${CF412_BAND_FLOOR:-0.0568}"
R200_ALLOWANCE="${CF412_R200_ALLOWANCE:-0.0291}"
SCORES="$CF412_RESULTS/scores.csv"
VERDICTS="$CF412_RESULTS/auc_verdicts.tsv"
[ -s "$SCORES" ] || { echo "ABORT: no scores at $SCORES" >&2; exit 2; }

python3 - "$SCORES" "$VERDICTS" "$STOP" "$BAND_FLOOR" "$R200_ALLOWANCE" <<'PY'
import csv, os, re, sys

scores_path, verdicts_path, stop, band_floor, r200_allow = sys.argv[1:6]
stop = int(stop); band_floor = float(band_floor); r200_allow = float(r200_allow)

score = {}
with open(scores_path) as fh:
    for row in csv.DictReader(fh):
        if int(row['stop']) == stop:
            score[row['arm']] = float(row['score'])

# A run's verdict rides on its losses CSV name, which carries the arm.
lost = set()
if os.path.exists(verdicts_path):
    with open(verdicts_path) as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            if row.get('verdict') == 'lost':
                lost.add(row['run'])

def arm_lost(arm):
    # The arm token sits between `_cf412_` and `_losses.csv` in the run name.
    # An open prefix match reads `k32_r100_09_dec` as `k32_r100_09` and stops
    # an arm that held the task, so the test is anchored at both ends. A
    # re-fired leg writes `_cf412_<arm>_rN_losses.csv`, which the pattern
    # takes too. See the `_rN` branch of `auc_guard.sh`.
    pat = re.compile(rf'_cf412_{re.escape(arm)}(_r\d+)?_losses\.csv$')
    return any(pat.search(run) for run in lost) or os.path.exists(
        os.path.join(os.path.dirname(scores_path), f'collapsed_{arm}.txt'))

print(f'--- the 40,000-step scores ({len(score)} arm(s)) ---')
for arm, v in sorted(score.items(), key=lambda kv: kv[1]):
    print(f'  {arm:<16} {v:.4f}{"   LOST the contrastive task" if arm_lost(arm) else ""}')

live = {a: v for a, v in score.items() if not arm_lost(a)}
if not live:
    print('\nNo arm held the contrastive task. Nothing climbs.'); sys.exit(0)

# The band.
seeds = [score[a] for a in ('k3_r100_09', 'k3_r100_09b') if a in score]
spread = abs(seeds[0] - seeds[1]) if len(seeds) == 2 else 0.0
band = max(band_floor, spread)
print(f'\n--- the band ---')
print(f'  this card, 11.4M, two seeds of k3_r100_09    {band_floor:.4f}')
print(f'  the pair on disk now                         {spread:.4f}' if len(seeds) == 2
      else '  the pair on disk now                         not measured')
print(f'  THE BAND                                     {band:.4f}')

leader_arm = min(live, key=live.get)
leader = live[leader_arm]
print(f'\n--- the leader ---\n  {leader_arm} at {leader:.4f}')

climb, reasons = [], {}
for arm in sorted(live, key=live.get):
    allowance = r200_allow if arm == 'k32_r200_08' else 0.0
    ceiling = leader + band + allowance
    if arm == 'k3_r100_09':
        climb.append(arm); reasons[arm] = 'climbs ALWAYS — its 1.0651 reference is a 200,000-step number'
    elif live[arm] <= ceiling:
        extra = f' (+{allowance:.4f} mid-ramp allowance)' if allowance else ''
        climb.append(arm)
        reasons[arm] = f'{live[arm]:.4f} <= {ceiling:.4f}{extra}'
    else:
        reasons[arm] = f'{live[arm]:.4f} > {ceiling:.4f} — outside the band'
for arm in sorted(score):
    if arm_lost(arm):
        reasons[arm] = 'lost the contrastive task — a higher stop trains the same collapse'

# The pair.
pair = None
if 'k32_r100_09' in score and 'k32_r100_09_dec' in score:
    gap = score['k32_r100_09_dec'] - score['k32_r100_09']
    inside = abs(gap) <= band
    print(f'\n--- the decay pair ---')
    print(f'  k32_r100_09_dec minus k32_r100_09 = {gap:+.4f}, band {band:.4f}')
    if inside and 'k32_r100_09' in climb and 'k32_r100_09_dec' in climb:
        climb = [a for a in climb if a not in ('k32_r100_09', 'k32_r100_09_dec')]
        for a in ('k32_r100_09', 'k32_r100_09_dec'):
            reasons[a] = (f'inside the band of its twin ({gap:+.4f}) — the plan '
                          'repeats the seed instead of climbing the pair')
        pair = 'repeat k32_r100_09 at a second seed, 40,000 steps'
        print('  INSIDE the band, and both arms would climb. The plan repeats')
        print('  k32_r100_09 at a second seed instead of climbing both.')
    elif inside:
        print('  INSIDE the band. The decay changes nothing this card can measure.')
    else:
        print('  OUTSIDE the band. The pair separates.')

print('\n--- the verdict ---')
for arm in sorted(score, key=lambda a: score[a]):
    mark = 'CLIMB ' if arm in climb else 'stops '
    print(f'  {mark} {arm:<16} {reasons.get(arm, "")}')
if pair:
    print(f'  EXTRA  {pair}')
print('\nCommand:')
if climb:
    print(f'  STOPS=200000 ARMS="{" ".join(climb)}" bash run.sh phase1')
else:
    print('  no arm climbs')
PY
