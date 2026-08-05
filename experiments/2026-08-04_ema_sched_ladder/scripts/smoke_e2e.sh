#!/bin/bash
# #393 — end-to-end smoke of the two new mechanisms, on CPU, in ~5 minutes.
#
# Usage: bash scripts/smoke_e2e.sh [workdir]
#
# Unit tests prove the pieces in isolation. This walks the real launchers
# end to end on a real (tiny) backbone, which is what the pre-merge
# checklist asks for:
#
#   1. train.py trains a few steps with --ema-tau-ramp-steps and saves a
#      checkpoint. The α it logged must follow the ANCHORED curve, not the
#      budget-relative one — the two differ here, so a flag that parsed but
#      never reached the loop fails this.
#   2. a SECOND leg resumes that checkpoint and doubles the budget, the way
#      every stop past 40k does. α must keep climbing across the process
#      boundary, --total-steps must mean an absolute global step, and the
#      optimizer must come back. The ladder is all resumes; a from-scratch
#      leg proves none of the three.
#   3. train_forecasting_head.py trains one head with --encoder-source
#      student and one with --encoder-source teacher off that checkpoint.
#      The two must not train identically, else the teacher never loaded.
#   4. eval_gift_eval_official.py REFUSES the teacher head on the student
#      encoder, and runs it on the teacher encoder. Both directions, so the
#      guard is not just a blanket failure.
#
# Everything is 32-dim / 64-step / a handful of steps. No GPU, no HF
# network (the backbone and the heads run on synthetic data); step 4 reads
# one small GIFT-Eval config from $GIFT_EVAL.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WORK="${1:-$(mktemp -d /tmp/cf393_smoke.XXXXXX)}"
export PYTHONPATH="$REPO"
export GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}"

TRAIN="$REPO/experiments/2026-04-27_freq-embedding/scripts/train.py"
HEAD_TRAIN="$REPO/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
GEVAL="$REPO/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"

mkdir -p "$WORK"
echo "=== #393 smoke — work dir $WORK ==="
fails=0
ok(){   echo "  PASS  $*"; }
bad(){  echo "  FAIL  $*"; fails=$((fails+1)); }

# Tiny backbone geometry, shared by all three scripts.
ARCH=(--t-raw 64 --n-channels 1 --d-model 32 --n-heads 2 --num-layers 1
      --encoder-type gru --rev-norm-kind ewma --rev-norm-span 8)
# The ramp anchor. 6 steps of budget, 30 steps of ramp: α at step 1 is
# 0.9 + 0.1/30 = 0.903333 anchored, and 0.9 + 0.1/6 = 0.916667 if the
# schedule fell back to --total-steps.
STEPS=6
RAMP=30

# One leg of the ladder. Each leg gets its OWN save dir, exactly as
# run_leg.sh does: train.py branches --run-name to `<name>_r2` when the
# save dir it is handed already holds `<name>_*.pth`, and a renamed
# checkpoint is not where the next leg or the eval looks for it.
leg(){  # <save dir> <total steps> [resume checkpoint]
  local dir="$1" total="$2" resume="${3:-}"
  local args=(--device cpu --total-steps "$total" --save-every "$STEPS"
              --batch-size 2 --lr 1e-3 --weight-decay 0.1 --seed 42
              --log-every 1 --tau 0.10 --save-dir "$dir" --run-name smoke393
              --hf-repo none --hf-path none --mix-ratio 1.0
              --synth-kind periodic --num-encoder-layers 1 "${ARCH[@]}"
              --loss-shape cosine_similarity_batch_split_pred_rep
              --ema-embedding --ema-encoder
              --ema-tau 0.9 --ema-tau-end 1.0 --ema-tau-ramp-steps "$RAMP")
  [ -n "$resume" ] && args+=(--resume "$resume")
  mkdir -p "$dir"
  python3 -u "$TRAIN" "${args[@]}"
}

# ---- 1. backbone with --ema-tau-ramp-steps ---------------------------------
echo "[1/4] backbone: $STEPS steps, --ema-tau-ramp-steps $RAMP"
BB_DIR="$WORK/bb/leg_1"
leg "$BB_DIR" "$STEPS" >"$WORK/backbone.log" 2>&1
rc=$?
BB="$BB_DIR/smoke393_0k.pth"
[ $rc -eq 0 ] && [ -f "$BB" ] \
  && ok "checkpoint saved ($(du -h "$BB" | cut -f1))" \
  || { bad "backbone rc=$rc; tail:"; tail -15 "$WORK/backbone.log"; exit 1; }

python3 - "$BB_DIR/smoke393_losses.csv" "$RAMP" "$STEPS" <<'PY'
import csv, sys
path, ramp, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = list(csv.DictReader(open(path)))
tau = {int(r["step"]): float(r["ema_tau"]) for r in rows if r.get("ema_tau")}
anchored = {s: 0.9 + 0.1 * min(s / ramp, 1.0) for s in tau}
budget = {s: 0.9 + 0.1 * min(s / steps, 1.0) for s in tau}
bad = [(s, tau[s], anchored[s]) for s in tau if abs(tau[s] - anchored[s]) > 1e-6]
if bad:
    print(f"  FAIL  logged α is not the anchored curve: {bad[:3]}")
    sys.exit(1)
if all(abs(anchored[s] - budget[s]) < 1e-9 for s in tau):
    print("  FAIL  the two curves coincide; this smoke proves nothing")
    sys.exit(1)
lo, hi = min(tau), max(tau)
print(f"  PASS  α follows the anchor: step {lo} -> {tau[lo]:.6f}, "
      f"step {hi} -> {tau[hi]:.6f} (budget-relative would be "
      f"{budget[hi]:.6f})")
PY
[ $? -eq 0 ] || fails=$((fails+1))

# ---- 2. the second leg: resume to twice the budget --------------------------
RESUMED=$(( STEPS * 2 ))
echo "[2/4] resumed leg: $STEPS -> $RESUMED steps, same ramp"
BB2_DIR="$WORK/bb/leg_2"
leg "$BB2_DIR" "$RESUMED" "$BB" >"$WORK/backbone_resume.log" 2>&1
rc=$?
[ $rc -eq 0 ] \
  && ok "resumed leg exited clean" \
  || { bad "resumed leg rc=$rc; tail:"; tail -20 "$WORK/backbone_resume.log"; }

grep -q "Restored optimizer from" "$WORK/backbone_resume.log" \
  && ok "optimizer state restored" \
  || bad "no optimizer-restore banner — --resume did not bring AdamW back"

python3 - "$BB_DIR/smoke393_losses.csv" "$BB2_DIR/smoke393_losses.csv" \
         "$RAMP" "$STEPS" "$RESUMED" <<'PY'
import csv, sys
first, second, ramp, steps, resumed = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]))
def tau_of(path):
    with open(path) as fh:
        return {int(r["step"]): float(r["ema_tau"])
                for r in csv.DictReader(fh) if r.get("ema_tau")}
a, b = tau_of(first), tau_of(second)
want = sorted(range(steps + 1, resumed + 1))
if sorted(b) != want:
    print(f"  FAIL  resumed leg logged steps {sorted(b)}, expected {want}. "
          f"The step counter restarted per process.")
    sys.exit(1)
anchored = lambda s: 0.9 + 0.1 * min(s / ramp, 1.0)
off = [(s, b[s], anchored(s)) for s in b if abs(b[s] - anchored(s)) > 1e-9]
if off:
    print(f"  FAIL  resumed α is not the anchored curve at the GLOBAL "
          f"step: {off[:3]}")
    sys.exit(1)
seam = b[steps + 1] - a[steps]
if abs(seam - 0.1 / ramp) > 1e-9:
    print(f"  FAIL  α jumped {seam:.8f} across the seam; one anchored step "
          f"is {0.1 / ramp:.8f}")
    sys.exit(1)
print(f"  PASS  steps {min(b)}..{max(b)} are global, α continuous across "
      f"the seam: {a[steps]:.6f} -> {b[steps + 1]:.6f}")
PY
[ $? -eq 0 ] || fails=$((fails+1))

# Stages 3 and 4 run off THIS checkpoint: every stop past 40k is a resumed
# leg's output, so the heads and the eval have to work on one.
BB2="$BB2_DIR/smoke393_0k.pth"
[ -f "$BB2" ] \
  && ok "the stop's checkpoint kept its name under a fresh save dir" \
  || bad "no smoke393_0k.pth in $BB2_DIR — safe_run_name renamed the leg"

# ---- 3. one head per encoder ------------------------------------------------
HEAD_ARCH=(--head-arch transformer --head-num-layers 2 --head-nhead 8
           --head-ffn-mult 4.0 --head-causal true --head-dropout 0.1
           --head-train-input e_then_f)
for SRC in student teacher; do
  echo "[3/4] head: --encoder-source $SRC"
  D="$WORK/head_$SRC"; mkdir -p "$D"
  python3 -u "$HEAD_TRAIN" \
    --backbone-path "$BB2" --encoder-source "$SRC" \
    --device cpu --quantile-head --grad-clip 1.0 --forecast-len 16 \
    --batch-size 4 --lr 1e-3 --total-steps 4 --save-every 100 --log-every 1 \
    --save-dir "$D" --run-name "h_$SRC" --seed 20260722 \
    --hf-repo none --hf-path none --mix-ratio 1.0 \
    "${ARCH[@]}" "${HEAD_ARCH[@]}" >"$WORK/head_$SRC.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { bad "head[$SRC] rc=$rc; tail:"; tail -15 "$WORK/head_$SRC.log"; continue; }
  grep -q "encoder=$SRC" "$WORK/head_$SRC.log" \
    && ok "trained, log says encoder=$SRC" || bad "head[$SRC] log never says encoder=$SRC"
  marker="$D/h_${SRC}_final_encoder_source.txt"
  [ "$(cat "$marker" 2>/dev/null)" = "$SRC" ] \
    && ok "marker next to the checkpoint reads '$SRC'" \
    || bad "marker $marker missing or wrong"
done

python3 - "$WORK" <<'PY'
import csv, sys
work = sys.argv[1]
def losses(src):
    with open(f"{work}/head_{src}/h_{src}_losses.csv") as fh:
        return [float(r["loss"]) for r in csv.DictReader(fh)]
s, t = losses("student"), losses("teacher")
if s == t:
    print(f"  FAIL  the two heads trained identically: {s}")
    sys.exit(1)
print(f"  PASS  the heads differ — student {s[0]:.4f}..{s[-1]:.4f}, "
      f"teacher {t[0]:.4f}..{t[-1]:.4f}")
PY
[ $? -eq 0 ] || fails=$((fails+1))

# ---- 3. the eval guards the pairing -----------------------------------------
TEACHER_HEAD="$WORK/head_teacher/h_teacher_final.pth"
EVAL_ARCH=(--t-raw 64 --n-channels 1 --d-model 32 --n-heads 2 --num-layers 1
           --encoder-type gru --rev-norm-kind ewma --rev-norm-span 8
           --head-nhead 8 --head-causal true --forecast-len 16 --strategy B4)

echo "[4/4] eval: teacher head through the STUDENT encoder (must refuse)"
python3 -u "$GEVAL" --backbone-path "$BB2" --head-path "$TEACHER_HEAD" \
  --encoder-source student --device cpu --output-dir "$WORK/ev_mismatch" \
  --config-filter 'us_births/D' --test-only 1 \
  "${EVAL_ARCH[@]}" >"$WORK/eval_mismatch.log" 2>&1
rc=$?
if [ $rc -ne 0 ] && grep -q "trained on the teacher encoder" "$WORK/eval_mismatch.log"; then
  ok "refused (rc=$rc): $(grep -o 'was trained on the teacher encoder.*' "$WORK/eval_mismatch.log" | head -1 | cut -c1-70)..."
else
  bad "mismatch NOT refused (rc=$rc); tail:"; tail -15 "$WORK/eval_mismatch.log"
fi

echo "[4/4] eval: teacher head through the TEACHER encoder (must run)"
python3 -u "$GEVAL" --backbone-path "$BB2" --head-path "$TEACHER_HEAD" \
  --encoder-source teacher --device cpu --output-dir "$WORK/ev_match" \
  --config-filter 'us_births/D' --test-only 1 \
  "${EVAL_ARCH[@]}" >"$WORK/eval_match.log" 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q "\[eval\] encoder=teacher" "$WORK/eval_match.log"; then
  ok "ran on the teacher: $(grep -h 'Aggregate' "$WORK/ev_match"/*.txt 2>/dev/null | head -1)"
else
  bad "matched pair failed (rc=$rc); tail:"; tail -20 "$WORK/eval_match.log"
fi

echo
if [ $fails -eq 0 ]; then
  echo "=== SMOKE PASSED — logs under $WORK ==="
else
  echo "=== SMOKE FAILED: $fails check(s) — logs under $WORK ==="
fi
exit $fails
