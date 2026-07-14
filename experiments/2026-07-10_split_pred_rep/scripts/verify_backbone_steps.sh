#!/bin/bash
# Verify each arm's FINAL.pth by (a) md5 against best_loss.pth / final.pth /
# intermediate saves, and (b) the last `Saved …_best_loss.pth` event step
# in the arm's run log. Emits a text log used by the report to justify
# each arm's backbone-step table entry. Idempotent; safe to re-run.
set -euo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$EXP/results/backbone_step_verification.log"
: >"$OUT"

log(){ echo "$*" | tee -a "$OUT"; }

verify(){
  local arm="$1" runs_dir="$2" bb_stem="$3" run_log="$4"
  log "=== $arm ==="
  cd "$runs_dir"
  local final="${bb_stem}_FINAL.pth"
  [ -f "$final" ] || { log "  MISSING $final"; cd - >/dev/null; return; }
  local final_md=$(md5sum "$final" | awk '{print $1}')
  log "  FINAL.pth md5: $final_md"
  # candidate twins
  for cand in best_loss.pth final.pth 2k.pth 5k.pth 7k.pth 10k.pth 12k.pth; do
    local f="${bb_stem}_${cand}"
    [ -f "$f" ] || continue
    local m=$(md5sum "$f" | awk '{print $1}')
    if [ "$m" = "$final_md" ]; then log "  == $f (md5 match)"; fi
  done
  cd - >/dev/null

  if [ -f "$run_log" ]; then
    local n_saves=$(grep -c "Saved.*${bb_stem}_best_loss\.pth" "$run_log" 2>/dev/null)
    [ -n "$n_saves" ] || n_saves=0
    local last_step=$(grep -B4 "Saved.*${bb_stem}_best_loss\.pth" "$run_log" | \
                       grep -oE "^\[ *[0-9]+\]" | tr -d '[] ' | sort -n | tail -1)
    log "  run-log best_loss saves: n=$n_saves  last-save-step=${last_step:-none}"
    local closing=$(grep "^Done" "$run_log" | tail -1)
    log "  closing: $closing"
  else
    log "  MISSING run log $run_log"
  fi
  log ""
}

verify "arm 1 (split)" \
  "$EXP/runs" \
  "bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090" \
  "$EXP/results/run_bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090.log"

verify "arm 3 (split + MoCo)" \
  "$EXP/runs" \
  "bb_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090" \
  "$EXP/results/run_bb_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090.log"

verify "arm 4 (pooled + MoCo)" \
  "$EXP/runs_arm4" \
  "bb_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090" \
  "$EXP/results_arm4/run_bb_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090.log"

verify "arm 5 (L_align + L_rep)" \
  "$EXP/runs_arm5" \
  "bb_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090" \
  "$EXP/results_arm5/run_bb_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090.log"

verify "arm 6 (L_align_moco + L_rep)" \
  "$EXP/runs_arm6" \
  "bb_lalignmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6_tau090" \
  "$EXP/results_arm6/run_bb_lalignmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6_tau090.log"

verify "arm bimoco (split + moco on L_pred and L_rep)" \
  "$EXP/runs_bimoco" \
  "bb_split_pred_rep_bimoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090" \
  "$EXP/results_bimoco/run_bb_split_pred_rep_bimoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090.log"

# Additionally, for arm 1: check torch.equal across all 193 tensors between
# FINAL.pth, final.pth and 12k.pth — the state-dict identity used in the
# report to say arm 1's backbone did not update in the final 500 steps.
python3 - "$EXP/runs" <<'PY' | tee -a "$OUT"
import sys, torch
d = sys.argv[1]
stem = "bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"
def sm(x): return x.get("model", x.get("state_dict", x)) if isinstance(x, dict) else x
def load(name): return sm(torch.load(f"{d}/{stem}_{name}.pth", map_location="cpu", weights_only=False))
final = load("FINAL"); f = load("final"); k = load("12k"); bl = load("best_loss")
def eq(a,b): return all(torch.equal(a[k_], b[k_]) for k_ in a)
print(f"arm 1 torch.equal FINAL == final    : {eq(final,f)}")
print(f"arm 1 torch.equal FINAL == 12k      : {eq(final,k)}")
print(f"arm 1 torch.equal FINAL == best_loss: {eq(final,bl)}")
print(f"arm 1 num keys checked              : {len(final)}")
PY

log "== done: $OUT =="
