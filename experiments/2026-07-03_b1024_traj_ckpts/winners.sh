# #369 launch-time manifest.
# Winner arm from #366's τ=0.90 A–I grid: arm C (λ_e=1, λ_h=1) —
# joint minimiser of GM-Rel MASE on both 2L/last (1.1491) and 6L/last
# (1.1254). Every other arm A,B,D,E,F,G,H,I sits above C on both loci.
# Arm I 6L/last still evaluating at launch time; its 2L/last (1.1811),
# 6L/best (1.1649), and 2L/best (1.1813) place it 4th–5th, so 6L/last
# below C's 1.1254 is not credible.
#
# Parent (arm C) best-loss step from
# `bb_..._cpc_lC_emb10_enc10_tau090_losses.csv`: step 533. Snapped to
# nearest multiple of TRAJ_SAVE_EVERY (500) → step 500. Drift 33 steps.

LAMBDA_E=1.0
LAMBDA_H=1.0
TAU=0.90
PARENT_BEST_LOSS_STEP=500

WINNERS_VERIFIED_BY="Orchestrator claude-opus-4-7"
WINNERS_VERIFIED_AT="2026-07-04"
