# BYOL alignment term on top of the contrastive loss (#309)

**Question.** Does adding a non-saturating BYOL/SimSiam alignment term
`L_align = 2 − 2·cos(f_t, sg(h_{t+1}))` (weight λ = 1) on top of the
`#303` contrastive loss improve held-out GIFT-Eval **GM-MASE**?

## Rationale (what we're testing, not a claim)

The normalized-InfoNCE positive gradient is `−(1−p₊)/τ`: it **fades** as
the positive separates from the negatives, so once the contrastive task is
won, more steps stop refining the positive. In a single-epoch regime that
is costly — a saturating objective extracts nothing from each new batch.
`L_align = ‖f̂ − ĥ‖² ∈ [0, 4]` (min 0 at cos = 1) has a **constant**
gradient (2) until `cos = 1`, decoupled from the negatives — it keeps a
live alignment pull after the contrastive term converges.

Caveat carried from `#296`/`#303`: every contrastive-side proxy (loss,
`loss_tau_ref`, AUC, dim-usage) has **decoupled** from GM-MASE. So the
above is a hypothesis; **only GM-MASE adjudicates.** The "positive
saturates" premise itself is *unverified* (the AUC evidence was retracted)
— this run is partly to instrument it.

## What's new in the code (`src/loss.py`, #309)

- `--align-loss-weight λ` → `train_configuration['align_loss_weight']`
  (default `0.0` = off, so every prior run is byte-for-byte unchanged).
  Adds `λ·(2 − 2·cos(f_t, sg(h_{t+1}))).mean()`; stop-grad on the encoder
  target (gradient flows only through the forecaster). λ in **front** of
  L_align; this run uses **λ = 1.0**.
- `--subtract-contrastive-floor` → re-bases the loss by the constant
  `infonce_floor(τ, N) = log(1 + N·e^(−1/τ))` so the logged curve reads
  ~0 at the uniformity floor. **Gradient-neutral** (a constant; argmin /
  EMA / NaN-checks unchanged). Needs `--pos-in-denominator`. `N` is the
  negative count, computed from the variant and B/T/C — so the floor is a
  function of τ **and** N, not τ alone, and it is the *theoretical* floor
  (assumes cos⁻≈0, cos⁺=1), so the re-based curve settles slightly above 0.
  With both on, the total `(L_c − floor) + λ·L_align` has theoretical
  minimum 0 (L_align is already min-0).

## Protocol

Single-variable add of `L_align` (λ=1) on the **#303 arm-B** backbone
recipe (`full_hh_negs`, the current best contrastive arm, −5.6% vs A),
`--pos-in-denominator`, τ=0.10, 1L forecaster (the stable fp16 recipe).
Then the standard 30k 2L-causal q-head + official GIFT-Eval (triage 11,
full 97). **Baseline = the same recipe with λ=0** (i.e. arm B as-is).

## Diagnostics to watch (test the premise, don't assume it)

- **GM-MASE (full 97)** — the only verdict.
- **Mean positive `cos(f_t, h_{t+1})`** and the persistence-gap
  `[cos(f,h_{t+1}) − cos(h,h_{t+1})] / [1 − cos(h,h_{t+1})]` — is the
  forecaster learning dynamics, and does L_align move it?
- **`u_batch` / `u_temporal`** — does the encoder-spreading that arm B
  rewards **decay** once L_align dominates the late gradient? (the risk;
  if so, that argues for an explicit uniformity term instead.)
- **`loss_tau_ref`** — kept a *pure* contrastive reference (the diagnostic
  call forces `align_loss_weight=0`, `subtract_contrastive_floor=False`).

Per-step grad-norm logging (‖∇L_c‖ vs ‖∇L_align‖) is the cleanest handoff
diagnostic but is **deferred** (a trainer/CSV change) — follow-up.

## Decision rule

Judge on **GM-MASE** only. If λ=1 is promising or ambiguous, sweep
λ ∈ {0.25, 0.5, 1, 2}. Single seed (matched to the baseline) for the first
pass; a per-arm CI needs multiple seeds.
