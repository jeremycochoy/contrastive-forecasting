# #341 stop-grad × capacity — run STATUS (snapshot 2026-06-12 ~18:16 BST)

Machine paused for maintenance mid-run. **To resume:**
```
bash ~/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity/BOOTSTRAP_RESUME.sh
```
(re-creates the /tmp/cf-341 worktree from git if /tmp was wiped, then relaunches
chains + watchdog + per-arm phase 2; everything idempotent — done work skipped).

## Arms (all backbones DONE, 12.5k steps, seed 20260520)
| # | arm | encoder | forecaster | stop-grad | best-loss step |
|---|---|---|---|---|--:|
| 1 | base+triplet (#336) | 6L | 128-bn | no | — |
| 2 | #339 winner | 3L | full | yes | ~6.6k |
| 3 | **new** nobn_enc6 | 6L | full | yes | **1300** |
| 4 | **new** bn_enc6 | 6L | 128-bn | yes | **1000** |

Reference GM-Relative MASE (lower better): arm1 2L 1.186/1.187, 6L 1.185/1.190 (best/last);
arm2 2L 1.177/1.180, 6L **1.159**/1.163.

## Done on disk (NOT recomputed on resume)
- Both new backbones (FINAL=best-loss, final=last).
- All 8 chain heads (best + 10k-re-adapt-last, both HL, both arms).
- arm4 **2L lastfresh** head (fresh 30k on the last backbone).
- Eval cells merged: **arm4 bn_enc6 2L re-adapt-last = 2.2652** (the only merged cell).

## Key finding so far (verified, with a caveat)
arm4 (bn_enc6 + stop-grad), 2L, **re-adapt-last = 2.27** — far worse than arm1's 1.187
without stop-grad. The eval pipeline was **cross-checked correct** (reproduced arm1's
#336 numbers to 4 decimals), so 2.27 is a real measurement. **Caveat:** arm4's best-loss
is step 1000, so the 10k-re-adapt last head fine-tunes from a head trained on a barely-
trained backbone — a likely confound. Hence the **fresh-last** heads (30k trained directly
on the last backbone) — those cells are the clean last-checkpoint read and are PENDING.

## Pending (phase 2 finishes on resume)
- Fresh-last heads: arm4 6L, arm3 2L+6L (arm4 2L done). [in-progress at pause]
- Eval cells: lastfresh (PRIMARY) ×4, best ×4, re-adapt-last ×3 — per-arm, both GPUs.
- Then: analyze_sgcap.py (paired bootstrap, all pairwise contrasts) → plots → report
  stopgrad_capacity.md → PR into experiments. Verdict question: does stop-grad flip the
  capacity sign (#336: enc6 reliably worse than enc3 WITHOUT stop-grad)?

## Verdict watch
- arm4 fresh-last vs 2.27: if it normalises (~1.2), the 2.27 was the re-adapt confound;
  if it stays bad, stop-grad genuinely wrecks the bottleneck recipe.
- arm3 (full-width, unaffected by the bottleneck question) is the clean enc3→enc6 test.
