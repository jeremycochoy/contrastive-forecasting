# Session Handoff — late April 2026

This handoff covers a multi-day experimental sequence on top of the v3b
backbone, branching from `experiments` into `feat/periodic-synth-mix`.
Two open PRs, eight ablations run, three reports landed.

## TL;DR for the next agent

1. **Start by retraining the RevIN model**. The backbone+head from
   ablation #28 were lost to a partial-transfer SSH drop. We have the
   eval CSV but no weights, so the RevIN synth-grid plot the user
   asked for can't be made yet. Reproducing the run is ~1.3h backbone
   + 1.3h head + 5h eval ≈ ~$1.50 on a 4090. The PR #47 sync fixes
   should make this safer.
2. **Then the next planned experiment is synthetic-only**: train a
   backbone on PURE periodic synth (no base-bundles mix) for both 30k
   and 60k steps, then a reconstruction head also on synth-only. The
   open question is whether reconstruction works on the *training*
   distribution at all — the synth-grid plot
   ([`experiments/freq-embedding/plots/synth_qhead_grid.png`](experiments/freq-embedding/plots/synth_qhead_grid.png))
   currently shows the head fails to reproduce SN even on its own data.
3. **Then a span sweep**: RevEWMNorm with span ∈ {32, 64, 128, 256}
   (start with these four, adjust based on what's monotonic). 32 is
   the current default, settled in [revnorm-span-search](experiments/revnorm-span-search/report.md)
   on ARMA-only data — we now have reason to think a longer span might
   help once periodic content dominates the input.

After those three, the user said I can go idle.

## What we ran this session

| # | What | Result | Where |
|---|---|---|---|
| #11 | periodic-synth-mix CONTROL + MIX 30k | aggregate tied; periodic subset −3.4%; tiny generalisation tax on non-periodic | `experiments/periodic-synth-mix/REPORT.md` |
| #19 | MIX 30k → 90k extension | aggregate +0.5%; periodic subset another −1.9%; non-periodic +2% worse (synth bias grows with training) | same REPORT.md addendum |
| #23 | freq embedding (with/without mixup) | embedding alone = wash; with mixup = small but consistent win on aggregate; 7-curve plots committed | `experiments/freq-embedding/DESIGN.md`, `REPORT.md` |
| #26 | head 30k → 90k on fe+mu | training-MSE down 18%, OOD WQL +6%: classic overfit. Head was *not* undertrained at 30k | `experiments/freq-embedding/REPORT.md` |
| #24 | quantile (pinball) head | **+15 points of WQL skill** at zero MASE cost. The cleanest single win of the sequence | `experiments/freq-embedding/REPORT.md` |
| #28 | RevIN vs RevEWMNorm | preserves periodic amplitude (huge ett1/15T win, big solar/10T win) but loses to EWMA on trend (m4_*, covid). Aggregate slight loss; periodic subset substantial WQL gain | `experiments/freq-embedding/REPORT.md` |
| #25/#27 | qualitative plots (multi-model + focused) | committed | `experiments/freq-embedding/plots/` |
| PR #47 | sync-loop atomic + rotation + safe_pull.sh | merged separately to fix the corruption pattern | branch `fix/sync-rotation-atomicity` |

## Final results table

Skill scores vs Seasonal Naive on the 43 univariate configs the local SN
baseline covers (multivariate datasets failed in the SN script — known
limitation, see freq-embedding REPORT.md "caveats"). Higher is better.

| Arm | MASE skill | WQL skill |
|---|---:|---:|
| v2 (500k, MSE) | −14.5% | −16.9% |
| v3b (120k, MSE) | −17.6% | −19.0% |
| mix90 (90k, MSE) | −22.8% | −25.7% |
| fe+mu (30k, MSE) | −18.2% | −20.5% |
| fe+mu+qh (qhead) | −18.0% | **−5.1%** |
| **RevIN+qh** (qhead) | **−13.2%** | **+1.1%** |
| Seasonal Naive | 0.0% | 0.0% |
| Moirai-2.0-small (11.4M, ref) | +27% | +48% |

Same-size SOTA reference is on the full 97-config GIFT-Eval; ours is on
43. Direct comparison is therefore approximate, but ~30 points below
SOTA on each metric is the order of magnitude. The gap is dominated by
training-corpus scale (LOTSA full vs our small subset), not architecture.

## What we KNOW (facts only — user feedback emphasis)

The user repeatedly pushed for facts vs aspiration. Things established:

1. **Adding 50% clean-periodic synth at matched 30k compute** delivers a
   modest, uneven improvement on the 6 periodic failure configs from
   v3b's HANDOFF — wins on 4/6, biggest at ett2/W/short (-15.5%).
   Aggregate is essentially tied.
2. **Extending the same setup to 90k** continues to improve the
   periodic subset slightly but **degrades** non-periodic and stationary
   subsets — the bias grows. Net aggregate is barely better than 30k.
3. **A pinball-loss head universally improves WQL** on every subset by
   8–13% (raw) without hurting MASE. This is structural — the MSE head
   collapses to the conditional mean and underestimates peak amplitudes.
4. **RevIN preserves periodic amplitude** much better than RevEWMNorm
   (span=32). On cleanly periodic configs (ett1/15T/short, solar/10T)
   RevIN wins big. On trend-heavy configs (m4_*, covid_deaths) it
   regresses just as big. Net trade-off, not a uniform improvement.
5. **The head was NOT undertrained at 30k.** Extending to 90k overfits.
   This was specifically tested at the user's request.
6. **The synth grid plot reveals an upstream problem.** Even on the
   model's *own training distribution* (clean periodic synth), the
   qhead median doesn't match seasonal-naive (which has the known P
   and is therefore essentially optimal). The bottleneck is in the
   backbone's latent representation, not the head — likely a phase
   information loss at the patch boundary (W=16 collapses sub-patch
   position).
7. **CONTROL for matched compute matters.** Without the from-scratch
   30k pure-base-bundles arm, we couldn't have separated "MIX
   improvement" from "more training improvement" in #19. The user
   flagged this when designing the experiment.
8. **The freq embedding plumbing is incomplete on the HF side.** All
   real-data rows are tagged class 0 = unknown. The win we measured is
   from synth-half regularization + mixup, not from frequency-aware
   eval-time forecasting. A real implementation would thread the row's
   actual frequency through.
9. **vast.ai SSH is unreliable enough that anything important must be
   incrementally backed up.** PR #47 documents both the failure (#28's
   RevIN checkpoints were lost) and the fix (atomic-write + one-deep
   rotation in sync_loop, never raw scp via safe_pull.sh).

## User feedback worth carrying forward

- **"Don't push in the direction of what you can conclude. Focus on
  facts and what we know."** Several times during the session the user
  redirected away from speculation toward measured results. The
  freq-embedding REPORT.md and this handoff try to honour that.
- **"Always test the sync loop first / verify all expected files
  exist."** Already in CLAUDE.md after PR #45. The RevIN loss was
  exactly this rule being skipped (no sync_loop running, manual scp
  at the end).
- **"The MASE numbers don't tell the whole story — let's see WQL."**
  The quantile-head ablation was queued specifically because the user
  pointed out that MSE collapsing to the conditional mean was the
  visible failure in the periodic-synth-mix prediction plots.
- **"We probably want only ground truth, SN, and the new
  architectures we made — no need for all the others, it will be too
  cluttered."** That's why the focused qhead plots
  ([`experiments/freq-embedding/plots/predictions_qhead/`](experiments/freq-embedding/plots/predictions_qhead/))
  show only 4 curves + uncertainty band, not 6.
- **"For the synth grid: I can see we are quite bad at reconstructing
  such patterns. This will help us plan our strategy moving forward."**
  This is the prompt for the synth-only experiment queued below.

## Operational template — how the last experiment looked

The pattern that worked end-to-end (modulo the RevIN sync gap):

```bash
# 1. Code changes locally, smoke test
python3.11 -m pytest tests/test_<new_module>.py -q
python3.11 -c "<minimal CPU smoke of the new code path>"

# 2. Provision vast.ai 4090 (avoid vastrun-kit's ssh-key bug)
vastai search offers "gpu_name=RTX_4090 num_gpus=1 gpu_ram>=20 \
  inet_up>=1000 verified=true rentable=true rented=false \
  reliability>=0.99 cuda_vers>=12.8 dph<0.4" -o "dph_total+" --raw
vastai create instance <id> \
  --image "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime" \
  --disk 50 --label <something>

# 3. Wait for SSH (Monitor with until-loop)

# 4. Upload code + checkpoints
tar czf /tmp/code.tgz --exclude='*.pth' --exclude='__pycache__' \
  src/ tests/ experiments/<...>/scripts/ experiments/hf_token.txt
scp -P <port> /tmp/code.tgz <preexisting checkpoints> root@<host>:/workspace/app/
ssh root@<host> "cd /workspace/app && tar xzf code.tgz && \
  mkdir -p checkpoints && mv <pths> checkpoints/"

# 5. Write run_*.sh on the box and launch via nohup (with HF_TOKEN export!)
#    Tee everything to /workspace/app/run_all.log

# 6. START THE SYNC LOOP IMMEDIATELY (before walking away)
bash experiments/periodic-synth-mix/scripts/sync_loop.sh \
  <host> <port> sync_<runname>/ /workspace/app > sync_<runname>/sync.log 2>&1 &

# 7. Verify the FIRST tick succeeded — manually list local files —
#    before considering the run unattended (CLAUDE.md rule).

# 8. Set up a Monitor on the run_all.log via SSH polling. Poll patterns:
#    "STAGE|NaN/Inf|Traceback|FAILED|Killed|OOM|ALL DONE|Done in|Saved.*_[0-9]+k.pth"

# 9. Schedule periodic ScheduleWakeup (1h is a good default).

# 10. When it's done: pull final results, destroy the instance,
#     stop the sync loop and monitor, commit + push.
```

Code references the next agent will need:

- `src/synthetic_periodic.py` — clean periodic synth, vectorised, ~1 ms/batch.
- `src/freq_embedding.py` — small (3-dim) embedding + mixup helper.
- `src/forecasting_head.py` — `ForecastingHead`, `QuantileForecastingHead`,
  `quantile_loss`, `forecast_with_strategy(...)`, `extract_*_latents`.
  The B4 strategy is what we've been using throughout.
- `src/dataloader.py` — `MixedPeriodicLoader` (HF + on-the-fly synth),
  `HFStreamingLoader` with the *fast skip* shipped in commit `bf3687c`
  (parquet-shard-aware, avoids the multi-million-row sequential .skip).
- `src/models.py` `ConfigurableModel` — has `freq_emb_dim` and
  `rev_norm_kind` ('ewma' | 'revin' | 'none') as kwargs.
- `src/norm.py` — `RevEWMNorm` (EWMA, span-parametrised),
  `RevIN` (single per-instance z-score).
- `experiments/freq-embedding/scripts/train.py` — the canonical
  contrastive-train script with all of the above flags exposed.
- `experiments/gift-eval/scripts/train_forecasting_head.py` —
  head training; auto-detects freq_emb_dim from the backbone checkpoint;
  `--quantile-head` flag for pinball loss; `--rev-norm-kind` MUST match
  the backbone's training-time choice.
- `experiments/gift-eval/scripts/eval_gift_eval_official.py` — GIFT-Eval
  predictor wrapper; auto-detects quantile head from checkpoint
  (`forecast_head.weight` shape) and emits real per-quantile arrays
  for QuantileForecast.

A working run script template is at
[`experiments/freq-embedding/scripts/train.py`](experiments/freq-embedding/scripts/train.py)
and the run_*.sh staging is in `/tmp/` (not committed; copy patterns
from past `git log --oneline` commit messages — search for "STAGE A").

## Queued for the next session

Three experiments, in order. After all three, the user said I can go idle.

### 1. Reproduce the RevIN run for the synth-grid plot

The eval CSV for RevIN+qh is at `results/R1q_femu_revin/all_results.csv`
and is fine. But the model weights themselves were lost (partial scp +
SSH drop). We need them to make the RevIN-equivalent synth grid plot
the user requested.

Reproduce with the same script: `experiments/freq-embedding/scripts/train.py`
with `--freq-emb-dim 3 --mixup-p 0.3 --rev-norm-kind revin --mix-ratio 0.5`,
30k steps. Then the quantile head 30k. Then run the synth-grid plotter:

```
python3.11 experiments/freq-embedding/scripts/plot_synth_qhead.py \
  --backbone <new_revin_backbone>.pth \
  --head     <new_revin_qhead>.pth \
  --out      experiments/freq-embedding/plots/synth_qhead_grid_revin.png
```

Use the **fixed** sync_loop from PR #47 (atomic + rotation). And use
`safe_pull.sh` not raw scp for the final pull.

Budget: ~$1.50, ~6h.

### 2. Synth-only training experiment

Train the backbone on **PURE periodic synth** (no base-bundles mix) at
30k steps and 60k steps. Then a reconstruction head, also on synth-only.
Open question: does the reconstruction-head qualitative behaviour on
training-distribution clean periodics improve when we strip the
real-data half? If yes, the bottleneck on the synth grid was the MIX
ratio diluting periodic learning. If no, it's structural to the
backbone (likely the patch-boundary phase loss the user identified).

Plumbing notes:
- `mix_ratio = 1.0` in the existing `experiments/freq-embedding/scripts/train.py`
  gives pure synth.
- For the reconstruction head, swap the standard `train_forecasting_head.py`
  for one that ALSO uses synth-only (currently the head trains on
  base-bundles; would need a synth-aware loader).
- Run twice: 30k and 60k. Compare the synth-grid plots qualitatively
  before launching expensive full-eval runs.

### 3. RevEWMNorm span sweep

Span = {32 (current), 64, 128, 256}. Same architecture, same data
(fe+mu setup), 30k each backbone + 30k qhead + GIFT-Eval B4. The
hypothesis is that 32 is too short for our periodic-rich setup —
half-life ~11 steps under-tracks long-period structure. Wider span
might preserve more of the periodic amplitude (closer to RevIN
behavior) without RevIN's trend-blindness.

Adjust the 4-point grid based on early findings (e.g. if 256 wins
strictly, try 512). Run only the periodic-focus 6 configs first as a
cheap screen, then the full 97 only on the winning span.

## Open PRs

- **#45** [`feat/periodic-synth-mix`](https://github.com/jeremycochoy/contrastive-forecasting/pull/45) —
  the main results / experimental code / reports. Targets `experiments`.
- **#47** [`fix/sync-rotation-atomicity`](https://github.com/jeremycochoy/contrastive-forecasting/pull/47) —
  the sync-loop fix (atomicity + .prev rotation + safe_pull.sh + CLAUDE.md
  rules). Targets `experiments`.

Merge order: probably #47 first (it's small and a pure improvement),
then #45 (which carries everything else).

## Cost summary

Total vast.ai spend across the sequence: roughly **~$15** of the user's
budget (8 ablations × $1–3 each + a couple wasted hours on slow
networks). vastrun-kit had its known issues (#296) so several
provisions failed and required raw `vastai create` workarounds.
