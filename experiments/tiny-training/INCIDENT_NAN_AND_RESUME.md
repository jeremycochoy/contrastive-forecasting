# Incident Report: NaN Crash & Incomplete Resume State

**Date:** 2026-04-14
**Affected runs:** `tiny_fresh` (Vast.ai), `forecasting_head_50k` (Elisa), `tiny_150k` (Elisa)

## Timeline

### Step 1: NaN crash at step 24,970

**Root cause:** Row ~2,397,070 in shard_00239.parquet of
`jeremycochoy/contrastive-training-tiny-bundles/tiny_mixed_v1` is entirely NaN
(all 1025 values). The `_forward_fill_nan()` function had:

```python
if not valid.any():
    return  # all-NaN sequence — silently passes NaN through
```

At step 24,969, this all-NaN row entered the batch as channel 2 of sample 11.
NaN propagated through RevEWMNorm → GRU encoder → transformer → contrastive loss.
The NaN loss produced NaN gradients, which corrupted model weights via the
optimizer update. At step 24,970, the now-corrupted model produced NaN on a
clean batch, triggering the NaN detection and emergency stop.

Both the backbone (`tiny_fresh`) and the forecasting head hit the exact same
row at the exact same step — they stream the same HF data in the same order.

### Step 2: Intermediate fix — zero-fill (harmful)

First fix replaced the early return with `arr[:] = 0.0`. This prevented the
crash but injected a bad training signal: a batch with one all-zeros channel
produces zero-norm latents, which give meaningless cosine similarity values.

The `tiny_150k` run on Elisa used this fix. At step ~25k, the zero-filled
batch entered training. It did not crash, but it destabilized the model:
- Steps 25k–40k: gap dropped from 0.40 → 0.17 (loss spike up to 4.17)
- Steps 40k–50k: model recovered to gap 0.57

The 15k-step dip is visible in the training curve and was entirely caused by
the zero-fill injecting a corrupted gradient.

### Step 3: Correct fix — skip all-NaN rows

Final fix: `_forward_fill_nan()` returns `False` for all-NaN arrays, and
callers (`ShardDataset._load_shard`, `HFStreamingLoader._raw_iter`) skip the
row entirely. No bad signal enters training.

Logic: 1) forward-fill, 2) backfill, 3) if still NaN after both, skip.

### Step 4: Incomplete resume state discovered

On investigating the training curve, we found that checkpoint resume loses
several pieces of state:

| State | Saved? | Effect of loss |
|---|---|---|
| Model weights | ✅ | — |
| Optimizer (AdamW momentum/variance) | ✅ | — |
| Step counter | ✅ | — |
| Best gap + step | ✅ | — |
| **Best loss + step** | ❌ | `best_loss` resets to `inf` → best_loss checkpoint immediately overwritten |
| **EMA loss / EMA gap** | ❌ | EMA restarts from `None` → noisy metrics for ~500 steps after resume |
| **Random state** (torch, numpy) | ❌ | Different random augmentations on same data after resume |
| **Actual HF rows consumed** | ❌ | Computed as `step * rows_per_step`, but skipped all-NaN rows make this an overestimate |

## Fixes Applied

### 1. `src/dataloader.py` — skip all-NaN rows
- `_forward_fill_nan()` returns `bool` (True=clean, False=skip)
- Callers skip rows that return False
- 14 regression tests in `tests/test_nan_robustness.py`

### 2. `src/checkpoint.py` — save complete training state
- Added fields: `best_loss`, `best_loss_step`, `ema_loss`, `ema_gap`,
  `hf_rows_consumed`, `rng_state_torch`, `rng_state_numpy`
- Backward-compatible: missing fields fall back to defaults on load

### 3. `experiments/tiny-training/scripts/train.py` — restore full state
- Restores all new fields from checkpoint on resume
- Uses `torch.set_rng_state()` / `np.random.set_state()` for reproducibility
- Uses saved `hf_rows_consumed` (not computed from step) for accurate data position

## Lessons

1. Never fill missing data with constants (zeros) — skip the row instead.
2. Save ALL training state in checkpoints, not just model + optimizer.
3. Always verify background scripts produce correct output on first run.
4. When fixing a data bug, re-run from a clean checkpoint; don't continue
   from a checkpoint that was trained with the buggy code.
