# Restart / migration protocol

If a training run is killed on machine A and resumed on machine B, historical logs must be **archived under a step-range name on machine A** *before* the sync loop starts pulling from machine B. Otherwise the sync overwrites the pre-migration history with the short post-resume version.

## The 4 files to copy

For any run identified by `<name>`, before shipping the resume bundle:

| File                             | Copy to                                           |
|----------------------------------|---------------------------------------------------|
| `<name>_losses.csv`              | `<name>_losses_step_1_to_<N>.csv` (on machine A)  |
| `<name>_attn_amplitude.csv`      | `<name>_attn_amplitude_step_1_to_<N>.csv`         |
| `run_<name>.log`                 | `run_<name>_step_1_to_<N>.log`                    |
| `<name>_<M>k.pth` (+ optimizer)  | pick the highest-step periodic pair as the resume source and rename it `resume_source_step<M000>.pth` (+ `_optimizer.pth`) on machine B *before* launch, so `safe_run_name` does not branch to `_r2` |

`<N>` = the last step in `losses.csv` at the moment of the kill.

## Order of operations

1. Stop training on machine A.
2. `cp` the 3 log/csv files to their `_step_1_to_<N>` names on machine A.
3. `scp` those `_step_1_to_<N>` files + the periodic `.pth` (+ optimizer) to machine B.
4. On machine B, rename the resume `.pth` (+ optimizer) out of the `<name>_*` pattern so `safe_run_name` does not add `_r2`.
5. Truncate the copied `_losses.csv` on machine B to step ≤ N (the trainer opens in append mode; without truncation the CSV double-counts).
6. Launch training on machine B with `--resume` pointing at the renamed `.pth`.
7. Start the sync loop **only after** step 2 has landed on machine A.

## Concat at report time

The archived `_step_1_to_<N>` files + the new `_losses.csv` from machine B are concatenated at report-writing time:

```bash
head -1 <name>_losses.csv > <name>_losses_full.csv
tail -n +2 <name>_losses_step_1_to_<N>.csv >> <name>_losses_full.csv
tail -n +2 <name>_losses.csv >> <name>_losses_full.csv
```

## Why this exists

#374 lost the split_pred_rep backbone losses for steps 1–900 because the pre-migration `losses.csv` on elisa was overwritten by the sync loop pulling machine B's shorter version. The archive-then-ship order below prevents that.
