# τ sweep — fixed-τ from-scratch trainings

## Goal

See whether fixing the contrastive temperature τ to specific values
changes the converged AUC vs backbone-beta's learnable τ (which settled
near τ ≈ 0.072 after 167k steps). Each arm is trained **from scratch**
at one fixed τ; everything else matches backbone-beta exactly.

## Reference (not an arm)

backbone-beta is the learnable-τ run already in this repo — it stays as
an external reference point, **not** an arm in this sweep. The
apples-to-apples comparison inside the sweep is fixed-τ vs fixed-τ.

## τ values

| τ    | run name           | rationale                                           |
|------|--------------------|-----------------------------------------------------|
| 0.03 | `tau_sweep_0_03`   | sharp — punishes near-misses harder                 |
| 0.05 | `tau_sweep_0_05`   | moderately sharp                                    |
| 0.07 | `tau_sweep_0_07`   | closest fixed value to backbone-beta's converged τ  |
| 0.10 | `tau_sweep_0_10`   | moderately soft                                     |
| 0.20 | `tau_sweep_0_20`   | soft — high entropy, harder to discriminate         |

## Recipe (per arm)

- From scratch (no `--resume`); **fixed** τ (no `--learnable-tau`).
- Arch: T_RAW=4096, C=1, d_model=384, num_layers=6, n_heads=6,
  freq_emb_dim=3, seasonality_emb_dim=3, rev_norm_kind=ewma span=128.
- Optim: AdamW lr=1e-3 wd=0.1 β1=0.9 β2=0.98.
- Aug: mixup_p=0.3, mix_ratio=0.0; loss `cosine_similarity_batch`.
- batch_size=256, total_steps=50000, save_every=5000.

After each arm: `cp <run>_best_loss.pth <run>_FINAL.pth`.

## Budget rationale

50k steps per arm — backbone-beta ran 167k but its AUC was already
near-converged well before that, so 50k is "just enough to acquire
knowledge" while keeping all five arms tractable. **We can extend the
winner later** if 50k turns out to be too short.

## Idempotency

The launcher skips any arm whose `<run>_FINAL.pth` already exists, so a
crash mid-sweep is a safe restart.
