# τ sweep — fixed-τ fine-tunes from backbone-beta

## Goal

Determine whether the fixed contrastive temperature τ at which backbone-beta
converged (~0.0720, learnable) is optimal for downstream AUC, or whether a
sharper / softer τ improves performance after a short fine-tune.

## This is NOT a from-scratch training

Each arm is a **20 000-step fine-tune** resumed from backbone-beta's
`best_loss` checkpoint (167k steps, learnable-τ run). All arms share
identical architecture and HP with backbone-beta — only τ differs, and it
is now **fixed** (not learnable).

## τ values

| τ    | rationale                                                |
|------|----------------------------------------------------------|
| 0.03 | sharp — punishes near-misses harder                      |
| 0.05 | moderately sharp                                         |
| 0.07 | **control** — closest fixed value to converged learnable τ ≈ 0.0720 |
| 0.10 | moderately soft                                          |
| 0.20 | soft — high entropy, harder to discriminate              |

## Recipe (per arm)

- Resume: `checkpoints/backbone_beta_167k.pth` (vast-side; user scp's it
  from local `sync_realonly_full4096_moirai_hp_FRESH_RESUME50k/moirai_hp_FRESH_RESUME50k/checkpoints/tiny_full4096_moirai_hp_FRESH_RESUME50k_best_loss.pth`)
- Arch: T_RAW=4096, C=1, H=384, num_layers=6, nhead=6, freq_emb_dim=3,
  seasonality_emb_dim=3, rev_norm_kind=ewma span=128
- Optim: AdamW lr=1e-3 wd=0.1 β1=0.9 β2=0.98
- Aug: mixup_p=0.3, mix_ratio=0.0; loss `cosine_similarity_batch`, fixed τ
- batch_size=256, total_steps=20000, save_every=5000

After each arm: `cp <run>_best_loss.pth <run>_FINAL.pth`. Run names:
`bb_beta_tau_fixed_{0_03,0_05,0_07,0_10,0_20}`.

## Resume gotchas (handled in launcher)

1. **`log_inv_tau` mismatch** — backbone-beta carries `log_inv_tau` in its
   state_dict; with learnable-τ off, strict load fails. Launcher pre-strips
   the key into `backbone_beta_167k_stripped.pth` (idempotent).
2. **Optimizer companion** — Strategy A: omit `<stripped>_optimizer.pth`.
   Trainer falls back to fresh AdamW. For a 20k-step fine-tune from
   already-converged weights this is fine, and avoids the `log_inv_tau`
   Adam-moments shape mismatch.
