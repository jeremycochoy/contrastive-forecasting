# SIGReg-e inspection (#356 review P1.2)

The issue spec mandates: *"if either `u_batch_e` stays near `1/K`, the corresponding `L_sigreg_*` term should be inspected."* The final trajectory shows `u_batch_e = 0.0438` (vs floor `1/K = 1/384 ≈ 0.00260`) and `u_temporal_e = 0.0315`. The patch-embed `e_t` regulariser failed to fill the sphere. This note answers the three sub-questions.

Data: `runs/bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_losses.csv` (12 500 rows, 50-step rolling means below).

![SIGReg-e inspection](../plots/sigreg_e_inspection.png)

## (a) Trajectory of `sigreg_e` vs `sigreg_h`

| step | `sigreg_e` | `sigreg_h` | `u_batch_e` | `u_batch` (h_t) | loss |
|---:|---:|---:|---:|---:|---:|
| 250 | 1.76e-3 | 2.59e-3 | 0.0100 | 0.4069 | 3.13 |
| 500 | 1.36e-3 | 1.92e-3 | 0.0110 | 0.5379 | 2.99 |
| 1 000 | 7.55e-4 | 1.23e-3 | 0.0126 | 0.6173 | 2.88 |
| 2 000 | 6.41e-4 | 8.57e-4 | 0.0151 | 0.7190 | 3.07 |
| 5 000 | 9.38e-4 | 8.40e-4 | 0.0240 | 0.7867 | 4.50 |
| 7 500 | 9.96e-4 | 5.17e-4 | 0.0341 | 0.7925 | 4.54 |
| 10 000 | 9.70e-4 | 4.12e-4 | 0.0395 | 0.7923 | 4.43 |
| 12 500 | 1.01e-3 | 3.81e-4 | 0.0438 | 0.8020 | 4.24 |

`sigreg_h` decays monotonically by ~54× over training (2.03e-2 at step 1 → 3.81e-4 at the tail). `sigreg_e` drops by ~3.5× over the first 2 000 steps (2.63e-3 → 6.4e-4), then **rises** back to ~1.0e-3 from step 5 000 onward and stays there — i.e. it saturates while the main loss continues to evolve. The two terms **cross** between step 4 000 and 5 000: at the tail `sigreg_e ≈ 2.6 × sigreg_h`. So `sigreg_e` does not "stay flat from the start" — it rebounds against further downward pressure once the encoder starts shaping `h_t`.

## (b) Gradient path from `sigreg_e` to the patch-embedding

The plumbing is intact under the actual run config (`--sigreg-embedding`, `--sigreg-encoding`, `--sigreg-post-normalization` OFF):

- `experiments/2026-04-27_freq-embedding/scripts/train.py:1216` — `want_embed = use_sigreg` is True whenever either flag is set, so `forward_step` returns the 4-tuple including `e_lat`.
- `train.py:1257` — `e_lat = e_lat.float()` upcasts in-place; gradient is preserved (no detach).
- `train.py:1276` — `gather_latents(e_lat, e_lat)` is gated on `args.sigreg_embedding`. With both flags ON this run pools `e_t` over the global batch, identical to the contrastive contract.
- `train.py:1330-1335` — `sigreg_e = sigreg_loss(e_lat, ...)`; `loss = loss + args.sigreg_weight * sigreg_e`. The gradient reaches `e_lat` via the chain rule on the total `loss`.
- `tests/test_sigreg.py:242 test_sigreg_on_real_patch_embed_admits_backward` exercises this end-to-end at the real model: SIGReg → `e_t` produces a non-zero gradient on the GRU patch-embed parameters. **Test passes** on the run code.

Empirical confirmation in the CSV: `u_batch_e` advances `0.009 → 0.045` over 12 500 steps. If the SIGReg-e gradient were not reaching the GRU at all, `u_batch_e` would remain at its initialisation value (~0.009). Movement is small but non-zero, consistent with a live but weak gradient path.

## (c) Magnitude balance — λ·L_sigreg_e versus the total loss

λ = 0.1 (`--sigreg-weight`, the default). Over the last 50 steps:

| quantity | mean |
|---|---:|
| total `loss` | 4.2478 |
| `λ·sigreg_e` | 1.001e-4 |
| `λ·sigreg_h` | 3.805e-5 |
| `λ·sigreg_e / loss` | **2.36e-5** |
| `λ·sigreg_h / loss` | **8.96e-6** |

The SIGReg-e contribution is **~42 000× smaller** than the main loss; SIGReg-h **~110 000× smaller**. The gradient signal from either regulariser is materially swamped by the contrastive + CPC + EMA-target terms.

That `h_t` still reaches `u_batch ≈ 0.80` even with `λ·sigreg_h ≈ 9e-6` of loss is because the contrastive loss `cosine_similarity_batch_full_hh_negs_xshh_allt` directly rewards angular separation between `h_t` vectors — it does the sphere-filling work for `h_t` on its own. The contrastive loss does **not** see `e_t` directly; `e_t` receives gradient only through (i) backprop through the encoder + CPC + main losses (which optimise for `h_t` properties, not `e_t`'s marginal), and (ii) the explicit `L_sigreg_e` term at λ=0.1.

## Conclusion

`L_sigreg_e` is not silently disconnected and not stuck on a numerical floor. Three concrete observations:

1. **Gradient path is live.** Plumbing trace + per-block test + non-zero progression of `u_batch_e` confirm SIGReg's gradient reaches the GRU patch-embed.
2. **Magnitude is swamped.** `λ·L_sigreg_e` is ~2.4e-5 of the total loss at convergence — the regulariser does not have the weight to bend `e_t` against the main-loss pressure. The same arithmetic on `L_sigreg_h` would have looked equally inert if `h_t` were not already being sphere-filled by the contrastive negatives.
3. **Trajectory is not monotone.** `sigreg_e` falls early then **rises** back to a steady ~1.0e-3 from step 5k onward, against further regulariser pressure — consistent with the encoder/CPC objective pushing `e_t` into a distribution the SIGReg term opposes, with no weight to win that contest.

The single-arm spec (issue #355) does not include a λ sweep or post-normalisation variant; a follow-up arm with either a substantially larger `--sigreg-weight` or `--sigreg-post-normalization` ON would be needed to test whether the inertness is the weight or a deeper property of pre-normalised SIGReg on the GRU patch-embed at `K=384`.
