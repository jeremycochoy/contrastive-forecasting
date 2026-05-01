# exp_realonly_4096_smaller_2arm — smaller-arch variant of exp_realonly_4096_2arm

*Written: 2026-04-30. Date-stamp added: 2026-05-02.*

## Question

Same setup as `exp_realonly_4096_2arm` (real-data-only training on
`jeremycochoy/gift-pretrain-small-4096` at T=4096 / C=1 / mix=0.0) but
with a **smaller backbone**:

| Knob          | Tiny (#19)        | Smaller (this exp)        |
| ------------- | ----------------- | ------------------------- |
| num_layers    | 6                 | 6                         |
| H (d_model)   | 512               | **384**                   |
| nhead         | 8                 | **6**                     |
| ffn_mult      | 4.0               | 4.0                       |
| W (patch)     | 16                | 16                        |
| Params        | **19,956K**       | **11,429K** (43% smaller) |

The user phrased the request as "instead 6L layers" — both archs are
already L=6, so the actual change is in the hidden dim and head count.

## Hypothesis

A smaller model may train faster per step, generalise differently
(less overfit risk, more inductive bias from the smaller capacity),
and serve as the basis for an even longer training run on the bigger
gift-pretrain-base dataset (task #21) once we know which capacity wins
at the small-data 30k-step budget.

## Setup

Identical to `exp_realonly_4096_2arm` except for the three new
training-script CLI flags:

* `--d-model 384`
* `--n-heads 6`
* `--num-layers 6` (explicit — defaults to 6 anyway, but kept for clarity)

Two arms (parallel): `ewma128` (rev_norm_kind=ewma, span=128),
`revin` (rev_norm_kind=revin). Total steps 30k each, bs=24, lr=1e-4,
freq+seas-emb 3, mixup_p=0.3, --grad-clip 1.0.

## Setup notes

Re-uses the same two 5090 vast.ai instances from exp_realonly_4096_2arm
once those finish their pipeline. No new instance provisioning needed
— just kick off `bash experiments/exp_realonly_4096_smaller_2arm/run.sh
<arm>` on each machine after the realonly-4096 pipeline ends.

Sync dir: `sync_realonly_4096_smaller/<arm>/` in main checkout
(separate from `sync_realonly_4096/<arm>/`).

## Status

- [x] CLI flags added (--d-model / --n-heads / --num-layers)
- [x] Forward+backward smoke test at L=6 H=384 (11.4M params verified
      on 5090; output shape (24, 256, 1, 384) correct)
- [x] run.sh built
- [ ] sync_loop_smaller.sh
- [ ] EWMA arm launched (waiting on EWMA #19 to complete)
- [ ] RevIN arm launched (waiting on RevIN #19 to complete)
- [ ] Both arms ALL DONE
- [ ] 4-way comparison plot vs Tiny (#19)
- [ ] REPORT.md
