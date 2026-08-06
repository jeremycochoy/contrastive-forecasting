# Scheduling the EMA momentum to 1.0 by 100k: the teacher encoder does not beat the student

Ten runs trained with the EMA momentum rising linearly from 0.9 to 1.0 at step 100k, each evaluated at every stop through a student head and a teacher head. The teacher encoder does not score better than the student, and no run beats seasonal naive.

## Teacher against student

![Teacher encoder against student encoder](plots/teacher_vs_student.png)

At bb40k the teacher head is lower in 5 of 10 cells and the student head in the other 5, every difference under 0.037, which is the size of a head-seed change.

## The ladder

![GM-Relative MASE against backbone step, every cell, both heads](plots/ladder.png)

Three cells reached bb200k; `arm5_combab_alignT` scored worse on both heads there than at bb100k and the extend rule stopped it.

## Per-domain split

![Per-domain GM-Relative MASE, five lowest entries and the entries that reached bb200k](plots/domain_radar.png)

Every entry sits outside the parity ring on Energy, Web/CloudOps and Econ/Fin, and inside it only on Transport and Nature.

## Uncertainty

![Head-seed spread at bb100k, three seeds per cell per head](plots/seed_spread.png)

Across those cells the head-seed standard deviation runs 0.0019 to 0.0149 and the range 0.0036 to 0.0272, well under the 0.0908 range the teacher-align retrain reported.

![Paired bb40k-to-bb100k delta, both ends at the same head seed](plots/paired_delta.png)

`arm6_v2_combab_alignT` recorded `none_down`, but at two of the three seeds its student head goes down instead, so a change of head seed alone moves where that run stopped.

## The EMA schedule and the step cap

![EMA momentum against training step](plots/alpha_schedule.png)

α is anchored to the fixed step 100k in every run (0.94 at step 40k, 1.00 at step 100k and held there), and one pass over `gift-pretrain-full-4096 / small_v1`, 42,571,692 rows at 64 real rows per step, caps training at **665,182 steps**, far above the deepest run's 200k.

## Tables

### GM-Relative MASE per run, per stop, per head

Source: `experiments/2026-08-04_ema_sched_ladder/results/ladder_all.csv`. Head budget is 15,000 steps at bb40k and 30,000 steps from bb100k on, so a bb40k-to-bb100k comparison moves the head budget as well as the backbone.

| Cell | align | bb40k student | bb40k teacher | bb100k student | bb100k teacher | bb200k student | bb200k teacher |
|---|---|---|---|---|---|---|---|
| `arm6_v2 combab` | student | **1.1603** | **1.1544** | 1.1945 | 1.1837 | — | — |
| `arm6_v2 combab` | teacher | 1.1895 | **1.1793** | 1.1921 | 1.1963 | — | — |
| `arm5 combab` | student | 1.2596 | 1.2347 | 1.2102 | 1.2407 | **1.1910** | — |
| `arm5 combab` | teacher | 1.3334 | 1.3190 | 1.2797 | **1.2772** | 1.4141 | 1.4207 |
| `arm4 combab` | n/a | **1.2503** | 1.2870 | 1.3479 | 1.3188 | — | — |
| `arm6_v2 ncpc` | student | **1.3611** | 1.3656 | 1.4951 | 1.5007 | — | — |
| `arm6_v2 ncpc` | teacher | **1.2955** | 1.3266 | 1.3904 | 1.3646 | — | — |
| `arm6_v2 nse` | student | **1.2690** | 1.2917 | 1.3572 | 1.3770 | — | — |
| `arm6_v2 nse` | teacher | 1.4238 | 1.4177 | 1.3913 | 1.3746 | 1.3586 | **1.3459** |
| `arm1 nse` | n/a | **1.4347** | 1.4512 | 1.5227 | 1.5604 | — | — |

Bold = the row's lowest value. `arm5 combab / student` has no bb200k teacher value: the extend rule dropped its teacher head from the evaluation at bb100k.

### The union table, extended

`prev` = the 30-cell student-align sweep, `new` = the teacher-align retrain. Those columns are the published parent numbers, one head each. The `this study` columns give the student head and the teacher head as `S / T`.

| Cell | align | prev best | new best | this study bb40k S / T | bb100k S / T | bb200k S / T | this study best |
|---|---|---|---|---|---|---|---|
| `arm6_v2 combab` | student | 1.1616 @100k | — | 1.1603 / 1.1544 | 1.1945 / 1.1837 | — | **1.1544** @40k T |
| `arm6_v2 combab` | teacher | — | 1.1850 @200k | 1.1895 / 1.1793 | 1.1921 / 1.1963 | — | 1.1793 @40k T |
| `arm5 combab` | student | 1.2034 @200k | — | 1.2596 / 1.2347 | 1.2102 / 1.2407 | 1.1910 / — | 1.1910 @200k S |
| `arm5 combab` | teacher | 1.2728 @40k | — | 1.3334 / 1.3190 | 1.2797 / 1.2772 | 1.4141 / 1.4207 | 1.2772 @100k T |
| `arm4 combab` | n/a | 1.2748 @40k | 1.2748 @40k | 1.2503 / 1.2870 | 1.3479 / 1.3188 | — | 1.2503 @40k S |
| `arm6_v2 ncpc` | student | 1.2978 @100k | — | 1.3611 / 1.3656 | 1.4951 / 1.5007 | — | 1.3611 @40k S |
| `arm6_v2 ncpc` | teacher | — | 1.3012 @100k | 1.2955 / 1.3266 | 1.3904 / 1.3646 | — | 1.2955 @40k S |
| `arm6_v2 nse` | teacher | — | 1.3074 @40k | 1.4238 / 1.4177 | 1.3913 / 1.3746 | 1.3586 / 1.3459 | 1.3459 @200k T |
| `arm1 nse` | n/a | 1.3308 @200k | 1.3308 @200k | 1.4347 / 1.4512 | 1.5227 / 1.5604 | — | 1.4347 @40k S |
| `arm6_v2 nse` | student | 1.3791 @40k | — | 1.2690 / 1.2917 | 1.3572 / 1.3770 | — | 1.2690 @40k S |

A `—` in a parent column means the cell never placed in that report's five lowest. Parent values are quoted from the two report tables; every `this study` value traces to `results/ladder_all.csv`.

### The raw change the extend rule read

Source: `experiments/2026-08-04_ema_sched_ladder/results/per_stop_changes.csv`. Each row is the change one head made from the previous stop, which is the only quantity the rule compares. Negative is down. Bold marks a change smaller than 0.0272, the largest head-seed range measured here.

| Cell | head | transition | from | to | change | rule read | branch |
|---|---|---|---|---|---|---|---|
| `arm1_nse` | student | 40k→100k | 1.4347 | 1.5227 | +0.0880 | not down | `none_down` |
| `arm1_nse` | teacher | 40k→100k | 1.4512 | 1.5604 | +0.1092 | not down | `none_down` |
| `arm4_combab` | student | 40k→100k | 1.2503 | 1.3479 | +0.0976 | not down | `none_down` |
| `arm4_combab` | teacher | 40k→100k | 1.2870 | 1.3188 | +0.0318 | not down | `none_down` |
| `arm5_combab_alignS` | student | 40k→100k | 1.2596 | 1.2102 | -0.0494 | down | `one_down` |
| `arm5_combab_alignS` | teacher | 40k→100k | 1.2347 | 1.2407 | **+0.0060** | not down | `one_down` |
| `arm5_combab_alignS` | student | 100k→200k | 1.2102 | 1.1910 | **-0.0192** | down | `one_down` |
| `arm5_combab_alignT` | student | 40k→100k | 1.3334 | 1.2797 | -0.0537 | down | `both_down` |
| `arm5_combab_alignT` | teacher | 40k→100k | 1.3190 | 1.2772 | -0.0418 | down | `both_down` |
| `arm5_combab_alignT` | student | 100k→200k | 1.2797 | 1.4141 | +0.1344 | not down | `none_down` |
| `arm5_combab_alignT` | teacher | 100k→200k | 1.2772 | 1.4207 | +0.1435 | not down | `none_down` |
| `arm6_v2_combab_alignS` | student | 40k→100k | 1.1603 | 1.1945 | +0.0342 | not down | `none_down` |
| `arm6_v2_combab_alignS` | teacher | 40k→100k | 1.1544 | 1.1837 | +0.0293 | not down | `none_down` |
| `arm6_v2_combab_alignT` | student | 40k→100k | 1.1895 | 1.1921 | **+0.0026** | not down | `none_down` |
| `arm6_v2_combab_alignT` | teacher | 40k→100k | 1.1793 | 1.1963 | **+0.0170** | not down | `none_down` |
| `arm6_v2_ncpc_alignS` | student | 40k→100k | 1.3611 | 1.4951 | +0.1340 | not down | `none_down` |
| `arm6_v2_ncpc_alignS` | teacher | 40k→100k | 1.3656 | 1.5007 | +0.1351 | not down | `none_down` |
| `arm6_v2_ncpc_alignT` | student | 40k→100k | 1.2955 | 1.3904 | +0.0949 | not down | `none_down` |
| `arm6_v2_ncpc_alignT` | teacher | 40k→100k | 1.3266 | 1.3646 | +0.0380 | not down | `none_down` |
| `arm6_v2_nse_alignS` | student | 40k→100k | 1.2690 | 1.3572 | +0.0882 | not down | `none_down` |
| `arm6_v2_nse_alignS` | teacher | 40k→100k | 1.2917 | 1.3770 | +0.0853 | not down | `none_down` |
| `arm6_v2_nse_alignT` | student | 40k→100k | 1.4238 | 1.3913 | -0.0325 | down | `both_down` |
| `arm6_v2_nse_alignT` | teacher | 40k→100k | 1.4177 | 1.3746 | -0.0431 | down | `both_down` |
| `arm6_v2_nse_alignT` | student | 100k→200k | 1.3913 | 1.3586 | -0.0327 | down | `both_down` |
| `arm6_v2_nse_alignT` | teacher | 100k→200k | 1.3746 | 1.3459 | -0.0287 | down | `both_down` |

Four of 25 changes are inside that band. Two stops rest entirely on them: `arm6_v2_combab_alignT` at 100k, whose `none_down` ended the run on +0.0026 and +0.0170, and `arm5_combab_alignS` at 200k, whose `one_down` rests on -0.0192. The first of those two is the branch that flips when the head seed changes.

Every 40k→100k row moves the head budget from 15,000 to 30,000 steps as well as the backbone.

### How each run ended

Source: `experiments/2026-08-04_ema_sched_ladder/results/stop_reason.csv`.

| Cell | last stop | branch that fired | extended | heads kept | ended by |
|---|---|---|---|---|---|
| `arm6_v2_combab_alignS` | 100k | `none_down` | no | student, teacher | extend rule |
| `arm6_v2_combab_alignT` | 100k | `none_down` | no | student, teacher | extend rule |
| `arm6_v2_ncpc_alignS` | 100k | `none_down` | no | student, teacher | extend rule |
| `arm6_v2_ncpc_alignT` | 100k | `none_down` | no | student, teacher | extend rule |
| `arm6_v2_nse_alignS` | 100k | `none_down` | no | student, teacher | extend rule |
| `arm4_combab` | 100k | `none_down` | no | student, teacher | extend rule |
| `arm1_nse` | 100k | `none_down` | no | student, teacher | extend rule |
| `arm5_combab_alignT` | 200k | `none_down` | no | student, teacher | extend rule |
| `arm5_combab_alignS` | 200k | `one_down` | yes | student | ladder ceiling |
| `arm6_v2_nse_alignT` | 200k | `both_down` | yes | student, teacher | ladder ceiling |

No run was stopped by the compute budget. The two rows marked *ladder ceiling* were still improving at bb200k; the study stopped there, not the rule.

### Head-seed replicates at bb100k

Source: `experiments/2026-08-04_ema_sched_ladder/results/seed_spread.csv`. Head seeds 20260722, 20260723, 20260724, for six of the ten cells.

| Cell | head | mean | sd | range | bb40k-to-bb100k change clears the spread |
|---|---|---|---|---|---|
| `arm6_v2_combab_alignS` | student | 1.1923 | 0.0024 | 0.0047 | yes |
| `arm6_v2_combab_alignS` | teacher | 1.1858 | 0.0044 | 0.0080 | yes |
| `arm6_v2_combab_alignT` | student | 1.1900 | 0.0019 | 0.0036 | **no** |
| `arm6_v2_combab_alignT` | teacher | 1.1922 | 0.0039 | 0.0077 | yes |
| `arm5_combab_alignS` | student | 1.2041 | 0.0053 | 0.0095 | yes |
| `arm5_combab_alignS` | teacher | 1.2437 | 0.0106 | 0.0206 | **no** |
| `arm5_combab_alignT` | student | 1.2887 | 0.0082 | 0.0159 | yes |
| `arm5_combab_alignT` | teacher | 1.2896 | 0.0119 | 0.0238 | yes |
| `arm6_v2_nse_alignS` | student | 1.3580 | 0.0098 | 0.0195 | yes |
| `arm6_v2_nse_alignS` | teacher | 1.3721 | 0.0043 | 0.0082 | yes |
| `arm6_v2_nse_alignT` | student | 1.3955 | 0.0037 | 0.0064 | yes |
| `arm6_v2_nse_alignT` | teacher | 1.3815 | 0.0149 | 0.0272 | yes |

At the cell level (`results/seed_branches.csv`), five of the six replicated cells keep their recorded branch at all three seeds. `arm6_v2_combab_alignT` does not: it flips from `none_down` to `one_down`.

## Protocol

- Backbone `d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3, batch_size=64, seed=20260520`; dataset `gift-pretrain-full-4096 / small_v1`.
- EMA momentum α rises linearly from 0.9 at step 0 to 1.0 at step 100k, anchored to the fixed step 100k in every run, and held at 1.0 after (`results/alpha_schedule.csv`).
- Each run starts fresh at step 0 and trains as one continuous run, checkpointed at each stop and resumed with its optimizer state. `hf_rows_consumed` is checkpointed and restored as `skip_rows`, so a resumed leg continues the stream instead of restarting at the head of the dataset.
- Stops: 40k and 100k unconditionally, then +100k per extension. Extend rule, per head against its own previous stop: both heads down → extend and keep both; one head down → extend and keep that head; neither down → stop.
- Two heads per checkpoint, trained separately, each evaluated on its own encoder: student head on the student encoder, teacher head on the teacher encoder. Head budget 15,000 steps at bb40k, 30,000 steps from bb100k.
- 97 GIFT-Eval configs, official B4 strategy, forecast horizon 16. The seasonal-naive denominator file is byte-identical on every machine (`results/denominator_checksums.txt`), so every score is on one scale.
- Head seed 20260722 for the ladder. Six cells carry two extra head seeds at bb100k; `arm4_combab`, `arm1_nse`, `arm6_v2_ncpc_alignS` and `arm6_v2_ncpc_alignT` carry one seed only. That split was a scope decision: the replicates went to the cells whose extend-rule branch turned on the smallest changes.
- The head keeps `--grad-clip 1.0`. The project rule bans grad clipping; the previous study kept it, and it is kept here for comparability with those numbers.
- GIFT-Eval ran sharded across CPU workers on one machine and unsharded elsewhere. Both write the same 97-row `all_results.csv` and the sharded and unsharded aggregates agree (`results/audit_scores_all.txt`).

## Annex

Artefacts: `experiments/2026-08-04_ema_sched_ladder/`.

| File | Contents |
|---|---|
| `results/ladder_all.csv` | 45 scored stops, pooled across machines |
| `results/per_stop_changes.csv` | the change each head made per stop, and the branch it produced |
| `results/stop_reason.csv` | last stop, branch and cause per run |
| `results/seed_spread.csv` | three head seeds per cell per head at bb100k |
| `results/seed_branches.csv` | whether each recorded branch survives a change of head seed |
| `results/paired_delta.csv` | bb40k-to-bb100k delta with both ends at the same head seed |
| `results/alpha_schedule.csv` | α per 1,000 steps |
| `results/dataset_rows.json` | row count, batch composition and derived step cap |
| `results/denominator_checksums.txt` | seasonal-naive denominator hash per machine |
| `results/eval/<cell>/eval/bb<stop>_<head>/gift/all_results.csv` | per-config MASE behind every score |

`results/paired_delta.csv` carries three seeds for `arm6_v2_combab_alignS` and `arm6_v2_combab_alignT` and fewer for the other rows; rows below `n_seeds=3` are measured, not tested, and carry no significance claim.

Plot sources: `plots/_make_domain_radar.py`, `plots/_make_teacher_vs_student.py`, and `experiments/2026-08-04_ema_sched_ladder/scripts/plot_ladder.py`, `plot_seed_spread.py`, `plot_paired_delta.py`, `alpha_schedule.py`.
