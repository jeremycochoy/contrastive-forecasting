# Frequency and seasonality embedding: vocabulary and coverage

## What this document is

The model has two parallel categorical embeddings:

* **Frequency** — wall-clock sample rate. 10 buckets including `unknown`.
* **Seasonality** — dominant period in samples. 10 buckets including
  `unknown / no period`.

Both are concatenated to every patch and learned end-to-end. This file
specifies the buckets, how they map to GIFT-Eval's per-task labels, what
combinations the on-the-fly synthesizer covers, and which combinations
are intentionally exposed in synth even though GIFT-Eval doesn't include
them.

## Vocabulary

### Frequency (`FREQ_NAMES`, 10 classes)

| id | label | maps from gluonts/pandas freq |
|----|-------|-------------------------------|
| 0 | `unknown` | freq we can't recognise (monthly, yearly, sub-second, …) |
| 1 | `10s`     | `10S` |
| 2 | `1min`    | `T`, `min`, `1T`, `1min` |
| 3 | `5min`    | `5T`, `5min` |
| 4 | `10min`   | `10T`, `10min` |
| 5 | `15min`   | `15T`, `15min` |
| 6 | `30min`   | `30T`, `30min` |
| 7 | `1h`      | `H`, `1H` |
| 8 | `1d`      | `D`, `1D` |
| 9 | `1w`      | `W`, `1W`, any `W-*` |

The mapping lives in `src/freq_embedding.py:gluonts_freq_to_id`.

### Seasonality (`SEASONALITY_NAMES`, 10 classes)

Doubling buckets on samples-per-period (`spp`). Bucket **0** is the
"no period info" sentinel and is also returned when `spp <= 1`, because
gluonts's default seasonality for daily/weekly is 1 (treated as
non-periodic). Sending `spp=1` to the same row as truly-unknown labels
means the embedding doesn't try to learn a period from a no-period
signal.

| id | range | typical examples |
|----|-------|------------------|
| 0 | `spp <= 1`, or unknown | gluonts default for D/W; gift train rows; bundle synth rows |
| 1 | `2 ≤ spp ≤ 4`           | very short cycles |
| 2 | `5 ≤ spp ≤ 8`           | weekly on daily (7); business day patterns |
| 3 | `9 ≤ spp ≤ 16`          | yearly on monthly (12) |
| 4 | `17 ≤ spp ≤ 32`         | daily on hourly (24); monthly on daily (30) |
| 5 | `33 ≤ spp ≤ 64`         | daily on 30-min (48); yearly on weekly (52) |
| 6 | `65 ≤ spp ≤ 128`        | daily on 15-min (96) |
| 7 | `129 ≤ spp ≤ 256`       | daily on 10-min (144); weekly on hourly (168) |
| 8 | `257 ≤ spp ≤ 512`       | daily on 5-min (288); ~daily on 10s (360) |
| 9 | `spp > 512`             | weekly on 5-min (2016); yearly on daily (365); ... |

The function lives in `src/freq_embedding.py:seasonality_to_id`.

## What GIFT-Eval covers (97 configs, 14 distinct (freq, seas) pairs)

| freq token | gluonts seasonality | (freq_id, seas_id) | n configs | examples |
|------------|--------------------:|:------------------:|----------:|----------|
| 10S    | 360 | (1, 8) | 6  | bizitobs_application |
| 5T     | 288 | (3, 8) | 12 | LOOP_SEATTLE/5T, bitbrains_*/5T, bizitobs_l2c/5T |
| 10T    | 144 | (4, 7) | 6  | jena_weather/10T, solar/10T |
| 15T    | 96  | (5, 6) | 12 | electricity/15T, ett1/15T, ett2/15T, SZ_TAXI/15T |
| H      | 24  | (7, 4) | 31 | electricity/H, ett1/H, kdd_cup_2018/H, ... |
| D      | 1   | (8, 0) | 15 | covid_deaths, m_dense/D, LOOP_SEATTLE/D, ... |
| W-FRI  | 1   | (9, 0) | 2  | electricity/W, solar/W |
| W-SUN  | 1   | (9, 0) | 1  | m4_weekly |
| W-THU  | 1   | (9, 0) | 3  | ett1/W, ett2/W, saugeenday/W |
| W-TUE  | 1   | (9, 0) | 1  | us_births/W |
| W-WED  | 1   | (9, 0) | 1  | hierarchical_sales/W |
| M      | 12  | (0, 3) | 5  | car_parts, hospital, m4_monthly, saugeenday/M, us_births/M |
| Q-DEC  | 4   | (0, 1) | 1  | m4_quarterly |
| A-DEC  | 1   | (0, 0) | 1  | m4_yearly |

Notes on this table:
* `M`, `Q-DEC`, `A-DEC` map to `freq_id=0` because the freq vocabulary
  intentionally stops at `1w`. Anything monthly or longer collapses into
  the unknown bucket. Their seasonality labels stay accurate.
* All `W-*` weekly variants land in `(9, 0)` per gluonts's default; the
  weekday suffix doesn't change the cycle.

## Joint (freq, seasonality) sampling in synth

The on-the-fly periodic synth (`src/synthetic_periodic.py:generate_periodic_batch`
with `return_labels=True`) samples a pair per batch row:

1. `freq_id ~ Uniform({1, ..., 9})`
2. `seasonality_id ~ Uniform({0, ..., 9})` — independent of freq
3. `spp ~ LogUniform(SEASONALITY_BUCKET_SPP_RANGES[seasonality_id])` per channel

The two axes are sampled independently so every cell of the 9×10=90 grid
gets non-zero training density. At a 5000-sample batch every cell
appears at least once and the marginals are within ±10% of uniform.

`SEASONALITY_BUCKET_SPP_RANGES` (in `freq_embedding.py`):

| bucket | spp range | what the visible signal looks like |
|--------|-----------|-------------------------------------|
| 0 | `[1024, 4096]` | a fraction of one cycle in 1024 samples → looks aperiodic |
| 1 | `[2, 4]`       | dozens to hundreds of cycles |
| 2 | `[5, 8]`       | many short cycles |
| 3 | `[9, 16]`      | tens of cycles |
| 4 | `[17, 32]`     | ~30 cycles |
| 5 | `[33, 64]`     | ~16 cycles |
| 6 | `[65, 128]`    | ~8 cycles |
| 7 | `[129, 256]`   | ~4-8 cycles |
| 8 | `[257, 512]`   | ~2-4 cycles |
| 9 | `[513, 1024]`  | ~1-2 cycles |

The freq label carries no constraint on spp. A row tagged `(freq=1d, seas=2)`
generates a periodic signal with cycle length 5–8 samples and labels it as
"daily-sampled, weekly seasonality" — equivalent to the wiki-daily real
case (daily data with a 7-day cycle).

## Combinations exposed in synth that GIFT-Eval doesn't include

GIFT-Eval picks one seasonality per freq via `gluonts.time_feature.get_seasonality`.
The synth fills in the alternatives so the model sees them at training time:

| (freq_id, seas_id) | physical interpretation | in GIFT-Eval? |
|--------------------|--------------------------|:-:|
| (1, 7) `10s` × 168-256 | 10s sampling with ~30-min cycle | no |
| (3, 3) `5min` × 12 | 5-min with hourly cycle | no |
| (4, 2) `10min` × 6 | 10-min with hourly cycle | no |
| (7, 7) `1h` × 168  | hourly with weekly cycle | **no** (GIFT only tests daily-on-hourly = 24) |
| (8, 2) `1d` × 7    | daily with weekly cycle | **no** (GIFT uses gluonts default = 1) |
| (8, 4) `1d` × 30   | daily with monthly cycle | no |
| (8, 9) `1d` × >512 | daily with yearly cycle (365) | no |
| (9, 1)–(9, 5) `1w` × 4..52 | weekly with monthly to yearly cycles | no (GIFT uses 1) |

The two highlighted rows are the most important: real-world daily and
hourly series very commonly have weekly cycles, but GIFT-Eval tests
both with seasonality=1 (daily) or seasonality=24 (hourly = daily cycle
only). A model trained only on the GIFT-Eval pairing wouldn't see
weekly-on-daily or weekly-on-hourly; the synth fills this gap.

## Combinations also covered: weekly with both seas=1 and seas=7

Per the user's spec, weekly-cycle data should appear in synth at the
two natural sample rates:

* `(freq=1w, seas=0)` — gluonts default, "no useful period at the
  weekly sample rate". Synth bucket 0 covers this.
* `(freq=1d, seas=2)` — daily sample rate with 7-day cycle. Synth
  samples seas=2 with spp∈[5,8] ⊃ {7}. Covered.

The model sees both forms tagged differently and learns that weekly
cycles can be expressed as either depending on the sample rate.

## What `seasonality_id = 0` means to the model

Three distinct kinds of rows land in bucket 0:

1. **Unknown** — gift training rows and bundle-synth rows where the
   build pipeline didn't preserve metadata (~74% of bundle rows).
2. **Trivial seasonality (gluonts default 1)** — daily and weekly
   GIFT-Eval configs at eval time (`get_seasonality(D) = 1`,
   `get_seasonality(W-*) = 1`).
3. **No visible period in the window** — synth rows sampled with
   `seasonality_id = 0`, which use `spp ~ [1024, 4096]` so at most a
   fraction of one cycle is visible.

All three share the same embedding row, which functions as a
"period-agnostic" prior: the model should fall back to other features
(values, patch stats, freq embedding) and not try to extract a periodic
component.

## Frequencies absent from both GIFT-Eval and synth

Industry-common rates not represented anywhere in our pipeline:
sub-second (`1S`, `30S`), `30min`, `2H`, `4H`, `12H`, `1M`, `1Q`, `1Y`.
The freq vocabulary stops at `1w`. Monthly and longer series will be
labeled `freq_id=0` (unknown); the seasonality embedding still applies.
This is acceptable for now because GIFT-Eval's contribution from the
M/Q/A buckets is small (7 out of 97 configs).

## Files

* `src/freq_embedding.py` — vocabularies, lookups, embedding modules
* `src/synthetic_periodic.py` — the joint-sampling synth
* `tests/test_seasonality_embedding.py` — vocab/lookup tests
* `tests/test_synthetic_periodic.py` — coverage and reproducibility tests
* `experiments/exp_dualemb_3arm/REPORT.md` — first downstream eval
  using the embeddings (single-axis spp_to_freq_id then; this doc
  describes the dual-axis successor)
