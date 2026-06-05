# RevIN on in-distribution synth: best of the original five, but at the wrong span

## Question

An earlier run found RevIN (per-instance z-score input normalisation)
helped on a GIFT-Eval setup — but that setup is out-of-distribution
relative to training. Does RevIN actually help on the **in-distribution**
synthetic data, where the normaliser arms can be compared cleanly?

> *RevIN = reversible instance normalisation: z-score each input window
> by its own mean/std, forecast in normalised space, invert on output.
> The incumbent normaliser is RevEWMNorm — a reversible exponential
> moving average with a span (window) hyperparameter; the four EWMA arms
> here all used span=32. RevIN has no span (it is a per-instance z-score),
> so the five-arm comparison is RevIN against four EWMA span=32 arms.*
>
> *`fe+mu` = frequency-embedding (a learned dim-3 period code) plus a
> per-channel mean (`mu`) feature on the backbone input. `pstats` =
> patch-stats: per-patch summary statistics (here the first-difference
> `diff` kind) fed alongside the patches. "Contrastive gap" = the
> train-time margin between positive (forecast-vs-future) and negative
> cosine similarities; larger = better-separated representation.*
>
> *GM-MASE = geometric mean over the eval set of (model MASE ÷ per-series
> MAE scale). Lower is better; seasonal-naive = 0.497 on this protocol.*

## Result

RevIN beat all four EWMA arms (best of the five arms in this comparison)
— but every one of the five is **4.5–5.0×** (≈4.8× mean) worse than
seasonal-naive. RevIN's gap is the smallest, at 4.5×. The four EWMA arms
all share the span=32 normaliser that the later span sweep showed was the
real bottleneck; RevIN is a different normaliser kind (per-instance
z-score, no span), so its gap to seasonal-naive is not a span effect.

![GM-MASE across the five original synth arms — RevIN vs four EWMA span=32 arms (single seed, 1024 held-out synth samples). RevIN-synth @ 60k (blue) is lowest at 2.230; the fe+mu and fe+mu+pstats EWMA arms follow. The seasonal-naive reference (0.497, red dashed) sits far below all of them — every arm here is 4.5–5.0× worse than seasonal-naive.](plots/revin_eval.png)

RevIN-synth @ 60k lands at **GM-MASE 2.230**, about **5.75% better** than
the next arm fe+mu @ 60k (2.366) and ahead of fe+mu @ 30k (2.394),
fe+mu+pstats @ 60k (2.411), and fe+mu+pstats @ 30k (2.485). So within
this group RevIN wins — consistent with the hypothesis that span=32 was
over-smoothing the periodic structure on synth.

But "RevIN beats EWMA" only holds at span=32, which the
[span sweep](../2026-04-27_exp_span_sweep_synth/exp_span_sweep_synth.md)
later showed is the wrong span. At the right span, EWMA wins by a wide
margin: span=512 reaches **GM-MASE 0.848** — 2.6× better than RevIN's
2.230, and the closest arm in this whole family to seasonal-naive (still
−71%). So the correct reading is **RevIN > EWMA at span=32 (the wrong
span); at the right span EWMA wins by a wide margin**, and the
normaliser-kind choice mattered far less than the span knob.

## Protocol

| Knob | Value |
|---|---|
| Steps | 60k backbone, 30k qhead |
| Mix ratio | 1.0 (synth-only, in-distribution) |
| Freq emb | dim=3, mixup=0.3 |
| Reversible norm | RevIN (per-instance z-score) |
| Patch stats | none |
| Loss | `cosine_similarity_batch_no_time_neg` |
| Eval | 1024 held-out synth samples; seasonal-naive GM-MASE = 0.497 |

The RevIN backbone trained to ~60k steps; the quantile head trained 30k
steps on synth-only. The 60k backbone budget (vs 30k for the fe+mu /
pstats arms) gave RevIN extra compute given its lower training-time
contrastive gap on synth. (Training-time gaps — RevIN ~0.77 vs
fe+mu+EWMA ~0.85 — were eyeballed during the run; no losses CSV is
committed, so treat them as unlogged observations, not verifiable
numbers.) Every GM-MASE / GM-WQL number above is the matching `arm` row in
[`../2026-04-27__aggregate/results/synth_eval.csv`](../2026-04-27__aggregate/results/synth_eval.csv);
[`scripts/plot_revin_eval.py`](scripts/plot_revin_eval.py) reads that CSV
and emits the figure.

## What we learned (single seed)

1. **RevIN was the best of the original five synth arms** (it beat all
   four EWMA span=32 arms) — ~5.75% better GM-MASE than fe+mu @ 60k —
   supporting the read that the shared span=32 normaliser was
   over-smoothing periodic structure on synth.

2. **That win is an artefact of the wrong span.** Once the span sweep
   opened the span knob, EWMA at span=512 (GM-MASE 0.848) beat RevIN
   (2.230) by 2.6×. The normaliser *kind* was a second-order effect; the
   *span* was the bottleneck. RevIN was not carried forward.

*Single seed throughout — one eval, no per-sample variance, so no error
bars are possible from the committed data. The 5.75% RevIN-vs-fe+mu
ordering is therefore **not seed-separated** and could be within
single-seed noise. The span-sweep gap (2.6×), by contrast, is far too
large to be plausibly seed noise.*

---

### Annex: full eval table

| Arm | GM-MASE | GM-WQL | MASE skill | WQL skill |
|---|---:|---:|---:|---:|
| **RevIN-synth @ 60k** | **2.230** | **1.201** | **−348%** | **−249%** |
| fe+mu @ 60k (span=32) | 2.366 | 1.293 | −376% | −276% |
| fe+mu @ 30k (span=32) | 2.394 | 1.306 | −381% | −280% |
| fe+mu+pstats @ 60k | 2.411 | 1.319 | −385% | −283% |
| fe+mu+pstats @ 30k | 2.485 | 1.368 | −400% | −298% |
| *EWMA span=512 (span sweep, for reference)* | *0.848* | *0.413* | *−71%* | *−20%* |

*MASE/WQL skill = percent improvement over seasonal-naive; negative =
worse than seasonal-naive. GM-WQL = geometric-mean weighted quantile
loss (probabilistic accuracy). Source rows in
[`../2026-04-27__aggregate/results/synth_eval.csv`](../2026-04-27__aggregate/results/synth_eval.csv).*

### Artefacts

- Backbone `checkpoints/tiny_femu_revin_synth60k_FINAL.pth`, qhead
  `checkpoints/R1q_femu_revin_synth60k_FINAL.pth` — ~80 MB / ~2.5 MB,
  **not tracked in git**.
- Eval CSV row: `RevIN-synth @ 60k`.
- Run reproduction and the run timeline are in
  [`notes/README.md`](notes/README.md).
