# Improvement leaderboard

**Control (150k backbone + standard 2L 30k q-head, e_then_f, causal):**
triage **1.5740**, full **1.4090**. Target full < 1.292 (v11c).
Triage is ~7–10% noisy (prior-experiment caveat) — small deltas are NOT
significant.

| tier | arm (one variable vs control) | triage(11) | full(97) | verdict |
|---|---|---|---|---|
| 0 | decoding strategy A1/A2/B1/B2/B3/B3R | eval-incompatible | — | ✗ uninformative (errored, not "tested & failed") |
| 0 | q-head input = f_only | 1.5846 | (not run) | ✗ worse on triage |
| 0 | q-head head-causal = false | 1.5643 | 1.4046 | ~ −0.3% (inside noise — not significant) |
| 1 | rev-norm-span 256 | — | — | started, **halted by user**, no result |

**Note (not a verdict).** Only a handful of head/eval arms were run, and
the decoding sweep errored rather than testing the lever — this is *not*
a systematic head search. No completed Tier-1 (backbone) experiment.
"The gap is architectural / backbone-side" remains an **untested
hypothesis**; nothing here isolates a cause.

_Exploration halted by user 2026-05-18 — only the 150k-step q-head
hypothesis kept._
