# What two arms on one card cost

`results/smoke_k16.csv` measures one leg alone on GPU 0. This measures two
legs at the same time, on the same card, 1,000 steps in. It is the number
`scripts/launch_elisa.sh` sets its arm count from.

| depth | alone (ms/step) | with a second arm (ms/step) | ratio |
|------:|----------------:|----------------------------:|------:|
| k = 16 | 347.6 | 375.9 | 1.08 |
| k = 8  | 257.1 | 277.8 | 1.08 |

Two arms give 1.85x the throughput of one. GPU 0 reports 94% utilization
with both running.

A third arm does not follow. GPU 0 carries 5.4 GiB of another session's
processes, and two legs take 10.9 GiB more, which leaves 7.8 GiB. The head
gate in #373's `head_eval_bb.sh` asks for 7.0 GiB before it starts a head. A
third leg would leave 2.4 GiB, so every head of phase 1 would wait on that
gate and time out.

Measured at 2026-08-16 00:33 on elisa, GPU 0.
