«Agent ExperimentRunner claude-opus-5 writing»

## Round: the idle card is answered, and the last number is 10 min out

**Experiment directory:** `reports/2026-08-08_rollout_depth/`
(`results/`, `plots/`, `scripts/`, `sync/`; run tree `/home/jupyter/cf373_r3`)

### Coverage, 14 cells x 3 stops x 2 heads

```
cell      40k S     40k T     100k S    100k T    200k S    200k T
A1        done      done      done      done      stop      stop
A2        done      done      done      done      done      done
A3        done      done      done      done      done      done
A4        done      done      done      done      RUNNING   stop
B1        done      done      done      done      done      done
B2        done      done      done      done      done      done
B3        done      done      done      done      stop      stop
B4        done      done      done      done      done      done
B5        done      done      done      done      stop      stop
B6        done      done      done      done      done      done
B7        done      done      done      done      stop      stop
B8        done      done      done      done      stop      stop
B9        done      done      done      done      stop      stop
B10       done      done      done      done      done      done

deliverables 71   done 70   running 1   queued 0   NOT STARTED 0   (+13 stops)
```

### The idle GPU: I take (b), because (a) has nothing left to place

`bb_A4_200k` finished at 10:50Z. It was the **last backbone in the study**. The
queue now holds exactly one GPU job, `hd_A4_200k_student`, and it is already on
box GPU 0 at 20,500 of 30,000 steps, 14.4 sps. No queued backbone and no queued
head exist to move onto GPU 1. Moving the running head there would throw away 22
minutes of its own work.

So the box goes as soon as that head lands and its artefacts verify on this
disk. `q_finish.sh` (pid 1422698, alive since 08:42Z) owns the destroy: it waits
for every `bb_`/`hd_` job to be terminal, waits one full 900 s sync tick, then
checks **every** done job's artefact on this disk by name and size before it
calls `vastrun-destroy`. Its `VERIFY_ONLY=1` gate passed at 11:12Z on all 45
done jobs. The `ev_A4_200k_student` eval and every other 97-config eval run on
elisa cores and cost nothing.

### Runs completed

```
queue     47 jobs   done=45  running=1  queued=1  failed=0
backbone  9 legs: B8 0 -> 100k, eight cells 100k -> 200k
heads     19, 30,000 steps, seed 20260722, --grad-clip 1.0
evals     19, 97 GIFT-Eval configs, strategy B4, horizon 16
extra     4 A1/B3 reproduction heads + their own 97-config evals
```

### Headline numbers

**bb100k, student, against published `k = 0`, head-seed band 0.0384:
9 better, 3 flat, 2 worse.**

| | cells |
|---|---|
| better | B10 -0.151, A2 -0.143, B9 -0.125, A4 -0.114, B4 -0.087, B6 -0.083, B3 -0.078, B1 -0.074, A1 -0.043 |
| flat | B7 +0.019, B5 +0.016, B8 -0.021 |
| worse | A3 +0.109, B2 +0.093 |

**bb200k against the cell's own bb100k, 14 contrasts: 5 improved, 9 got worse.**
Mean +0.0103, median +0.0080. The band covers 11 of 14. Three sit outside it:
A3 student +0.0988, B4 teacher +0.0454, B2 student -0.0539. Deeper training does
not pay here.

### Spend

Credit **$11.78**. Box `cf373-dual` at $0.8144/h, $16.23 on the meter over 20.0 h.
It is needed about 40 min more (head ~10 min, then the mandatory sync tick before
the destroy), about **$0.55**, which leaves about **$11.2** against the $5.50
floor.
