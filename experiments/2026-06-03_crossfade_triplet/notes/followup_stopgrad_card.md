# Follow-up — L3+nobn+triplet with stop-gradient on the encoder positive term

**Why.** #328's best arm (L3 + no-bottleneck + triplet) reliably beats base at full training.
Test whether stop-gradient on the encoder side of the positive pair changes the learning
dynamics and downstream transfer (target-stop-grad asymmetry, as in SimSiam/BYOL).

**What.** Re-run the #328 best recipe (L3 + no-bottleneck + triplet; keep
`--subtract-contrastive-floor`) with a single change: in the positive term
`sim(h_{t+1}, f_{t+1})`, stop-gradient the encoder term — `sim(stopgrad(h_{t+1}), f_{t+1})`.
Apply it everywhere `h_{t+1}` appears in this positive term: **numerator and denominator**.

**Report.** Downstream GM-Relative MASE vs the reference (#328 L3+nobn+triplet, no stop-grad),
at both head sizes (2L, 6L); plus the log-log training-dynamics curves of both models
(with vs without stop-grad).
