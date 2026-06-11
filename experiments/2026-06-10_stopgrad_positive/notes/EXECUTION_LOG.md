# Execution log (ops; kept out of the report)

- 2026-06-10: elisa 200-step false start (user redirected training to vast.ai;
  artifacts deleted, run restarted clean on vast). elisa = prep/sync/analysis only.
- 2026-06-10 13:36 UTC: backbone launched on vast.ai 40410773 (cf-sgpos-339,
  on-demand datacenter 4090, $0.4681/h, GB; Xeon Gold 6133). 12.5k steps in
  ~16.4h (0.21 sps; elisa 4090 did ~0.24 sps on the reference arm).
- 2026-06-11: GIFT-Eval on this host is CPU-single-thread-bound (GPU ~7%,
  ~1.8x slower than elisa per task → ~6.5h per full-97 eval). Attempted a
  second instance to parallelize the 6L stage: offer 40365283 (machine 17164)
  preempted before boot ($0); user declined further provisioning → serial.
- Monitoring cadence tightened to ~45 min after user feedback (another
  session's instance idled ~19h ≈ $30): every check = process alive +
  progress counter moving + GPU util, auto-restart idempotent stage if dead.
