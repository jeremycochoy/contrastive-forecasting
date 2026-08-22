**arm6_v2 (L_rep MoCo keys, tau_rep 1 + L_align on the student, no CPC, EMA 0.9 to 1.0).** At bb100k, on the student-encoder head [A4]. The cell that sets this study's frontier, at the deepest stop its parent published. Published `k = 0`.

| family | configs | rollout steps | k = 0 | k = 3 | difference | where k = 3 leaves it |
|---|---:|---:|---:|---:|---:|---|
| Energy ⚑ | 32 | 16.5 (1–45) | 1.481 | 1.279 | -0.201 | toward 1.0 |
| Web/CloudOps ⚑ | 20 | 30 (3–57) | 1.257 | 1.199 | -0.057 | toward 1.0 |
| Nature | 15 | 3 (1–45) | 0.866 | 0.822 | -0.044 | stays below 1.0, lower |
| Transport | 15 | 30 (2–45) | 1.021 | 0.901 | -0.120 | **past 1.0** |
| Econ/Fin ⚑ | 6 | 1 (1–3) | 1.414 | 1.150 | -0.263 | toward 1.0 |
| Healthcare ⚑ | 5 | 1 (1–2) | 1.171 | 1.113 | -0.058 | toward 1.0 |
| Sales | 4 | 1.5 (1–2) | 0.800 | 0.797 | -0.003 | stays below 1.0, lower |


**arm6_v2 (L_rep MoCo keys, tau_rep 1 + L_align on the student, no CPC, EMA 0.9).** At bb40k, on the student-encoder head [B1]. The pair whose `k = 0` side this study trained, so the depth is the only change.

| family | configs | rollout steps | k = 0 | k = 3 | difference | where k = 3 leaves it |
|---|---:|---:|---:|---:|---:|---|
| Energy ⚑ | 32 | 16.5 (1–45) | 1.471 | 1.270 | -0.200 | toward 1.0 |
| Web/CloudOps ⚑ | 20 | 30 (3–57) | 1.288 | 1.211 | -0.077 | toward 1.0 |
| Nature | 15 | 3 (1–45) | 0.884 | 0.840 | -0.044 | stays below 1.0, lower |
| Transport | 15 | 30 (2–45) | 1.040 | 0.907 | -0.133 | **past 1.0** |
| Econ/Fin ⚑ | 6 | 1 (1–3) | 1.466 | 1.212 | -0.254 | toward 1.0 |
| Healthcare ⚑ | 5 | 1 (1–2) | 1.103 | 1.077 | -0.026 | toward 1.0 |
| Sales | 4 | 1.5 (1–2) | 0.772 | 0.775 | +0.004 | stays below 1.0, higher |


**arm6_v2 (L_rep MoCo keys, tau_rep 1 + L_align on the teacher, no CPC, EMA 0.9).** At bb200k, on the student-encoder head [B2]. The arm and stop the card quotes its own per-family numbers from. Published `k = 0`.

| family | configs | rollout steps | k = 0 | k = 3 | difference | where k = 3 leaves it |
|---|---:|---:|---:|---:|---:|---|
| Energy ⚑ | 32 | 16.5 (1–45) | 1.388 | 1.587 | +0.198 | away from 1.0 |
| Web/CloudOps ⚑ | 20 | 30 (3–57) | 1.283 | 1.347 | +0.064 | away from 1.0 |
| Nature | 15 | 3 (1–45) | 0.867 | 0.914 | +0.047 | stays below 1.0, higher |
| Transport | 15 | 30 (2–45) | 1.021 | 1.077 | +0.056 | away from 1.0 |
| Econ/Fin ⚑ | 6 | 1 (1–3) | 1.489 | 1.869 | +0.380 | away from 1.0 |
| Healthcare ⚑ | 5 | 1 (1–2) | 1.261 | 1.283 | +0.022 | away from 1.0 |
| Sales | 4 | 1.5 (1–2) | 0.830 | 0.824 | -0.006 | stays below 1.0, lower |


⚑ marks the four families the card names as the ones seasonal naive wins by the largest margin: Energy, Econ/Fin, Web/CloudOps, Healthcare.

`rollout steps` is how many times the eval runs `rollout_latent` on a config of that family, median and range. It is the same column for every table here, because it depends on the config and not on the run.

