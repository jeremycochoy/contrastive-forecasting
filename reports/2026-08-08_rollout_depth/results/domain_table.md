**A4  arm6_v2 combab · L_align on the student, bb100k, student-encoder head.** The cell that sets this study's frontier, at the deepest stop its parent published. Two machines.

| family | configs | k = 0 | k = 3 | difference | where k = 3 leaves it |
|---|---:|---:|---:|---:|---|
| Energy ⚑ | 32 | 1.481 | 1.279 | -0.201 | toward 1.0 |
| Web/CloudOps ⚑ | 20 | 1.257 | 1.199 | -0.057 | toward 1.0 |
| Nature | 15 | 0.866 | 0.822 | -0.044 | stays below 1.0, lower |
| Transport | 15 | 1.021 | 0.901 | -0.120 | **past 1.0** |
| Econ/Fin ⚑ | 6 | 1.414 | 1.150 | -0.263 | toward 1.0 |
| Healthcare ⚑ | 5 | 1.171 | 1.113 | -0.058 | toward 1.0 |
| Sales | 4 | 0.800 | 0.797 | -0.003 | stays below 1.0, lower |


**B1  arm6_v2 combab · L_align on the student, bb40k, student-encoder head.** The pair whose two sides trained on ONE machine, so the depth is the only change.

| family | configs | k = 0 | k = 3 | difference | where k = 3 leaves it |
|---|---:|---:|---:|---:|---|
| Energy ⚑ | 32 | 1.471 | 1.270 | -0.200 | toward 1.0 |
| Web/CloudOps ⚑ | 20 | 1.288 | 1.211 | -0.077 | toward 1.0 |
| Nature | 15 | 0.884 | 0.840 | -0.044 | stays below 1.0, lower |
| Transport | 15 | 1.040 | 0.907 | -0.133 | **past 1.0** |
| Econ/Fin ⚑ | 6 | 1.466 | 1.212 | -0.254 | toward 1.0 |
| Healthcare ⚑ | 5 | 1.103 | 1.077 | -0.026 | toward 1.0 |
| Sales | 4 | 0.772 | 0.775 | +0.004 | stays below 1.0, higher |


**B2  arm6_v2 combab · L_align on the teacher, bb200k, student-encoder head.** The arm and stop the card quotes its own per-family numbers from. Two machines.

| family | configs | k = 0 | k = 3 | difference | where k = 3 leaves it |
|---|---:|---:|---:|---:|---|
| Energy ⚑ | 32 | 1.388 | 1.587 | +0.198 | away from 1.0 |
| Web/CloudOps ⚑ | 20 | 1.283 | 1.347 | +0.064 | away from 1.0 |
| Nature | 15 | 0.867 | 0.914 | +0.047 | stays below 1.0, higher |
| Transport | 15 | 1.021 | 1.077 | +0.056 | away from 1.0 |
| Econ/Fin ⚑ | 6 | 1.489 | 1.869 | +0.380 | away from 1.0 |
| Healthcare ⚑ | 5 | 1.261 | 1.283 | +0.022 | away from 1.0 |
| Sales | 4 | 0.830 | 0.824 | -0.006 | stays below 1.0, lower |


⚑ marks the four families the card names as the ones seasonal naive wins by the largest margin: Energy, Econ/Fin, Web/CloudOps, Healthcare.

