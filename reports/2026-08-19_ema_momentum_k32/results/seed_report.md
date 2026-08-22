**The s08 arm at 4 backbone seeds.** Alpha 0.8 rising to 1.0 at 200000, k = 32, mean reduction, align target teacher, 40000 backbone steps, 30,000 head steps, head seed 20260722, the 97-config eval.

| arm | backbone seed | AUC at 40,000 | GM-Relative MASE | verdict |
|---|---|---|---|---|
| `s08` | 20260520 | 0.957 | 1.1782 | stable |
| `s08b` | 20260521 | 0.575 | 1.5459 | **collapsed** |
| `s08c` | 20260522 | 0.978 | 1.3214 | stable |
| `s08d` | 20260523 | 0.975 | 1.2893 | stable |

**1 of 4 collapsed**, by the AUC at 40000 steps against a line at 0.8. The stable arms of this card hold 0.93 to 0.98 and the collapsed one holds 0.57, so any line inside that band gives the same count.

**The spread over the 3 seeds that did NOT collapse is 0.1432** in absolute terms: `s08`, `s08c`, `s08d` span 1.1782 to 1.3214.

**`a09` 1.1819 and `a095` 1.1907 are 0.0088 apart. The repeat spread is 0.1432. The gap is SMALLER than the spread, so this card does NOT separate the two arms.**
