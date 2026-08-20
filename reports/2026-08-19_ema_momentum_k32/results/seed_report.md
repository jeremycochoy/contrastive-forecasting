**The s08 arm at 2 backbone seeds.** Alpha 0.8 rising to 1.0 at 200000, k = 32, mean reduction, align target teacher, 40000 backbone steps, 30,000 head steps, head seed 20260722, the 97-config eval.

| arm | backbone seed | AUC at 40,000 | GM-Relative MASE | verdict |
|---|---|---|---|---|
| `s08` | 20260520 | 0.957 | 1.1782 | stable |
| `s08b` | 20260521 | 0.575 | 1.5459 | **collapsed** |

**1 of 2 collapsed**, by the AUC at 40000 steps against a line at 0.8. The stable arms of this card hold 0.93 to 0.98 and the collapsed one holds 0.57, so any line inside that band gives the same count.

Fewer than two seeds survived, so this round measures no spread.
