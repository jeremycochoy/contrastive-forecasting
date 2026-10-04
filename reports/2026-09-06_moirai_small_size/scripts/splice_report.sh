#!/bin/bash
# #412 — the splice is retired.
#
# WHY. While pass 1 ran, this script rewrote the report's arms and tables
# sections from `scripts/arms.tsv` and `results/tables.md` on every heartbeat
# tick, so a reader never saw a stale table. The report review then fixed the
# structure and the wording of those sections by hand, and a re-splice would
# undo the review. Pass-2 rows land through the follow-up card, not through
# this splice.
#
# The heartbeat still calls this script, so it must exit 0 and do nothing.
echo "splice retired: the report is review-final, pass-2 rows land via the follow-up card"
exit 0
