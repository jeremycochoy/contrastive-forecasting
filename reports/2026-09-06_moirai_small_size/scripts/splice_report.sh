#!/bin/bash
# #412 — put the arms table and the results tables INTO the report.
#
# WHY. The report standard asks for the tables at the back of the one canonical
# report. Those two sections held placeholders, `_(from scripts/arms.tsv)_` and
# `_(from results/tables.md)_`, so a reader of the report alone saw neither.
#
# The tables change while the card runs, so they cannot be pasted by hand and
# stay true. This splices them between markers on every heartbeat tick, and it
# rewrites only the text between the markers.
#
# Usage:  bash scripts/splice_report.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
REPORT="$CF412_STUDY/moirai_small_size.md"
[ -f "$REPORT" ] || { echo "ABORT: no report at $REPORT" >&2; exit 2; }

python3 - "$REPORT" "$CF412_SCRIPTS/arms.tsv" "$CF412_RESULTS/tables.md" <<'PY'
import sys, pathlib
report, arms_tsv, tables_md = (pathlib.Path(a) for a in sys.argv[1:4])
s = report.read_text()

def splice(text, heading, body):
    """Replace everything between `heading` and the next `## ` with `body`."""
    i = text.index(heading)
    j = text.find("\n## ", i + len(heading))
    j = len(text) if j < 0 else j
    return text[:i] + heading + "\n\n" + body.rstrip() + "\n" + text[j:]

# The arms, as a table a reader can scan. The tsv comments hold the design and
# they stay in the file, not in the report.
rows = []
for line in arms_tsv.read_text().splitlines():
    if line.startswith("#") or not line.strip():
        continue
    rows.append(line.split("\t"))
# EVERY non-comment row is an arm. The tsv's own header line starts with `#`,
# so it is already gone by here. An earlier version took rows[0] as a header
# and silently dropped `k3_r100_09`, the arm the headline rests on.
body = rows
cols = ["arm", "k", "reduce", "EMA start", "EMA end", "ramp", "seed",
        "L_rep decay", "lr"]
tbl = ["| " + " | ".join(cols) + " |",
       "|" + "|".join("---" for _ in cols) + "|"]
for r in body:
    r = r + ["-"] * (len(cols) - len(r))
    tbl.append("| " + " | ".join(r[:len(cols)]) + " |")
arms_block = ("Ten arms. Every one is a PUBLISHED configuration at the new "
              "width, so its depth, reduction, momentum, seed and decay are "
              "its parent's values. `scripts/arms.tsv` carries the reasoning "
              "for each row.\n\n" + "\n".join(tbl))

s = splice(s, "## The arms", arms_block)
s = splice(s, "## The tables", tables_md.read_text().strip())
tmp = report.with_suffix(".md.tmp")
tmp.write_text(s)
tmp.replace(report)
print(f"spliced {len(body)} arm(s) and the results tables into {report.name}")
PY
