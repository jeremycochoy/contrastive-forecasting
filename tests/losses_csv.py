"""Compare the losses CSV of a run with the one of a base commit.

Two kinds of columns differ from an older commit by design, and
:func:`rows_to_compare` leaves them out:

- Since #440, q_random pairs its rows by a cyclic shift in place of a random
  permutation, so the r2_random column of a run differs from the one of an
  older commit. The training does not: q_random takes the same draw of the
  global random stream as before (torch.randperm), so every other column
  stays bit for bit.
- A run with --train-rollout-depth writes the unweighted copy of each term
  at each depth (``l_pred_d<j>``, ``l_align_d<j>``, ``l_joint_d<j>``) as the
  last columns. An older commit has no such column.
"""
import csv
import io
import re
from pathlib import Path

# The columns that the pairing of q_random sets.
Q_RANDOM_COLUMNS = ("r2_random",)
# The columns of the term copies at each rollout depth.
TERM_DEPTH_COLUMN = re.compile(r"l_(pred|align|joint)_d\d+$")


def rows_to_compare(source):
    """The rows of a losses CSV (a path or its bytes), header first, without
    the columns of q_random and the columns of the term copies."""
    text = (Path(source).read_text() if isinstance(source, (str, Path))
            else source.decode())
    rows = list(csv.reader(io.StringIO(text)))
    drop = {i for i, name in enumerate(rows[0])
            if name in Q_RANDOM_COLUMNS or TERM_DEPTH_COLUMN.match(name)}
    return [[v for i, v in enumerate(row) if i not in drop] for row in rows]
