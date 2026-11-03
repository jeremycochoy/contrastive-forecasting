"""Compare the losses CSV of a run with the one of a base commit.

Since #440, q_random pairs its rows by a cyclic shift in place of a random
permutation, so the r2_random column of a run differs from the one of an
older commit. The training does not: q_random takes the same draw of the
global random stream as before (torch.randperm), so every other column stays
bit for bit.
"""
import csv
import io
from pathlib import Path

# The columns that the pairing of q_random sets.
Q_RANDOM_COLUMNS = ("r2_random",)


def rows_without_q_random(source):
    """The rows of a losses CSV (a path or its bytes), header first, without
    the columns of q_random."""
    text = (Path(source).read_text() if isinstance(source, (str, Path))
            else source.decode())
    rows = list(csv.reader(io.StringIO(text)))
    drop = {i for i, name in enumerate(rows[0]) if name in Q_RANDOM_COLUMNS}
    return [[v for i, v in enumerate(row) if i not in drop] for row in rows]
