"""freq_family: compare a family of one member with the control of its wave,
and compare two eval tables.

A family that sends each row and each config to one member is the standard
head. So in one wave, such a family must write the loss rows of the control,
end with its weights, and get its score.

Usage:
    python3 compare_parity.py losses <label> <family CSV> <control CSV>
    python3 compare_parity.py heads <label> <family head> <control head>
    python3 compare_parity.py tables <label> <table> <second table>

``losses``: the steps that the two loss CSVs hold, the rows that are
identical, and the largest loss difference.
``heads``: the tensors of the control head, and how many of them the member
16 of the family head holds bit for bit.
``tables``: the configs of two `all_results.csv` of the eval, how many have
the same MASE, the largest relative difference, and the ratio of the two
GM-Relative MASE (the seasonal-naive terms cancel).
"""

from __future__ import annotations

import csv
import math
import sys

MASE = "eval_metrics/MASE[0.5]"
# Two MASE values of one config are the same under this relative difference.
SAME_MASE = 1e-6


def loss_rows(path):
    with open(path) as f:
        return [tuple(row) for row in csv.reader(f)][1:]


def compare_losses(label, family_csv, control_csv):
    family, control = loss_rows(family_csv), loss_rows(control_csv)
    pairs = list(zip(family, control))
    same = sum(a == b for a, b in pairs)
    worst = max(abs(float(a[1]) - float(b[1])) for a, b in pairs)
    return (f"{label}: {len(family)} and {len(control)} loss rows, {same} "
            f"identical, max |loss difference| {worst:.3g}")


def compare_heads(label, family_head, control_head):
    import torch
    from src.freq_family import family_member_state
    family, control = (torch.load(path, map_location="cpu", weights_only=True)
                       for path in (family_head, control_head))
    member = family_member_state(family, 16)
    same = sum(key in member and torch.equal(member[key], control[key])
               for key in control)
    worst = max(float((member[key] - control[key]).abs().max())
                for key in control if key in member)
    return (f"{label}: {len(control)} tensors in the control head, {same} "
            f"identical in the member 16, max |weight difference| "
            f"{worst:.3g}")


def config_mase(path):
    with open(path) as f:
        return {row["dataset"]: float(row[MASE]) for row in csv.DictReader(f)}


def compare_tables(label, table, second):
    mase, other = config_mase(table), config_mase(second)
    shared = sorted(set(mase) & set(other))
    diffs = [abs(mase[c] - other[c]) / abs(other[c]) for c in shared]
    ratio = math.exp(sum(math.log(mase[c] / other[c]) for c in shared)
                     / len(shared))
    return (f"{label}: {len(mase)} and {len(other)} configs, "
            f"{sum(d < SAME_MASE for d in diffs)} with the same MASE, max "
            f"relative difference {max(diffs):.2g}, ratio of the two "
            f"GM-Relative MASE {ratio:.6f}")


MODES = {"losses": compare_losses, "heads": compare_heads,
         "tables": compare_tables}


def main(argv=None):
    mode, *args = sys.argv[1:] if argv is None else argv
    for i in range(0, len(args), 3):
        print(MODES[mode](*args[i:i + 3]))


if __name__ == "__main__":
    main()
