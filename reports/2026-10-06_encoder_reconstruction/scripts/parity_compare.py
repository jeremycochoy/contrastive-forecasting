"""#425: compare the loss CSVs of a head trained in a wave with the CSV of
the same head trained alone.

Usage: python3 parity_compare.py <label> <wave CSV> <solo CSV> [...]
Prints one line for each pair: the steps that both CSVs hold, how many rows
are identical, the first step that differs, and the largest difference.
"""
import csv
import sys


def rows(path):
    with open(path) as f:
        return [(int(r["step"]), float(r["loss"]), int(r["hf_rows_consumed"]))
                for r in csv.DictReader(f)]


def compare(label, wave_csv, solo_csv):
    wave, solo = rows(wave_csv), rows(solo_csv)
    n = min(len(wave), len(solo))
    pairs = list(zip(wave[:n], solo[:n]))
    same = sum(a == b for a, b in pairs)
    first = next((a[0] for a, b in pairs if a != b), None)
    worst = max(abs(a[1] - b[1]) for a, b in pairs)
    return (f"{label}: {n} steps, {same} identical rows, first difference "
            f"at step {first}, max |loss diff| {worst:.3g}, loss at step "
            f"{pairs[-1][0][0]}: {pairs[-1][0][1]:.6f} (wave) "
            f"{pairs[-1][1][1]:.6f} (solo)")


def main():
    args = sys.argv[1:]
    for i in range(0, len(args), 3):
        print(compare(*args[i:i + 3]))


if __name__ == "__main__":
    main()
