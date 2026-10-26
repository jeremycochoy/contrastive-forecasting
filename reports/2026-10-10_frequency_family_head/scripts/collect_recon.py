"""The R scores of the runs of this report, as the tables of the figure
(plot_green_scores.py): arm, stop in thousands of steps, GM-Relative MASE.

The R queues (queue.sh of 2026-10-06_encoder_reconstruction) write one score
file for each head: score_<arm>_bb<stop>k_h30k_recon.txt for the transformer
head and score_<arm>_bb<stop>k_h30k_recon_lin.txt for the linear head. Give
the result folders of the queues:

    python3 collect_recon.py /home/jupyter/cf_runs/recon_*/results* \
        /home/jupyter/cf_runs/box_heads/recon_results* \
        /home/jupyter/cf_runs/box_heads3/recon_results*

(box_heads holds the R results of the vast box from 10-14. box_heads3
holds the missing R scores, computed on another vast box from 10-20.)

For each table of TABLES, it writes results/recon_<table>_trajectories.tsv
(transformer head) and results/recon_<table>_lin_trajectories.tsv (linear
head). The older runs (table first_steps) give only their score at 1k steps.
"""
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
# arm -> table
TABLES = {"abc_gift": "green", "blk_ema": "blk_ema", "grn_ema095": "grn_ema095",
          "grn_ema098": "grn_ema098", "grn_ema099": "grn_ema099", "grn_ema0999": "grn_ema0999",
          "grn_ema09998": "grn_ema09998", "grn_ema099995": "grn_ema099995",
          "grn_ema099995_f6": "grn_ema099995_f6",
          "grn_ema099995_f6_rep01": "grn_ema099995_f6_rep01",
          "grn_ema09998_rep003_halving": "grn_ema09998_rep003_halving",
          "grn_ema099995_rep003_halving": "grn_ema099995_rep003_halving",
          "blk_emacos": "blk_emacos",
          "cf419_cos200k": "first_steps",
          "k3_r100_09_lr56_fix09_dec10k_lr10x": "first_steps"}
SCORE_FILE = re.compile(r"score_(.+)_bb(\d+)k_h30k_recon(_lin)?\.txt$")


def read_scores(folders):
    """``{(table, "" or "_lin"): {(arm, stop k): score}}`` from the score files."""
    tables = {}
    for path in sorted(p for folder in folders for p in Path(folder).glob("score_*.txt")):
        match = SCORE_FILE.match(path.name)
        if match is None or match[1] not in TABLES:
            continue
        arm, stop, head = match[1], int(match[2]), match[3] or ""
        if TABLES[arm] == "first_steps" and stop != 1:
            continue
        tables.setdefault((TABLES[arm], head), {})[(arm, stop)] = path.read_text().strip()
    return tables


def main():
    for (table, head), scores in read_scores(sys.argv[1:]).items():
        rows = [f"{arm}\t{stop}\t{scores[arm, stop]}\n" for arm, stop in sorted(scores)]
        (RESULTS / f"recon_{table}{head}_trajectories.tsv").write_text("".join(rows))


if __name__ == "__main__":
    main()
