"""One line per row of the CPU row check of #482 (cpu_check.py rows): its
exit code and a sign that its flag acts, read from its log, its losses CSV
and its last checkpoint.

Usage: python3 row_effects.py <rows dir> ... > results/rows_cpu.txt
"""
import csv
import re
import sys
from pathlib import Path

import torch


def last_csv_row(folder):
    with open(next(folder.glob("*_losses.csv"))) as f:
        return list(csv.DictReader(f))[-1]


def effect(run, folder):
    """The sign that the flag of the row acts."""
    log = (folder / "train.log").read_text()
    row = last_csv_row(folder)
    sd = torch.load(next(folder.glob("*_final.pth")), map_location="cpu", weights_only=True)
    data = re.search(r"hf_bs=\d+, synth_bs=\d+, cross_bs=\d+, total_bs=\d+", log).group(0)
    tables = ("freq_embedding.embedding.weight", "seasonality_embedding.embedding.weight")
    signs = {  # each value is computed only for its own row
        "_ab01_": lambda: data, "_ab02_": lambda: data,
        "_ab03_": lambda: re.search(r"mixup_p=[\d.]+", log).group(0),
        "_ab04_": lambda: "Scaling (#482) line: " + str("Scaling (#482)" in log)
                          + ", no-target rule: "
                          + str("No term reads a window with no target" in log),
        "_ab05_": lambda: "tables all 0: " + str(all(
            int(torch.count_nonzero(sd[k])) == 0 for k in tables)),
        "_ab06_": lambda: "encoder.residual_layer.weight "
                          + str(tuple(sd["encoder.residual_layer.weight"].shape))
                          + ", encoder.layer_norm keys: "
                          + str(sum(k.startswith("encoder.layer_norm.") for k in sd)),
        "_ab07_": lambda: "sphere buffer: " + str("transformer.latent_sphere" in sd),
        "_ab08_": lambda: "conv weights: " + str(sum("depthwise_conv" in k for k in sd)),
        "_ab09_": lambda: re.search(r"--encoder-dropkey [\d.]+", log).group(0),
        "_ab10_": lambda: re.search(r"--dropout [\d.]+", log).group(0),
        "_ab11_": lambda: f"ema_tau {row['ema_tau']}",
        "_ab12_": lambda: f"sigreg_e '{row['sigreg_e']}', sigreg_h {row['sigreg_h']}",
        "_ab13_": lambda: f"sigreg_e {row['sigreg_e']}, sigreg_h '{row['sigreg_h']}'",
        "_ab14_": lambda: f"rep_w {row['rep_w']}",
        "_ab15_": lambda: "rollout columns: " + str([k for k in row if k.startswith("cos_err_d")]),
    }
    return next(v() for k, v in signs.items() if k in run), row["loss"]


def main():
    for rows_dir in map(Path, sys.argv[1:]):
        for log in sorted(rows_dir.glob("*/train.log")):
            folder, run = log.parent, log.parent.name
            done = list(folder.glob("*_final.pth")) and list(folder.glob("*_losses.csv"))
            if not done:
                print(f"{run}: no final checkpoint")
                continue
            sign, loss = effect(run, folder)
            print(f"{run}: 2 steps, final loss {float(loss):.4f}, {sign}")


if __name__ == "__main__":
    main()
