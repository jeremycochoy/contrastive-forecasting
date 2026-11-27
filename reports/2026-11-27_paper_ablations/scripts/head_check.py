"""CPU check of the score path of the ablation runs of #482: the head trainer
of the score code trains a few steps on the backbone of each row.

The backbone of a row is the last checkpoint of its CPU check (cpu_check.py
rows). The flags are those of a head job of the follower of #425 (the control
head of the reference, head_eval_bb.sh), with only these changes: the CPU, a
small batch, a few steps, a save folder in --out, and the local corpus of the
#419 tests in place of the Hub. Rows 6 and 8 take the patched score code
(make_head_code.sh), the other rows the score code of #425.

Usage:
    python3 head_check.py --rows-dir <dir> --out <dir> [--patched <ff_code_sha>]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

SCORE_CODE = "/home/jupyter/cf_runs/hp64/ff/code"
PATCHED_ROWS = ("_ab06_", "_ab08_")
JOB = ["--encoder-source", "student", "--device", "cpu", "--quantile-head",
       "--grad-clip", "1.0", "--forecast-len", "16", "--batch-size", "8",
       "--lr", "1e-3", "--total-steps", "3", "--save-every", "1000000",
       "--log-every", "1", "--seed", "20260722",
       "--hf-repo", "jeremycochoy/gift-pretrain-full-4096", "--hf-path", "small_v1",
       "--head-arch", "transformer", "--head-num-layers", "2", "--head-nhead", "8",
       "--head-ffn-mult", "4.0", "--head-causal", "true",
       "--head-train-input", "e_then_f", "--head-dropout", "0.1",
       "--t-raw", "4096", "--n-channels", "1", "--d-model", "320",
       "--n-heads", "8", "--num-layers", "6", "--encoder-type", "gru",
       "--rev-norm-kind", "ewma", "--rev-norm-span", "128",
       "--freq-emb-dim", "3", "--seasonality-emb-dim", "3"]


def head_run(code, backbone, corpus, save):
    """Three head steps on one backbone. Returns (exit code, log path)."""
    save.mkdir(parents=True, exist_ok=True)
    script = Path(code) / "experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
    argv = [sys.executable, "-u", str(script), "--backbone-path", str(backbone), *JOB,
            "--save-dir", str(save), "--run-name", "qhead",
            "--gift-pretrain-root", str(corpus / "corpus"),
            "--gift-pretrain-index", str(corpus / "index.json.gz")]
    env = dict(os.environ, PYTHONPATH=str(code), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="8")
    log = save / "head.log"
    with open(log, "w") as f:
        rc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=env,
                            cwd=str(code), timeout=3600).returncode
    return rc, log


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rows-dir", type=Path, nargs="+", required=True,
                   help="the --out folders of cpu_check.py rows")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--patched", default=None, help="the patched score code")
    a = p.parse_args()
    failed = []
    for rows_dir in a.rows_dir:
        for backbone in sorted(rows_dir.glob("*/*_final.pth")):
            run = backbone.parent.name
            code = a.patched if any(r in run for r in PATCHED_ROWS) else SCORE_CODE
            if code is None:
                print(f"{run}: SKIP, needs --patched")
                failed.append(run)
                continue
            rc, log = head_run(code, backbone, rows_dir, a.out / run)
            built = [line.strip() for line in open(log) if "the checkpoint names" in line]
            print(f"{run}: rc {rc}, score code {Path(code).name}"
                  + (f", {built[0]}" if built else ""), flush=True)
            if rc:
                failed.append(run)
    print(f"heads: failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
