"""Tests for #412: our contrastive objective with four parts of the Moirai
copy (#421f): the mean/std scaling, the patch sizes 8 to 128, the Moirai
recipe, and the drop of the rows that make a step bad.

Groups, all on the CPU:

1. The contrastive path without the new parts writes the losses CSV of the
   base commit byte for byte.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
BASE_COMMIT = "3d57bf9a"

# The objective of the cyan run (#414 cos200k on the #419 stream), as the
# spec of #412 lists it.
CYAN = (
    "--qk-norm", "--attn-out-norm", "--d-model", "384", "--n-heads", "8",
    "--num-encoder-layers", "3", "--num-layers", "3",
    "--encoder-dropkey", "0.70", "--encoder-dropkey-share-heads",
    "--encoder-dropkey-share-layers", "--depthwise-conv", "3",
    "--deprecated-depthwise-conv", "0",
    "--loss-shape", "cosine_similarity_batch_rep_only",
    "--align-loss-weight", "1.0", "--moco-rep-keys", "--tau-rep", "1.0",
    "--align-target", "teacher", "--ema-embedding", "--ema-encoder",
    "--ema-tau", "0.9", "--cpc-infonce-weight", "0.0", "--sigreg-embedding",
    "--sigreg-encoding", "--sigreg-n-chunk", "2048",
    "--sigreg-embedding-weight", "1.0", "--sigreg-encoding-weight", "1.0",
    "--tau", "0.10", "--encoder-type", "gru", "--synth-kind", "forked-arma",
    "--mix-ratio", "0.0078125", "--crossfade-triplets", "1",
    "--mixup-p", "0.3", "--freq-emb-dim", "3", "--seasonality-emb-dim", "3",
    "--train-rollout-depth", "3", "--train-rollout-reduce", "sum",
    "--rep-loss-weight", "1.0", "--rep-loss-weight-end", "0.0",
    "--rep-loss-weight-ramp-steps", "2500", "--gift-pretrain",
    "--freq-vocab", "v2", "--t-raw", "4096", "--n-channels", "1")

# The Moirai recipe of #421f.
MOIRAI_RECIPE = (
    "--batch-size", "256", "--lr", "1e-3", "--weight-decay", "0.1",
    "--adam-beta1", "0.9", "--adam-beta2", "0.98",
    "--lr-warmup-steps", "10000", "--lr-final", "0",
    "--lr-cosine-steps", "166000", "--grad-clip", "1.0",
    "--residual-dtype", "fp32", "--attn-dtype", "fp32", "--ffn-dtype", "fp32",
    "--conv-dtype", "fp32", "--patch-emb-dtype", "fp32")

# The four Moirai parts #412 adds to the contrastive objective.
MOIRAI_PARTS = (
    "--rev-norm-kind", "meanstd", "--meanstd-z-max", "100",
    "--multi-patch-sizes", "8,16,32,64,128", "--skip-nan-samples",
    "--skip-spike-samples", "10")

# A model the CPU trains in seconds. The last value of a flag wins.
TINY = ("--d-model", "16", "--n-heads", "2", "--num-layers", "1",
        "--num-encoder-layers", "1", "--batch-size", "8", "--log-every", "1",
        "--save-every", "1000000")


def run(root, save_dir, *flags, env=None):
    """One run of the trainer in ``root`` on the CPU."""
    full_env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES="",
                    OMP_NUM_THREADS="4")
    full_env.pop("CF_NAN_DEBUG", None)
    full_env.update(env or {})
    script = (root / "experiments" / "2026-04-27_freq-embedding" / "scripts"
              / "train.py")
    return subprocess.run(
        [sys.executable, str(script), "--device", "cpu", *flags,
         "--save-dir", str(save_dir), "--run-name", "r"],
        capture_output=True, text=True, env=full_env, timeout=1800)


def corpus(root):
    """The #419 test corpus: its flags for the trainer."""
    pytest.importorskip("pyarrow")
    from tests.test_419_gift_pretrain import build_corpus, write_index
    folder, index, _ = build_corpus(root / "corpus")
    return ("--gift-pretrain-root", str(folder),
            "--gift-pretrain-index", str(write_index(root, index)))


def git_tree(commit, tmp_path):
    """The src and the trainer of ``commit``, or a skip without history."""
    if subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e",
                       f"{commit}^{{commit}}"], capture_output=True).returncode:
        pytest.skip(f"{commit} is not in this checkout's history")
    tree = tmp_path / commit
    tree.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "archive", commit, "src",
         "experiments/2026-04-27_freq-embedding/scripts"],
        capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
    return tree


def losses(save_dir):
    return list(csv.DictReader(open(save_dir / "r_losses.csv")))


# ---------------------------------------------------------------------------
# 1. The contrastive path without the new parts is unchanged
# ---------------------------------------------------------------------------

# The cyan objective on the EWMA norm, with mixup at every step and two
# synthetic rows, so every term and every row transform runs.
CYAN_EWMA = CYAN + ("--rev-norm-kind", "ewma", "--rev-norm-span", "128",
                    "--weight-decay", "0.1") + TINY + (
    "--mix-ratio", "0.25", "--mixup-p", "1.0", "--total-steps", "3")


def test_the_cyan_run_writes_the_losses_csv_of_the_base_commit(tmp_path):
    data = corpus(tmp_path)
    old = run(git_tree(BASE_COMMIT, tmp_path), tmp_path / "old", *CYAN_EWMA,
              *data)
    new = run(REPO_ROOT, tmp_path / "new", *CYAN_EWMA, *data)
    assert old.returncode == 0, old.stdout[-2000:] + old.stderr[-2000:]
    assert new.returncode == 0, new.stdout[-2000:] + new.stderr[-2000:]
    assert len(losses(tmp_path / "new")) == 3
    assert ((tmp_path / "new" / "r_losses.csv").read_bytes()
            == (tmp_path / "old" / "r_losses.csv").read_bytes())
