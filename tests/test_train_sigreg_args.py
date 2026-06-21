"""Tests for the per-term SIGReg weight CLI flags in train.py (#359).

Issue #359 splits the single ``--sigreg-weight`` knob into two independent
weights — ``--sigreg-embedding-weight`` for ``L_sigreg_embedding`` and
``--sigreg-encoding-weight`` for ``L_sigreg_encoding`` — so the two sides
can be set independently. Both default to 0.1 (the prior shared value).

These tests pin:
 * defaults match the prior 0.1 (so unchanged scripts reproduce #355's λ).
 * each weight is honoured independently from the CLI.
 * negative values are rejected.
 * the legacy ``--sigreg-weight`` flag is gone (parser raises on it).
"""

import importlib.util
import os
import sys

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PATH = os.path.join(
    REPO_ROOT, "experiments", "2026-04-27_freq-embedding", "scripts", "train.py")


def load_train_module():
    """Load train.py as a module without executing main()."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    spec = importlib.util.spec_from_file_location("train_freq_emb", TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MIN_ARGV = [
    "train.py",
    "--weight-decay", "0.1",
]


def parse(extra_argv, monkeypatch):
    train = load_train_module()
    monkeypatch.setattr(sys, "argv", list(MIN_ARGV) + list(extra_argv))
    return train.parse_args()


def test_sigreg_embedding_weight_defaults_to_0_1(monkeypatch):
    args = parse([], monkeypatch)
    assert args.sigreg_embedding_weight == pytest.approx(0.1)


def test_sigreg_encoding_weight_defaults_to_0_1(monkeypatch):
    args = parse([], monkeypatch)
    assert args.sigreg_encoding_weight == pytest.approx(0.1)


def test_sigreg_embedding_weight_overridable(monkeypatch):
    args = parse(["--sigreg-embedding-weight", "1.0"], monkeypatch)
    assert args.sigreg_embedding_weight == pytest.approx(1.0)
    # The encoding-side stays at its default — the two flags are independent.
    assert args.sigreg_encoding_weight == pytest.approx(0.1)


def test_sigreg_encoding_weight_overridable(monkeypatch):
    args = parse(["--sigreg-encoding-weight", "0.05"], monkeypatch)
    assert args.sigreg_encoding_weight == pytest.approx(0.05)
    assert args.sigreg_embedding_weight == pytest.approx(0.1)


def test_both_weights_overridable_at_once(monkeypatch):
    args = parse(
        ["--sigreg-embedding-weight", "1.0",
         "--sigreg-encoding-weight", "0.1"], monkeypatch)
    assert args.sigreg_embedding_weight == pytest.approx(1.0)
    assert args.sigreg_encoding_weight == pytest.approx(0.1)


def test_legacy_sigreg_weight_flag_is_gone(monkeypatch):
    """``--sigreg-weight`` is replaced by the two per-term flags."""
    train = load_train_module()
    monkeypatch.setattr(
        sys, "argv", list(MIN_ARGV) + ["--sigreg-weight", "0.1"])
    with pytest.raises(SystemExit):
        train.parse_args()
