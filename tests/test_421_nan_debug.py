"""Tests for the NaN diagnostic mode of the trainer (#421): CF_NAN_DEBUG=1.

1. Off (the default), the mode registers nothing and records nothing, and a
   run writes the losses CSV of the commit before the mode, byte for byte.
2. On, it observes only: a run without a NaN writes the same losses CSV.
3. On, an injected NaN loss, or a NaN gradient behind a finite loss, stops
   the run before the step with a dump that holds the batch, the loss terms,
   the module maxima and the names of the non-finite gradients.
4. On, a weight that the optimizer step makes non-finite stops the run too.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import nan_debug as nd  # noqa: E402
from src.nan_debug import PROBE, NanDebug  # noqa: E402

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
EXIT = nd.EXIT_CODE

# The #421z cell at a size the CPU trains in seconds, with the run's
# attention flags so the probes inside the attention fire too.
TINY_RUN = (
    "--value-space-objective", "--gift-pretrain", "--freq-vocab", "v2",
    "--multi-patch-sizes", "8,16,32,64,128", "--rev-norm-kind", "meanstd",
    "--t-raw", "4096", "--n-channels", "1", "--d-model", "16",
    "--n-heads", "2", "--num-layers", "1", "--num-encoder-layers", "1",
    "--batch-size", "8", "--synth-kind", "forked-arma",
    "--mix-ratio", "0.25", "--crossfade-triplets", "1", "--mixup-p", "1.0",
    "--freq-emb-dim", "3", "--seasonality-emb-dim", "3",
    "--qk-norm", "--attn-out-norm", "--encoder-dropkey", "0.70",
    "--encoder-dropkey-share-heads", "--encoder-dropkey-share-layers",
    "--attn-dtype", "fp16", "--ffn-dtype", "fp16", "--conv-dtype", "fp16",
    "--train-rollout-depth", "3", "--train-rollout-reduce", "sum",
    "--lr", "1e-3", "--lr-final", "0", "--lr-cosine-steps", "166000",
    "--lr-warmup-steps", "10000", "--grad-clip", "1.0",
    "--log-every", "1", "--save-every", "1000000", "--total-steps", "3")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    pytest.importorskip("pyarrow")
    from tests.test_419_gift_pretrain import build_corpus, write_index
    root = tmp_path_factory.mktemp("gep_nan")
    corpus_dir, index, _ = build_corpus(root / "corpus")
    return ("--gift-pretrain-root", str(corpus_dir),
            "--gift-pretrain-index", str(write_index(root, index)))


def run(root, save_dir, corpus, env=None, flags=TINY_RUN):
    """One tiny run of the trainer in ``root``: (result, losses CSV path)."""
    full_env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES="",
                    OMP_NUM_THREADS="4")
    for key in ("CF_NAN_DEBUG", "CF_NAN_DEBUG_DUMP", "CF_NAN_DEBUG_INJECT"):
        full_env.pop(key, None)
    full_env.update(env or {})
    script = (root / "experiments" / "2026-04-27_freq-embedding" / "scripts"
              / "train.py")
    result = subprocess.run(
        [sys.executable, str(script), "--device", "cpu", "--weight-decay",
         "0.1", *flags, *corpus, "--save-dir", str(save_dir),
         "--run-name", "r"],
        capture_output=True, text=True, env=full_env, timeout=900)
    return result, save_dir / "r_losses.csv"


def debug_env(dump, inject=None):
    env = {"CF_NAN_DEBUG": "1", "CF_NAN_DEBUG_DUMP": str(dump)}
    if inject:
        env["CF_NAN_DEBUG_INJECT"] = inject
    return env


def tiny_model():
    from src.models import ConfigurableModel
    torch.manual_seed(0)
    return ConfigurableModel(
        C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=2.0, dropout=0.0, rev_norm_kind="meanstd",
        num_encoder_layers=1, value_head_quantiles=9,
        enc_transformer_use_grad_checkpoint=False)


# ---------------------------------------------------------------------------
# 1. Off
# ---------------------------------------------------------------------------

def test_off_by_default_nothing_registers(monkeypatch):
    monkeypatch.delenv("CF_NAN_DEBUG", raising=False)
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    assert NanDebug.from_env(model, optimizer, "unused.pt") is None
    assert not PROBE.active
    assert all(not m._forward_hooks for m in model.modules())


def test_the_probe_records_nothing_while_inactive():
    PROBE.reset()
    t = torch.ones(3, requires_grad=True) * 2.0
    PROBE.record("x", t)
    PROBE.note("y", 1.0)
    assert PROBE.forward == PROBE.backward == PROBE.notes == []
    assert not t._backward_hooks


def git_tree(commit, tmp_path):
    """The src and trainer of ``commit``, or a skip without git history."""
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


def test_off_the_run_writes_the_csv_of_the_commit_before_the_mode(
        corpus, tmp_path):
    """The probe calls in the attention and in the rollout change nothing:
    with the mode off, the losses CSV equals the one of 4b521ea8."""
    old_root = git_tree("4b521ea8", tmp_path)
    old, old_csv = run(old_root, tmp_path / "old", corpus)
    new, new_csv = run(REPO_ROOT, tmp_path / "new", corpus)
    assert old.returncode == 0, old.stdout[-2000:] + old.stderr[-2000:]
    assert new.returncode == 0, new.stdout[-2000:] + new.stderr[-2000:]
    assert "NaN debug" not in new.stdout
    assert new_csv.read_bytes() == old_csv.read_bytes()


# ---------------------------------------------------------------------------
# 2. On, with no NaN
# ---------------------------------------------------------------------------

def test_on_the_mode_observes_only(corpus, tmp_path):
    off, off_csv = run(REPO_ROOT, tmp_path / "off", corpus)
    on, on_csv = run(REPO_ROOT, tmp_path / "on", corpus,
                     env=debug_env(tmp_path / "dump.pt"))
    assert off.returncode == 0 and on.returncode == 0, on.stderr[-2000:]
    assert "NaN debug (CF_NAN_DEBUG=1)" in on.stdout
    assert on_csv.read_bytes() == off_csv.read_bytes()
    assert not (tmp_path / "dump.pt").exists()


# ---------------------------------------------------------------------------
# 3. On, with an injected NaN
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grad_dump(corpus, tmp_path_factory):
    """A run whose step 2 has a finite loss and a NaN gradient."""
    root = tmp_path_factory.mktemp("nan_grad")
    result, losses = run(REPO_ROOT, root / "save", corpus,
                         env=debug_env(root / "dump.pt", "grad:2"))
    return result, losses, root / "dump.pt"


def test_a_nan_gradient_stops_the_run_before_the_step(grad_dump):
    result, losses, dump = grad_dump
    assert result.returncode == EXIT, result.stdout[-3000:] + result.stderr
    assert "*** CF_NAN_DEBUG: non-finite gradient at step 2" in result.stdout
    rows = list(csv.DictReader(open(losses)))
    assert [row["step"] for row in rows] == ["1"]
    data = torch.load(dump, weights_only=False)
    assert data["reason"] == "gradient" and data["step"] == 2
    assert torch.isfinite(torch.tensor(data["loss"]))
    assert data["nonfinite_grads"] == ["value_heads.8.weight"]
    assert data["nonfinite_weights"] == []
    weights = data["weights_before_step"]
    assert all(torch.isfinite(w).all() for w in weights.values())


def test_the_dump_holds_the_batch_after_every_transform(grad_dump):
    batch = torch.load(grad_dump[2], weights_only=False)["batch"]
    B, T, C = batch["values"].shape
    assert (B, T, C) == (8 + 3, 1024, 1)
    for key in ("x_norm", "padding", "loc", "scale", "loaded"):
        assert batch[key].shape[0] == B
    for key in ("patch_size", "max_z", "freq_ids", "seasonality_ids", "kept"):
        assert batch[key].shape == (B,)
    assert batch["split"].shape == (B, C)
    assert batch["sign_flipped"].shape == (B, C)
    assert batch["row_kind"] == (["real"] * 6 + ["forked-arma"] * 2
                                 + ["triplet A", "triplet B", "triplet C"])
    assert set(batch["mixup"]) == {"weight", "partner"}
    assert sorted(batch["mixup"]["partner"].tolist()) == list(range(B))


def test_the_dump_holds_the_terms_and_the_maxima(grad_dump):
    data = torch.load(grad_dump[2], weights_only=False)
    notes = dict(data["notes"])
    sizes = sorted(set(data["batch"]["patch_size"].tolist()))
    for size in sizes:
        assert f"group P{size}" in notes
        assert all(f"loss P{size} depth {j}" in notes for j in range(4))
    tags = [tag for tag, _ in data["forward"]]
    for part in ("attention qkv", "attention sdpa", "attention out_proj",
                 "linear1", "linear2", "depthwise_conv", "value_heads",
                 "rollout feedback", "encoder.encoders"):
        assert any(part in tag for tag in tags), part
    assert data["first_nonfinite_forward"] is None
    assert len(data["backward"]) > 0
    assert any("rollout feedback" in tag for tag, _ in data["backward"])


def test_a_nan_loss_stops_the_run_before_the_backward(corpus, tmp_path):
    result, losses = run(REPO_ROOT, tmp_path / "save", corpus,
                         env=debug_env(tmp_path / "dump.pt", "loss:2"))
    assert result.returncode == EXIT, result.stdout[-3000:] + result.stderr
    data = torch.load(tmp_path / "dump.pt", weights_only=False)
    assert data["reason"] == "loss" and data["backward"] == []
    assert "NaN/Inf DETECTED" not in result.stdout
    assert not list((tmp_path / "save").glob("*EMERGENCY*"))


# ---------------------------------------------------------------------------
# 4. A weight that the step makes non-finite
# ---------------------------------------------------------------------------

def test_a_weight_the_step_makes_non_finite_is_caught(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    debug = NanDebug(model, optimizer, str(tmp_path / "dump.pt"))
    try:
        debug.start_step(7)
        model.value_head.weight.sum().backward()
        assert not debug.grads_are_bad(1.0)
        with torch.no_grad():
            model.value_head.weight[0, 0] = float("nan")
        assert debug.weights_are_bad(1.0, grad_norm=0.5)
    finally:
        for handle in debug.handles:
            handle.remove()
        PROBE.active = False
    data = torch.load(tmp_path / "dump.pt", weights_only=False)
    assert data["reason"] == "weight after the optimizer step"
    assert data["nonfinite_weights"] == ["value_head.weight"]
    before = data["weights_before_step"]["value_head.weight"]
    assert torch.isfinite(before).all() and data["grad_norm"] == 0.5


def test_the_inject_spec_is_checked():
    assert nd._parse_inject("grad:15393") == ("grad", 15393)
    assert nd._parse_inject(None) is None
    with pytest.raises(ValueError):
        nd._parse_inject("weight:3")
