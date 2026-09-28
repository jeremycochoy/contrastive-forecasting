"""Tests for #421: --skip-nan-samples drops only the rows of a NaN step.

A value-space step reads each row on its own. When a step goes non-finite,
the trainer finds the rows that cause it by bisection, makes them inert, and
takes the step on the other rows. Five groups, all on the CPU:

1. The bisection finds one or two culprits, and gives up on a fault that
   needs two rows at once or on a pass budget.
2. The step without the culprits: one poisonous row, then two, are dropped,
   and the loss and the gradients equal those of the batch without them. A
   clean batch runs no bisection. A fault that no row explains skips the
   step.
3. Each pass draws the same dropout and DropKey masks for each row.
4. The trainer: an end-to-end run on a stream with a poisoned source goes on
   and logs the dropped rows. With a clean stream the flag changes nothing,
   and with the flag off the losses CSV equals the one of the commit before.
5. The refusals.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.forecasting_head as fh  # noqa: E402
from src.forecasting_head import mean_std_inputs  # noqa: E402
from src.freq_embedding import FREQ_NAMES_V2  # noqa: E402
from src.models import ConfigurableModel  # noqa: E402
from src.nan_skip import (find_culprits, is_finite_step,  # noqa: E402
                          restore_rng, rng_state, weights_are_finite)

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
SIZES = (8, 16, 32, 64, 128)
T = 1024
CPU = torch.device("cpu")
ARGS = SimpleNamespace(train_rollout_depth=2, train_rollout_reduce="sum")


@pytest.fixture(scope="module")
def train_py():
    spec = importlib.util.spec_from_file_location("train_421_skip", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. The bisection
# ---------------------------------------------------------------------------

def bad_rows(culprits):
    """A pass is bad when it holds a culprit."""
    return lambda rows: bool(set(rows) & set(culprits))


@pytest.mark.parametrize("culprits", [[5], [0], [31], [3, 17], [8, 9]])
def test_the_bisection_finds_the_culprits(culprits):
    found, passes = find_culprits(bad_rows(culprits), range(32))
    assert found == culprits
    assert passes <= 2 * 5 * len(culprits)


def test_a_fault_that_needs_two_rows_at_once_finds_no_culprit():
    def both(rows):
        return 3 in rows and 20 in rows
    assert find_culprits(both, range(32))[0] is None


def test_non_finite_weights_are_seen():
    """With a non-finite weight no row to drop can help, and the trainer
    stops as before."""
    m = torch.nn.Linear(3, 2)
    assert weights_are_finite(m)
    with torch.no_grad():
        m.weight[0, 0] = float("inf")
    assert not weights_are_finite(m)


def test_the_bisection_stops_at_its_pass_budget():
    found, passes = find_culprits(bad_rows(range(32)), range(32),
                                  max_passes=10)
    assert found is None and passes == 10


# ---------------------------------------------------------------------------
# 2. The step without the culprits
# ---------------------------------------------------------------------------

def model(dropout=0.0, dropkey=0.0):
    torch.manual_seed(0)
    return ConfigurableModel(
        C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=2.0, dropout=dropout, rev_norm_kind="meanstd",
        rev_norm_skip_leading_zeros=True, num_encoder_layers=1,
        encoder_dropkey=dropkey, freq_emb_dim=3, num_freqs=15,
        seasonality_emb_dim=3, value_head_quantiles=9,
        multi_patch_sizes=SIZES, enc_transformer_use_grad_checkpoint=False)


def batch(m, n=8, seed=0):
    """The value inputs of ``n`` full windows at 1h, 1d and 5min."""
    g = torch.Generator().manual_seed(seed)
    x = 100 + (torch.randn(n, T, 1, generator=g)).cumsum(1)
    names = ["1h", "1d", "5min", "1h"] * (n // 4)
    freq = torch.tensor([FREQ_NAMES_V2.index(f) for f in names])
    torch.manual_seed(seed)
    x_norm, sizes, target = mean_std_inputs(m, x, freq, SIZES)
    return dict(x_norm=x_norm, sample_sizes=sizes, target_mask=target,
                pad_mask=m.rev_norm.pad_mask, freq_ids=freq, freq_embs=None,
                seasonality_ids=torch.zeros_like(freq), seasonality_embs=None)


def rows_of(inputs, rows):
    return {k: (v[rows] if torch.is_tensor(v) else v) for k, v in inputs.items()}


def poison(inputs, row):
    """A row whose values overflow the patch encoder: its loss is NaN."""
    inputs = dict(inputs)
    inputs["x_norm"] = inputs["x_norm"].clone()
    inputs["x_norm"][row, -64:] = 1e30
    return inputs


def loss_and_grads(train_py, m, inputs, active=None):
    m.zero_grad(set_to_none=True)
    loss = train_py.value_objective(m, inputs, ARGS, SIZES, active)[0]
    loss.backward()
    return loss, {n: p.grad.clone() for n, p in m.named_parameters()
                  if p.grad is not None}


def resolve(train_py, m, inputs):
    """What the trainer does after a non-finite first pass."""
    rng = rng_state(CPU)
    loss, _ = loss_and_grads(train_py, m, inputs)
    assert not is_finite_step(loss.item(), m)
    return train_py.skip_nan_rows(m, inputs, ARGS, SIZES, rng, CPU)


@pytest.mark.parametrize("culprits", [[3], [2, 5]])
def test_the_poisonous_rows_are_dropped(train_py, culprits):
    m = model()
    inputs = batch(m)
    for row in culprits:
        inputs = poison(inputs, row)
    found, passes, result = resolve(train_py, m, inputs)
    assert found == culprits and result is not None
    loss = result[0]
    got = {n: p.grad.clone() for n, p in m.named_parameters()
           if p.grad is not None}
    assert is_finite_step(loss.item(), m)
    rest = [r for r in range(8) if r not in culprits]
    want, want_grads = loss_and_grads(train_py, m, rows_of(inputs, rest))
    assert torch.allclose(loss, want, rtol=1e-5, atol=1e-7)
    for name, grad in want_grads.items():
        assert torch.allclose(got[name], grad, rtol=1e-4, atol=1e-7), name


def test_a_nan_gradient_behind_a_finite_loss_is_found(train_py, monkeypatch):
    """A row whose values give a finite loss and a NaN gradient: its value
    head output gets 0 * sqrt(0 * |v|), which is 0 with a gradient 0 * inf.
    The fault follows the values of the row, as a data fault does."""
    real = fh.value_space_forward
    marker = 7777.0

    def forward(model_, x, **kw):
        f, o, v = real(model_, x, **kw)
        rows = (x[:, -1, 0] == marker).nonzero().view(-1)
        if len(rows) == 0:
            return f, o, v
        bump = torch.zeros_like(v)
        bump[rows] = 0.0 * torch.sqrt(0.0 * v[rows].abs().sum())
        return f, o, v + bump

    monkeypatch.setattr(fh, "value_space_forward", forward)
    m = model()
    inputs = batch(m)
    inputs["x_norm"] = inputs["x_norm"].clone()
    inputs["x_norm"][6, -1, 0] = marker
    loss, _ = loss_and_grads(train_py, m, inputs)
    assert torch.isfinite(loss) and not is_finite_step(loss.item(), m)
    found, _, result = train_py.skip_nan_rows(
        m, inputs, ARGS, SIZES, rng_state(CPU), CPU)
    assert found == [6] and result is not None


def test_a_mixup_batch_builds_its_embeddings_again_in_each_pass(train_py):
    """The mixup embeddings carry the graph of the first pass, which its
    backward frees. Each later pass builds them again: the same values."""
    m = model()
    inputs = batch(m)
    freq, seas = inputs["freq_ids"], inputs["seasonality_ids"]
    mix = {"weight": 0.3, "partner": torch.tensor([3, 0, 6, 1, 7, 2, 4, 5])}
    kept = torch.arange(8)
    embs = train_py.mixed_label_embeddings(m, mix, freq, seas, kept)
    inputs["freq_embs"], inputs["seasonality_embs"] = embs()
    inputs["label_embs"] = embs
    again = embs()
    assert torch.equal(again[0], inputs["freq_embs"])
    assert torch.equal(again[1], inputs["seasonality_embs"])
    inputs = poison(inputs, 4)
    found, _, result = resolve(train_py, m, inputs)
    assert found == [4] and result is not None


def test_a_clean_batch_runs_no_bisection_and_is_unchanged(train_py):
    m = model()
    inputs = batch(m)
    loss, grads = loss_and_grads(train_py, m, inputs)
    assert is_finite_step(loss.item(), m)
    every = torch.ones(8, dtype=torch.bool)
    loss_all, grads_all = loss_and_grads(train_py, m, inputs, every)
    assert torch.allclose(loss, loss_all, rtol=1e-6)
    assert all(torch.allclose(grads[n], grads_all[n], rtol=1e-5, atol=1e-8)
               for n in grads)


def test_a_fault_that_no_row_explains_skips_the_step(train_py, monkeypatch):
    """A loss that is NaN only while rows 2 and 5 are both active: each
    half of the batch is clean, so no row alone explains it."""
    real = train_py.value_objective

    def objective(model_, inputs, args, sizes, active=None):
        out = real(model_, inputs, args, sizes, active)
        if active is None or (bool(active[2]) and bool(active[5])):
            out = (out[0] * float("nan"),) + tuple(out[1:])
        return out

    monkeypatch.setattr(train_py, "value_objective", objective)
    m = model()
    found, passes, result = resolve(train_py, m, batch(m))
    assert found is None and result is None and passes == 2
    assert all(p.grad is None for p in m.parameters())


# ---------------------------------------------------------------------------
# 3. The same random draws in every pass
# ---------------------------------------------------------------------------

def outputs_per_row(train_py, m, inputs, active, rng):
    """The value head outputs of every forward of one pass."""
    seen, real = [], fh.value_space_forward

    def spy(model_, x, **kw):
        out = real(model_, x, **kw)
        seen.append(out[2].detach().clone())
        return out

    fh.value_space_forward = spy
    try:
        if rng is not None:
            restore_rng(rng, CPU)
        train_py.value_objective(m, inputs, ARGS, SIZES, active)
    finally:
        fh.value_space_forward = real
    return seen


def test_each_row_draws_the_same_masks_in_every_pass(train_py):
    m = model(dropout=0.3, dropkey=0.7)
    m.train()
    inputs = batch(m)
    rng = rng_state(CPU)
    first = outputs_per_row(train_py, m, inputs, None, rng)
    active = torch.tensor([True, False, True, True, False, True, True, True])
    again = outputs_per_row(train_py, m, inputs, active, rng)
    groups = fh.patch_size_groups(inputs["sample_sizes"])
    calls = [rows for rows in groups.values() for _ in range(3)]
    for rows, a, b in zip(calls, first, again):
        keep = active[rows]
        assert torch.equal(a[keep], b[keep])
    fresh = outputs_per_row(train_py, m, inputs, active, None)
    assert any(not torch.equal(a[active[rows]], b[active[rows]])
               for rows, a, b in zip(calls, first, fresh))


def test_the_group_weights_count_the_active_rows_only(train_py):
    m = model()
    inputs = batch(m)
    every = torch.ones(8, dtype=torch.bool)
    with torch.no_grad():
        plain = fh.multi_patch_value_objective(
            m, inputs["x_norm"], inputs["sample_sizes"], depth=1,
            target_mask=inputs["target_mask"], pad_mask=inputs["pad_mask"],
            freq_ids=inputs["freq_ids"],
            seasonality_ids=inputs["seasonality_ids"])[0]
        same = fh.multi_patch_value_objective(
            m, inputs["x_norm"], inputs["sample_sizes"], depth=1,
            target_mask=inputs["target_mask"], pad_mask=inputs["pad_mask"],
            freq_ids=inputs["freq_ids"],
            seasonality_ids=inputs["seasonality_ids"], active=every)[0]
    assert torch.equal(plain, same)


# ---------------------------------------------------------------------------
# 4. The trainer
# ---------------------------------------------------------------------------

TINY_RUN = (
    "--value-space-objective", "--gift-pretrain", "--freq-vocab", "v2",
    "--multi-patch-sizes", "8,16,32,64,128", "--rev-norm-kind", "meanstd",
    "--t-raw", "4096", "--n-channels", "1", "--d-model", "16",
    "--n-heads", "2", "--num-layers", "1", "--num-encoder-layers", "1",
    "--batch-size", "8", "--synth-kind", "forked-arma",
    "--mix-ratio", "0.25", "--crossfade-triplets", "1",
    "--freq-emb-dim", "3", "--seasonality-emb-dim", "3",
    "--train-rollout-depth", "2", "--train-rollout-reduce", "sum",
    "--lr", "1e-3", "--lr-final", "0", "--lr-cosine-steps", "166000",
    "--lr-warmup-steps", "10000", "--grad-clip", "1.0",
    "--log-every", "1", "--save-every", "1000000")


def run(root, save_dir, *extra, env=None):
    full_env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES="",
                    OMP_NUM_THREADS="4")
    full_env.pop("CF_NAN_DEBUG", None)
    full_env.update(env or {})
    script = (root / "experiments" / "2026-04-27_freq-embedding" / "scripts"
              / "train.py")
    return subprocess.run(
        [sys.executable, str(script), "--device", "cpu", "--weight-decay",
         "0.1", *TINY_RUN, "--save-dir", str(save_dir), "--run-name", "r",
         *extra], capture_output=True, text=True, env=full_env, timeout=900)


def corpus(root, poisoned):
    """The #419 test corpus, and with ``poisoned`` a heavy source whose
    series end in values of 1e30."""
    pytest.importorskip("pyarrow")
    from tests.test_419_gift_pretrain import (build_corpus, index_sources,
                                              write_index, write_source)
    folder, index, _ = build_corpus(root / "corpus")
    if poisoned:
        rng = np.random.default_rng(1)
        rows = []
        for _ in range(4):
            s = (50.0 + rng.standard_normal(600).cumsum()).astype(np.float32)
            s[-40:] = 1e30
            rows.append({"target": s})
        write_source(folder, "tiny_poison", rows, "H")
        weights = {name: 1.0 for name in index["sources"]}
        weights["tiny_poison"] = 40.0
        index = index_sources(folder, weights)
    return ("--gift-pretrain-root", str(folder),
            "--gift-pretrain-index", str(write_index(root, index)))


@pytest.mark.parametrize("mixup", ["0", "1.0"])
def test_a_run_with_poisoned_rows_goes_on_without_them(tmp_path, mixup):
    """The poisoned source sends windows whose values overflow the patch
    encoder. The z-filter is off, so they reach the model. With mixup, a
    row mixed with a poisoned partner is poisonous too."""
    data = corpus(tmp_path, poisoned=True)
    r = run(REPO_ROOT, tmp_path / "save", "--skip-nan-samples",
            "--meanstd-z-max", "0", "--mixup-p", mixup, "--total-steps", "4",
            *data)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "NaN/Inf DETECTED" not in r.stdout
    rows = list(csv.DictReader(open(tmp_path / "save" / "r_losses.csv")))
    assert len(rows) == 4
    assert all(np.isfinite(float(row["loss"])) for row in rows)
    dropped = [int(row["nan_dropped"]) for row in rows]
    assert max(dropped) >= 1 and min(dropped) >= 0
    step = next(int(row["step"]) for row in rows
                if int(row["nan_dropped"]) > 0)
    assert f"[skip-nan] step {step}:" in r.stdout
    assert "freq 1h, patch " in r.stdout and "max |z|" in r.stdout
    saved = torch.load(tmp_path / "save" / f"r_nan_rows_{step}.pt",
                       weights_only=False)
    poisoned = (saved["loaded"] == 1e30).any(dim=1).view(-1)
    if mixup == "0":
        assert poisoned.all()
    else:
        partners = saved["mixup"]["partner"][saved["rows"]]
        assert "mixed" in r.stdout and len(partners) == len(saved["rows"])


def test_with_a_clean_stream_the_flag_changes_no_loss(tmp_path):
    data = corpus(tmp_path, poisoned=False)
    off = run(REPO_ROOT, tmp_path / "off", "--mixup-p", "1.0",
              "--total-steps", "3", *data)
    on = run(REPO_ROOT, tmp_path / "on", "--mixup-p", "1.0",
             "--total-steps", "3", "--skip-nan-samples", *data)
    assert off.returncode == 0 and on.returncode == 0, on.stderr[-2000:]
    a = list(csv.DictReader(open(tmp_path / "off" / "r_losses.csv")))
    b = list(csv.DictReader(open(tmp_path / "on" / "r_losses.csv")))
    assert [row["loss"] for row in a] == [row["loss"] for row in b]
    assert [row["nan_dropped"] for row in b] == ["0", "0", "0"]
    assert "nan_dropped" not in a[0]


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


def test_off_the_losses_csv_equals_the_commit_before(tmp_path):
    data = corpus(tmp_path, poisoned=False)
    old_root = git_tree("33cd6dbd", tmp_path)
    old = run(old_root, tmp_path / "old", "--mixup-p", "1.0",
              "--total-steps", "3", *data)
    new = run(REPO_ROOT, tmp_path / "new", "--mixup-p", "1.0",
              "--total-steps", "3", *data)
    assert old.returncode == 0 and new.returncode == 0, new.stderr[-2000:]
    assert ((tmp_path / "new" / "r_losses.csv").read_bytes()
            == (tmp_path / "old" / "r_losses.csv").read_bytes())


# ---------------------------------------------------------------------------
# 5. The refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extra,env,why", [
    (("--skip-nan-samples",), None, "--value-space-objective"),
])
def test_the_flag_needs_the_value_objective(tmp_path, extra, env, why):
    script = TRAIN_PY
    r = subprocess.run(
        [sys.executable, str(script), "--device", "cpu", "--weight-decay",
         "0.1", "--total-steps", "1", "--save-dir", str(tmp_path / "never"),
         *extra], capture_output=True, text=True, timeout=300,
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT),
                 CUDA_VISIBLE_DEVICES=""))
    assert r.returncode != 0 and why in r.stdout + r.stderr
    assert not (tmp_path / "never").exists()


def test_the_flag_refuses_the_nan_debug_mode(tmp_path):
    r = run(REPO_ROOT, tmp_path / "never", "--skip-nan-samples",
            "--total-steps", "1", env={"CF_NAN_DEBUG": "1"})
    assert r.returncode != 0 and "Use one of them" in r.stdout + r.stderr
    assert not (tmp_path / "never").exists()


def test_the_flag_is_on_the_value_space_allowlist(train_py):
    assert "skip_nan_samples" in train_py.VALUE_SPACE_FLAGS
