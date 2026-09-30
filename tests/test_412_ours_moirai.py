"""Tests for #412: our contrastive objective with four parts of the Moirai
copy (#421f): the mean/std scaling, the patch sizes 8 to 128, the Moirai
recipe, and the drop of the rows that make a step bad.

Groups, all on the CPU:

1. The contrastive path without the new parts writes the losses CSV of the
   base commit byte for byte.
2. A multi-patch contrastive step: a finite loss, a gradient for the patch
   encoder of each size, a teacher that moves by EMA only, and a rollout
   that advances P values per depth at size P.
3. The mean/std scaling: loc and scale read only the values before the
   split, and the contrastive terms read the values after it too.
4. The row drop with coupled terms: the step after the drop equals the step
   on the batch without the dropped rows, loss and gradients.
5. The trainer: three steps of the run's full command line, a resume, a
   stream with poisoned rows, and the refusals.
6. The scoring head: one head per patch size. Head P decodes P values, each
   config reads the head of its frequency's size, each row trains the head
   of its size, the head's statistics read only the context before the
   split, and the forecast is unscaled with the context loc and scale.
7. The scoring scripts: the head trainer trains a bank on the run's
   checkpoint, and the GIFT-Eval script loads it and forecasts with it.
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

import src.forecasting_head as fh  # noqa: E402
from src.checkpoint import multi_patch_sizes_of  # noqa: E402
from src.forecasting_head import (ForecastingHeadBank,  # noqa: E402
                                  TransformerQuantileForecastingHead,
                                  bank_quantile_loss, bank_training_inputs,
                                  forecast_B4, head_bank_sizes,
                                  mean_std_inputs)
from src.freq_embedding import FREQ_NAMES_V2  # noqa: E402
from src.models import ConfigurableModel  # noqa: E402
from src.nan_skip import (is_finite_step, restore_rng,  # noqa: E402
                          rng_state)

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
GIFT_SCRIPTS = REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
HEAD_PY = GIFT_SCRIPTS / "train_forecasting_head.py"
EVAL_PY = GIFT_SCRIPTS / "eval_gift_eval_official.py"
BASE_COMMIT = "3d57bf9a"
SIZES = (8, 16, 32, 64, 128)
T = 1024
CPU = torch.device("cpu")

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


# ---------------------------------------------------------------------------
# Unit helpers: the run's objective on a model the CPU trains at once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_py():
    spec = importlib.util.spec_from_file_location("train_412", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_args(train_py, *extra):
    """The run's parsed flags, at the tiny size, with the loss configured
    as the trainer configures it."""
    args = train_py.parse_args([*CYAN, *MOIRAI_RECIPE, *MOIRAI_PARTS, *TINY,
                                *extra])
    train_py.configure_loss_spec(args)
    return args


def model(dropout=0.0, dropkey=0.0):
    """The run's model at d_model 16: one GRU patch encoder per size, the
    EMA teacher of the patch encoders and the encoder stack, the mean/std
    scaling with zero padding."""
    torch.manual_seed(0)
    return ConfigurableModel(
        C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=2.0, dropout=dropout, rev_norm_kind="meanstd",
        rev_norm_skip_leading_zeros=True, num_encoder_layers=1,
        encoder_dropkey=dropkey, encoder_dropkey_share_heads=True,
        encoder_dropkey_share_layers=True, freq_emb_dim=3,
        num_freqs=len(FREQ_NAMES_V2), seasonality_emb_dim=3,
        multi_patch_sizes=SIZES, qk_norm=True, attn_out_norm=True,
        ema_embedding=True, ema_encoder=True,
        enc_transformer_use_grad_checkpoint=False)


LENGTHS = (T, 700, 400, T, 900, 300, T, 600)
FREQS = ("1h", "1d", "5min", "1M", "10s", "1h", "1Y", "1d")


def windows(lengths=LENGTHS, seed=0):
    """``[B, T, 1]``: one random walk per length, left zero padded."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(len(lengths), T, 1)
    for b, n in enumerate(lengths):
        x[b, T - n:, 0] = 100.0 + torch.randn(n, generator=g).cumsum(0)
    return x


def batch(m, sizes=None, seed=0):
    """The step inputs of eight windows at six frequencies, as the trainer
    builds them: sizes and splits drawn, loc and scale from the context.
    ``sizes`` replaces the drawn sizes."""
    freq = torch.tensor([FREQ_NAMES_V2.index(f) for f in FREQS])
    torch.manual_seed(seed)
    x_norm, drawn, target = mean_std_inputs(m, windows(seed=seed), freq,
                                            SIZES)
    return dict(x_norm=x_norm,
                sample_sizes=drawn if sizes is None else torch.tensor(sizes),
                target_mask=target, pad_mask=m.rev_norm.pad_mask,
                freq_ids=freq, freq_embs=None,
                seasonality_ids=torch.zeros_like(freq), seasonality_embs=None)


def rows_of(inputs, rows):
    return {k: (v[rows] if torch.is_tensor(v) else v)
            for k, v in inputs.items()}


def poison(inputs, row):
    """A row whose values overflow the patch encoder: its latents are NaN,
    and through the coupled terms every gradient is."""
    inputs = dict(inputs)
    inputs["x_norm"] = inputs["x_norm"].clone()
    inputs["x_norm"][row, -64:] = 1e30
    return inputs


def loss_and_grads(objective, m, inputs, args, active=None):
    m.zero_grad(set_to_none=True)
    loss = objective(m, inputs, args, SIZES, active)[0]
    loss.backward()
    return loss, {n: p.grad.clone() for n, p in m.named_parameters()
                  if p.grad is not None}


# ---------------------------------------------------------------------------
# 2. A multi-patch contrastive step
# ---------------------------------------------------------------------------

ALL_SIZES = (8, 16, 32, 64, 128, 8, 16, 32)


def test_a_step_trains_the_encoder_of_every_size(train_py):
    args = run_args(train_py)
    m = model()
    loss, f, o, extra = train_py.contrastive_objective(
        m, batch(m, ALL_SIZES), args, SIZES, rep_w=1.0)
    assert torch.isfinite(loss)
    assert {"l_rep", "l_align"} <= set(extra.values["terms"])
    assert np.isfinite(extra.values["sigreg_e"])
    assert np.isfinite(extra.values["sigreg_h"])
    loss.backward()
    for size in SIZES:
        grads = [p.grad for p in m.encoder.encoders[str(size)].parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(bool(g.abs().sum() > 0) for g in grads), size
    assert all(p.grad is None for p in m.teacher_input_to_latent.parameters())
    assert all(p.grad is None for p in m.teacher_encoder_layers.parameters())
    # The diagnostics read the base size's group: 1,024 / 16 patches.
    assert f.shape == o.shape == (2, T // 16, 1, 16)
    assert len(extra.rollout) == 3


def test_the_teacher_moves_by_ema_only(train_py):
    args = run_args(train_py)
    m = model()
    loss = train_py.contrastive_objective(m, batch(m, ALL_SIZES), args,
                                          SIZES, rep_w=1.0)[0]
    loss.backward()
    torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.1).step()
    before = {n: p.clone() for n, p in m.teacher_input_to_latent.named_parameters()}
    student = dict(m.transformer.input_to_latent.named_parameters())
    m.update_teacher(0.9)
    moved = 0
    for name, p in m.teacher_input_to_latent.named_parameters():
        assert name.startswith("encoders.")
        assert torch.allclose(p, 0.9 * before[name] + 0.1 * student[name])
        moved += int(not torch.equal(p, before[name]))
    assert moved > 0


def test_each_size_rolls_out_by_its_own_patches(train_py, monkeypatch):
    """Depth j of a group at size P is the forecaster on its own output at
    that size: one latent per patch of P values, so each depth advances P
    values."""
    args = run_args(train_py)
    seen = {}
    real = train_py.group_forward

    def spy(model_, x_norm, labels, size, args_):
        lat = real(model_, x_norm, labels, size, args_)
        seen[size] = lat
        return lat

    monkeypatch.setattr(train_py, "group_forward", spy)
    m = model()
    train_py.contrastive_objective(m, batch(m, ALL_SIZES), args, SIZES,
                                   rep_w=1.0)
    assert sorted(seen) == list(SIZES)
    for size, lat in seen.items():
        assert lat.f.shape[1] == T // size
        assert [r.shape[1] for r in lat.rollout] == [T // size] * 3
        assert lat.teacher.shape == lat.o.shape


def test_a_group_reads_its_own_rows_only(train_py):
    """The groups share no negative, key or statistic: new values in row 0
    (size 8) move the gradient of row 5, its partner at size 8, and leave
    the gradient of every row of another size as it is."""
    args = run_args(train_py)
    m = model()
    inputs = batch(m, ALL_SIZES)
    grads = []
    for scale in (1.0, 0.5):
        x = inputs["x_norm"].clone()
        x[0] = scale * x[0]
        x.requires_grad_(True)
        torch.manual_seed(2)
        train_py.contrastive_objective(m, dict(inputs, x_norm=x), args,
                                       SIZES, rep_w=1.0)[0].backward()
        grads.append(x.grad)
    assert not torch.allclose(grads[0][5], grads[1][5])
    for row in (1, 2, 3, 4, 6, 7):
        assert torch.equal(grads[0][row], grads[1][row]), row


# ---------------------------------------------------------------------------
# 3. The mean/std scaling reads the context before the split only
# ---------------------------------------------------------------------------

def test_values_after_the_split_leave_loc_and_scale_alone(train_py):
    """A change after the split of each window moves neither loc nor scale,
    and so no normalised value before the split. The contrastive terms read
    every real position, so the loss moves."""
    args = run_args(train_py)
    m = model()
    freq = torch.tensor([FREQ_NAMES_V2.index(f) for f in FREQS])
    x = windows()
    torch.manual_seed(5)
    x_norm, sizes, target = mean_std_inputs(m, x, freq, SIZES)
    loc, scale = m.rev_norm.mean.clone(), m.rev_norm.stdev.clone()
    changed = x + target * (1e3 * torch.rand(x.shape) + 1.0)
    torch.manual_seed(5)
    x_norm2, sizes2, target2 = mean_std_inputs(m, changed, freq, SIZES)
    assert torch.equal(sizes, sizes2) and torch.equal(target, target2)
    assert target.any(dim=1).all()
    assert torch.equal(m.rev_norm.mean, loc)
    assert torch.equal(m.rev_norm.stdev, scale)
    assert torch.equal(x_norm[~target], x_norm2[~target])
    inputs = dict(x_norm=x_norm, sample_sizes=sizes, target_mask=target,
                  pad_mask=m.rev_norm.pad_mask, freq_ids=freq, freq_embs=None,
                  seasonality_ids=torch.zeros_like(freq),
                  seasonality_embs=None)
    torch.manual_seed(0)
    one = train_py.contrastive_objective(m, inputs, args, SIZES, rep_w=1.0)[0]
    torch.manual_seed(0)
    two = train_py.contrastive_objective(
        m, dict(inputs, x_norm=x_norm2), args, SIZES, rep_w=1.0)[0]
    assert not torch.equal(one, two)


# ---------------------------------------------------------------------------
# 4. The row drop with coupled terms
# ---------------------------------------------------------------------------

def drop_and_compare(train_py, inputs, culprits):
    """The trainer's search after a bad first pass, and the step on the
    batch without the culprits from the same random state."""
    args = run_args(train_py)
    m = model()
    objective = train_py.step_objective(args, 1.0)
    rng = rng_state(CPU)
    loss, _ = loss_and_grads(objective, m, inputs, args)
    assert not is_finite_step(loss.item(), m)
    found, _, result = train_py.skip_nan_rows(
        m, inputs, args, SIZES, rng, CPU, objective=objective)
    assert sorted(found) == sorted(culprits) and result is not None
    got = {n: p.grad.clone() for n, p in m.named_parameters()
           if p.grad is not None}
    assert is_finite_step(result[0].item(), m)
    restore_rng(rng, CPU)
    rest = [r for r in range(len(LENGTHS)) if r not in culprits]
    want, want_grads = loss_and_grads(objective, m, rows_of(inputs, rest),
                                      args)
    assert torch.allclose(result[0], want, rtol=1e-5, atol=1e-6)
    assert set(got) == set(want_grads)
    for name, grad in want_grads.items():
        assert torch.allclose(got[name], grad, rtol=1e-4, atol=1e-6), name


@pytest.mark.parametrize("culprits", [[2], [2, 5]])
def test_the_step_after_the_drop_equals_the_batch_without_the_rows(
        train_py, culprits):
    """Row 2 shares size 32 with row 7, and row 5 shares size 8 with row
    0: each culprit leaves a group that goes on with its other row."""
    m = model()
    inputs = batch(m, ALL_SIZES)
    for row in culprits:
        inputs = poison(inputs, row)
    drop_and_compare(train_py, inputs, culprits)


def test_a_row_alone_in_its_group_leaves_with_its_group(train_py):
    """Row 4 is the only row at size 128: its group has no active row after
    the drop, and runs no term."""
    m = model()
    inputs = poison(batch(m, ALL_SIZES), 4)
    drop_and_compare(train_py, inputs, [4])


def test_a_clean_batch_is_unchanged_with_every_row_active(train_py):
    args = run_args(train_py)
    m = model()
    inputs = batch(m, ALL_SIZES)
    objective = train_py.step_objective(args, 1.0)
    torch.manual_seed(1)
    loss, grads = loss_and_grads(objective, m, inputs, args)
    torch.manual_seed(1)
    every = torch.ones(len(LENGTHS), dtype=torch.bool)
    loss_all, grads_all = loss_and_grads(objective, m, inputs, args, every)
    assert torch.equal(loss, loss_all)
    assert all(torch.equal(grads[n], grads_all[n]) for n in grads)


def test_each_row_draws_the_same_masks_in_every_pass(train_py, monkeypatch):
    """Every group runs its forward before any term, so a pass with some
    rows inert draws the same dropout and DropKey masks for the others."""
    args = run_args(train_py)
    m = model(dropout=0.3, dropkey=0.7)
    m.train()
    inputs = batch(m, ALL_SIZES)
    real = train_py.group_forward

    def outputs(active, rng):
        seen = []

        def spy(model_, x_norm, labels, size, args_):
            lat = real(model_, x_norm, labels, size, args_)
            seen.append(lat.o.detach().clone())
            return lat
        monkeypatch.setattr(train_py, "group_forward", spy)
        if rng is not None:
            restore_rng(rng, CPU)
        train_py.contrastive_objective(m, inputs, args, SIZES, active,
                                       rep_w=1.0)
        return seen

    rng = rng_state(CPU)
    first = outputs(None, rng)
    active = torch.tensor([True, False, True, True, False, True, True, True])
    again = outputs(active, rng)
    groups = fh.patch_size_groups(inputs["sample_sizes"])
    for rows, a, b in zip(groups.values(), first, again):
        keep = active[rows]
        assert torch.equal(a[keep], b[keep])
    fresh = outputs(active, None)
    assert any(not torch.equal(a[active[rows]], b[active[rows]])
               for rows, a, b in zip(groups.values(), first, fresh))


def test_a_step_with_no_active_row_trains_on_nothing(train_py):
    """Every window of the step is far from its context (#421): no term
    reads a row, the loss is a zero on the graph, and no gradient moves."""
    args = run_args(train_py)
    m = model()
    inputs = batch(m, ALL_SIZES)
    inputs["active"] = torch.zeros(len(LENGTHS), dtype=torch.bool)
    loss, f, o, extra = train_py.contrastive_objective(m, inputs, args,
                                                       SIZES, rep_w=1.0)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert all(float(p.grad.abs().sum()) == 0.0 for p in m.parameters()
               if p.grad is not None)
    assert f.shape[0] > 0 and np.isnan(extra.values["sigreg_e"])


# ---------------------------------------------------------------------------
# 5. The trainer
# ---------------------------------------------------------------------------

FULL_RUN = CYAN + MOIRAI_RECIPE + MOIRAI_PARTS + TINY


def finite(rows, *columns):
    return all(np.isfinite(float(row[c])) for row in rows for c in columns)


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """Two steps of the run's command line at the tiny size, then a resume
    to step 3."""
    root = tmp_path_factory.mktemp("full_run")
    data = corpus(root)
    first = run(REPO_ROOT, root / "save", *FULL_RUN, "--total-steps", "2",
                *data)
    resumed = run(REPO_ROOT, root / "save", *FULL_RUN, "--total-steps", "3",
                  "--resume", str(root / "save" / "r_final.pth"), *data)
    return root / "save", first, resumed


def test_the_full_command_line_trains_on_the_cpu(full_run):
    save, first, _ = full_run
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    assert "Objective (#412)" in first.stdout
    assert "NaN/Inf DETECTED" not in first.stdout
    rows = losses(save)
    assert [r["step"] for r in rows] == ["1", "2"]
    assert finite(rows, "loss", "loss_tau_ref", "l_rep", "l_align",
                  "sigreg_e", "sigreg_h", "grad_norm", "cos_err_d3")
    assert all(int(r["nan_dropped"]) == 0 for r in rows)
    sd = torch.load(save / "r_final.pth", map_location="cpu",
                    weights_only=True)
    assert multi_patch_sizes_of(sd) == SIZES
    assert "rev_norm.mean_std_scaling" in sd
    assert all(f"teacher_input_to_latent.encoders.{p}.skip.weight" in sd
               for p in SIZES)
    assert not any(k.startswith("value_head") for k in sd)


def test_the_run_resumes_with_its_teacher_bank(full_run):
    save, first, resumed = full_run
    assert first.returncode == 0, first.stderr[-3000:]
    out = resumed.stdout
    assert resumed.returncode == 0, out[-3000:] + resumed.stderr[-3000:]
    assert "Resumed from" in out and "[      3]" in out
    assert "[lr] WARNING" not in out


def poisoned(root):
    """The #421 corpus with a heavy source whose series end in 1e30."""
    from tests.test_421_skip_nan import corpus as corpus_421
    return corpus_421(root, poisoned=True)


def test_a_run_with_poisoned_rows_goes_on_without_them(tmp_path):
    """The z-filter is off, so the poisoned windows reach the model, and
    through the coupled terms every gradient of the step is NaN. The run
    drops the poisoned rows and goes on."""
    r = run(REPO_ROOT, tmp_path / "save", *FULL_RUN, "--meanstd-z-max", "0",
            "--skip-spike-samples", "0", "--mixup-p", "0",
            "--total-steps", "4", *poisoned(tmp_path))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "NaN/Inf DETECTED" not in r.stdout
    rows = losses(tmp_path / "save")
    assert len(rows) == 4 and finite(rows, "loss", "grad_norm", "l_rep")
    dropped = [int(row["nan_dropped"]) for row in rows]
    assert max(dropped) >= 1 and min(dropped) >= 0
    step = next(row["step"] for row in rows if int(row["nan_dropped"]) > 0)
    assert f"[skip-nan] step {step}:" in r.stdout
    saved = torch.load(tmp_path / "save" / f"r_nan_rows_{step}.pt",
                       weights_only=False)
    assert (saved["loaded"] == 1e30).any(dim=1).view(-1).all()


def test_with_a_clean_stream_the_row_drop_changes_no_loss(tmp_path):
    data = corpus(tmp_path)
    flags = FULL_RUN + ("--skip-spike-samples", "0", "--total-steps", "3")
    off = [f for f in flags if f != "--skip-nan-samples"]
    a = run(REPO_ROOT, tmp_path / "off", *off, *data)
    b = run(REPO_ROOT, tmp_path / "on", *flags, *data)
    assert a.returncode == 0, a.stdout[-3000:] + a.stderr[-3000:]
    assert b.returncode == 0, b.stdout[-3000:] + b.stderr[-3000:]
    loss = [row["loss"] for row in losses(tmp_path / "off")]
    assert loss == [row["loss"] for row in losses(tmp_path / "on")]


@pytest.mark.parametrize("flag", [
    ("--multi-patch-sizes", "8,16,32,64,128"), ("--rev-norm-kind", "meanstd"),
    ("--skip-nan-samples",)])
def test_a_term_without_a_row_mask_is_refused(tmp_path, flag):
    """The default shape takes no row mask, so none of the three flags can
    reach it."""
    r = run(REPO_ROOT, tmp_path / "never", "--weight-decay", "0.1",
            "--total-steps", "1", *flag)
    assert r.returncode != 0 and "takes no row mask" in r.stdout + r.stderr
    assert not (tmp_path / "never").exists()


def test_the_objective_by_row_runs_on_one_gpu(tmp_path):
    r = run(REPO_ROOT, tmp_path / "never", *CYAN, "--weight-decay", "0.1",
            "--multi-patch-sizes", "8,16,32,64,128", "--total-steps", "1",
            env={"WORLD_SIZE": "2"})
    assert r.returncode != 0 and "one GPU" in r.stdout + r.stderr
    assert not (tmp_path / "never").exists()


# ---------------------------------------------------------------------------
# 6. The scoring head: one head per patch size
# ---------------------------------------------------------------------------

Q = 9


def bank_of(sizes=SIZES):
    torch.manual_seed(1)
    return ForecastingHeadBank({
        p: TransformerQuantileForecastingHead(H=16, num_layers=1, nhead=2,
                                              forecast_len=p)
        for p in sizes})


def test_head_p_decodes_the_p_values_of_the_next_patch():
    bank = bank_of()
    assert head_bank_sizes(bank.state_dict()) == SIZES
    for size in SIZES:
        head = bank.head_for(size)
        assert head.patch_size == size
        assert head(torch.randn(3, 5, 16)).shape == (3, 5, Q, size)


@pytest.mark.parametrize("freq,size", [
    ("A-DEC", 8), ("Q-DEC", 8), ("M", 16), ("W-SUN", 16), ("D", 16),
    ("H", 32), ("15T", 64), ("10S", 128)])
def test_each_config_reads_the_head_of_its_frequency(freq, size):
    bank = bank_of()
    assert bank.for_frequency(model(), freq) is bank.head_for(size)


def test_the_head_statistics_read_the_context_before_the_split():
    """loc and scale of a head step read no value after the split, and the
    loss is scored on the real values after it only."""
    m = model()
    freq = torch.tensor([FREQ_NAMES_V2.index(f) for f in FREQS])
    x = windows()
    torch.manual_seed(5)
    x_norm, sizes, keep = bank_training_inputs(m, x, freq, SIZES)
    loc, scale = m.rev_norm.mean.clone(), m.rev_norm.stdev.clone()
    assert keep.any(dim=1).all() and not (keep & m.rev_norm.pad_mask).any()
    changed = x + keep * (1e3 * torch.rand(x.shape) + 1.0)
    torch.manual_seed(5)
    x_norm2, sizes2, keep2 = bank_training_inputs(m, changed, freq, SIZES)
    assert torch.equal(sizes, sizes2) and torch.equal(keep, keep2)
    assert torch.equal(m.rev_norm.mean, loc)
    assert torch.equal(m.rev_norm.stdev, scale)
    assert torch.equal(x_norm[~keep], x_norm2[~keep])
    for row, size in enumerate(sizes.tolist()):
        first = int(keep[row, :, 0].float().argmax())
        assert first % size == 0 and keep[row, first:, 0].all()


def test_each_row_trains_the_head_of_its_size():
    """Rows at sizes 8 and 32 only: those two heads get a gradient, the
    frozen backbone and the other heads get none, and the loss is the two
    group losses weighted by their share of the rows."""
    m = model().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    bank = bank_of().eval()  # no dropout: the two sums below compare
    freq = torch.tensor([FREQ_NAMES_V2.index(f) for f in FREQS])
    seas = torch.zeros_like(freq)
    torch.manual_seed(5)
    x_norm, _, keep = bank_training_inputs(m, windows(), freq, SIZES)
    sizes = torch.tensor([8, 8, 32, 32, 8, 32, 32, 32])
    loss = bank_quantile_loss(m, bank, x_norm, sizes, keep, freq, seas)
    loss.backward()
    for size in SIZES:
        grads = [p.grad for p in bank.head_for(size).parameters()]
        if size in (8, 32):
            assert all(g is not None for g in grads), size
        else:
            assert all(g is None for g in grads), size
    parts = []
    for size, rows in ((8, [0, 1, 4]), (32, [2, 3, 5, 6, 7])):
        labels = dict(freq_ids=freq[rows], seasonality_ids=seas[rows])
        parts.append(len(rows) / 8 * fh._bank_group_loss(
            m, bank.head_for(size), x_norm[rows], keep[rows], size, labels))
    assert torch.allclose(loss, parts[0] + parts[1])


@pytest.mark.parametrize("freq,size", [("A-DEC", 8), ("H", 32), ("10S", 128)])
def test_b4_reads_the_context_at_the_heads_size(monkeypatch, freq, size):
    seen, real = [], fh.extract_encoder_latents

    def spy(backbone, x, **kw):
        seen.append(kw.get("patch_size"))
        return real(backbone, x, **kw)

    monkeypatch.setattr(fh, "extract_encoder_latents", spy)
    m = model().eval()
    ctx = 50.0 + torch.randn(T, 1).cumsum(0)
    out = forecast_B4(m, bank_of().for_frequency(m, freq), ctx, 45, "cpu")
    assert out.shape == (Q, 45, 1) and np.isfinite(out).all()
    assert seen == [size]


def test_the_forecast_is_unscaled_with_the_context_statistics():
    """The forecast is the head's normalised output times the context scale
    plus its loc. Taking them off again gives the head's output back. An
    affine change of the context moves the forecast the same way."""
    m = model().eval()
    head = bank_of().head_for(32).eval()
    ctx = 50.0 + torch.randn(T, 1).cumsum(0)
    out = torch.as_tensor(forecast_B4(m, head, ctx, 40, "cpu"))
    e_ctx, _ = fh.extract_encoder_latents(m, ctx[None], patch_size=32)
    loc, scale = m.rev_norm.mean.view(()), m.rev_norm.stdev.view(())
    rolled = fh.rollout_latent(m, e_ctx, 2)
    with torch.no_grad():
        raw = fh._b_variant_decode(head, e_ctx, rolled, e_ctx.size(1))
    normalised = torch.cat([raw[0, 0], raw[0, 1]], dim=-1)[..., :40]
    assert torch.allclose(out[..., 0], normalised * scale + loc, atol=1e-4)
    assert torch.allclose((out[..., 0] - loc) / scale, normalised, atol=1e-5)
    moved = forecast_B4(m, head, 3.0 * ctx - 20.0, 40, "cpu")
    assert np.allclose(moved, 3.0 * out.numpy() - 20.0, rtol=1e-4, atol=1e-3)


# ---------------------------------------------------------------------------
# 7. The scoring scripts on the run's checkpoint
# ---------------------------------------------------------------------------

# The flags head_eval_bb.sh gives the head trainer, with the tiny backbone
# shape (CF_BB_SHAPE) and the CPU. The protocol names --rev-norm-kind ewma
# and --forecast-len 16. The checkpoint's scaling and sizes win.
BB_SHAPE = ("--d-model", "16", "--n-heads", "2", "--num-layers", "1")
HEAD_PROTOCOL = (
    "--device", "cpu", "--quantile-head", "--grad-clip", "1.0",
    "--forecast-len", "16", "--batch-size", "8", "--lr", "1e-3",
    "--total-steps", "3", "--save-every", "5000", "--log-every", "1",
    "--seed", "20260722", "--hf-repo", "jeremycochoy/gift-pretrain-full-4096",
    "--hf-path", "small_v1", "--head-arch", "transformer",
    "--head-num-layers", "2", "--head-nhead", "8", "--head-ffn-mult", "4.0",
    "--head-causal", "true", "--head-train-input", "e_then_f",
    "--head-dropout", "0.1", "--t-raw", "4096", "--n-channels", "1",
    *BB_SHAPE, "--encoder-type", "gru", "--rev-norm-kind", "ewma",
    "--rev-norm-span", "128", "--freq-emb-dim", "3",
    "--seasonality-emb-dim", "3")
# The flags eval_local.sh gives the GIFT-Eval script.
EVAL_PROTOCOL = (
    "--strategy", "B4", "--forecast-len", "16", "--device", "cpu",
    "--t-raw", "4096", "--n-channels", "1", *BB_SHAPE, "--encoder-type",
    "gru", "--rev-norm-kind", "ewma", "--rev-norm-span", "128",
    "--head-nhead", "8", "--head-causal", "true")


@pytest.fixture(scope="module", params=["student", "teacher"])
def head_run(request, full_run):
    """Three steps of the head protocol on the run's checkpoint."""
    save, first, resumed = full_run
    assert resumed.returncode == 0, resumed.stderr[-3000:]
    out = save / f"head_{request.param}"
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="4")
    data = corpus(save / f"data_{request.param}")
    r = subprocess.run(
        [sys.executable, str(HEAD_PY), "--backbone-path",
         str(save / "r_final.pth"), "--encoder-source", request.param,
         *HEAD_PROTOCOL, "--save-dir", str(out), "--run-name", "qhead",
         *data], capture_output=True, text=True, env=env, timeout=1800)
    return request.param, save / "r_final.pth", out, r


def test_the_head_trainer_trains_one_head_per_size(head_run):
    source, _, out, r = head_run
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "the checkpoint names the mean/std scaling" in r.stdout
    assert "Head bank (#412)" in r.stdout
    sd = torch.load(out / "qhead_final.pth", map_location="cpu",
                    weights_only=True)
    assert head_bank_sizes(sd) == SIZES
    for size in SIZES:
        assert sd[f"heads.{size}.forecast_head.weight"].shape[0] == Q * size
    rows = list(csv.DictReader(open(out / "qhead_losses.csv")))
    assert len(rows) == 3 and finite(rows, "loss")


def load_eval_module():
    pytest.importorskip("gluonts")
    pytest.importorskip("gift_eval")
    spec = importlib.util.spec_from_file_location("eval_412", EVAL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eval_args(module, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["eval", *EVAL_PROTOCOL, *extra])
    return module.parse_args()


def test_the_eval_forecasts_with_the_bank(head_run, monkeypatch):
    """The eval reads the sizes, the scaling and the zero padding from the
    checkpoint, and each config's head forecasts at its size."""
    import pandas as pd
    from src.norm import RevMeanStdNorm
    source, bb, out, r = head_run
    assert r.returncode == 0, r.stderr[-3000:]
    module = load_eval_module()
    args = eval_args(module, monkeypatch, "--backbone-path", str(bb),
                     "--head-path", str(out / "qhead_final.pth"),
                     "--encoder-source", source)
    backbone, bank = module.load_models(args, CPU)
    assert isinstance(bank, ForecastingHeadBank) and bank.sizes == SIZES
    assert backbone.multi_patch_sizes == SIZES
    assert isinstance(backbone.rev_norm, RevMeanStdNorm)
    assert args.context_pad == "zeros"
    head = bank.for_frequency(backbone, "H")
    assert head.patch_size == 32
    predictor = module.ContrastiveForecasterPredictor(
        backbone=backbone, head=head, prediction_length=48, device=CPU,
        strategy="B4", context_pad=args.context_pad)
    item = {"target": (50.0 + np.random.default_rng(0).standard_normal(700)
                       .cumsum()).astype(np.float32),
            "start": pd.Period("2020-01-01 00:00", freq="H")}
    forecast = predictor.predict_item(item)
    assert forecast.forecast_array.shape == (1 + Q, 48)
    assert np.isfinite(forecast.forecast_array).all()


def test_the_eval_refuses_a_bank_of_other_sizes(tmp_path, monkeypatch,
                                                full_run):
    save = full_run[0]
    path = tmp_path / "bank.pth"
    torch.save(bank_of((8, 16)).state_dict(), path)
    module = load_eval_module()
    args = eval_args(module, monkeypatch, "--backbone-path",
                     str(save / "r_final.pth"), "--head-path", str(path))
    with pytest.raises(SystemExit, match="the head bank holds the sizes"):
        module.load_models(args, CPU)
