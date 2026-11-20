"""Tests of the joint teacher fraction: ``--loss-shape
cosine_similarity_batch_joint_teacher`` and ``src.loss.joint_teacher_term``.

The shape replaces L_align and L_rep with one InfoNCE fraction for each
rollout depth j. The forecast pair (f^(j)_t, hT_{t+1+j}) and the
representation pair (h_t, hT_t) are both in the numerator and in the
denominator. hT is the EMA teacher. It gives every h key and target.

1. The value equals a reference with plain loops over (b, t, c) and the
   keys, without padding, for each reduction of the depths.
2. With left padding of a different length on each row, the value equals
   the reference that skips the padded anchors, keys and positives. The
   latents of the padded positions do not move the value.
3. No gradient reaches the teacher. Gradients reach f, each rollout depth
   and h, and they agree with finite differences.
4. The trainer refuses at start-up what the shape cannot run.
5. Three steps of train.py on the CPU, with and without padding.
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.loss import (JOINT_TEACHER_SHAPE,  # noqa: E402
                      contrastive_latent_loss, joint_teacher_term)

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
B, T, C, H = 3, 6, 2, 4
DEPTH = 2
REDUCTIONS = ("sum", "mean", "halving", "harmonic")
# Patches of left padding of each (row, channel). At t = 0 every row of
# channel 0 is padding, so no real anchor holds that (t, c).
PAD_LENGTHS = ((1, 0), (2, 3), (4, 1))


def spec(tau=1.0, **extra):
    """The loss keys of the trainer's LOSS_SPEC for the shape."""
    cfg = {"loss_shape": JOINT_TEACHER_SHAPE,
           "contrastive_divergence_temperature": tau}
    cfg.update(extra)
    return SimpleNamespace(train_configuration=cfg)


def latents(seed=0, depth=DEPTH, grad=False):
    """``(f, h, teacher, rollout)`` in float64: independent draws, so a
    term that reads the wrong tensor shows."""
    g = torch.Generator().manual_seed(seed)

    def draw():
        return torch.randn(B, T, C, H, generator=g, dtype=torch.float64,
                           requires_grad=grad)
    return draw(), draw(), draw(), [draw() for _ in range(depth)]


def left_padding(lengths=PAD_LENGTHS):
    pad = torch.zeros(B, T, C, dtype=torch.bool)
    for b, c in itertools.product(range(B), range(C)):
        pad[b, :lengths[b][c], c] = True
    return pad


def no_padding():
    return torch.zeros(B, T, C, dtype=torch.bool)


# ---------------------------------------------------------------------------
# The reference: plain loops over (b, t, c) and the keys
# ---------------------------------------------------------------------------

def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


def negatives(f, h, hT, real, j, b, t, c):
    """The sum of exp(cos / tau) over the negatives of the anchor
    (b, t, c) of depth j, as cosines. A padded key is not a negative."""
    sims = [cos(h[b, t, c], hT[b, t, k])                       # xx
            for k in range(C) if k != c and real(b, t, k)]
    sims += [cos(h[b, t, c], hT[b, l, c])                      # hh_all
             for l in range(T) if l != t and real(b, l, c)]
    sims += [cos(h[b, t, c], hT[o, l, c])                      # xs_allt
             for o in range(B) for l in range(T)
             if o != b and real(o, l, c)]
    sims += [cos(f[b, t, c], hT[o, t + 1 + j, c])              # cross-batch
             for o in range(B) if o != b and real(o, t + 1 + j, c)]
    sims += [cos(f[b, t + 1, k], f[b, t, c])                   # next forecast
             for k in range(C) if real(b, t + 1, k)]
    return sims


def reference_copy(f, h, hT, pad, j, tau):
    """L_j of the forecast f = f^(j), with plain loops."""
    def real(b, t, c):
        return not bool(pad[b, t, c])

    def anchor(b, t, c):
        return real(b, t, c) and real(b, t + 1 + j, c)

    values = []
    for t, c in itertools.product(range(T - 1 - j), range(C)):
        total = sum(math.exp(s / tau) for b in range(B) if anchor(b, t, c)
                    for s in negatives(f, h, hT, real, j, b, t, c))
        for b in (b for b in range(B) if anchor(b, t, c)):
            pos = (math.exp(cos(f[b, t, c], hT[b, t + 1 + j, c]) / tau)
                   + math.exp(cos(h[b, t, c], hT[b, t, c]) / tau))
            values.append(math.log(pos + total) - math.log(pos))
    return sum(values) / len(values)


def reference_loss(f, h, hT, rollout, pad, tau, reduce):
    """The copies of every depth that enters, combined by ``reduce``."""
    weight = {"sum": lambda j: 1.0, "mean": lambda j: 1.0,
              "halving": lambda j: 0.5 ** j,
              "harmonic": lambda j: 1.0 / (j + 1)}[reduce]
    copies = [(j, reference_copy(f_j, h, hT, pad, j, tau))
              for j, f_j in enumerate([f, *rollout]) if f_j is not None]
    total = sum(weight(j) * value for j, value in copies)
    return total / len(copies) if reduce == "mean" else total


def loss(f, h, hT, rollout, pad, tau, reduce, term_out=None):
    return joint_teacher_term(f, h, spec(tau), pad,
                              teacher_original_latent=hT,
                              rollout_latents=rollout, depth_reduce=reduce,
                              term_out=term_out)


# ---------------------------------------------------------------------------
# 1. The value equals the reference, without padding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reduce", REDUCTIONS)
@pytest.mark.parametrize("tau", [1.0, 0.5])
def test_the_value_equals_the_reference_without_padding(tau, reduce):
    f, h, hT, rollout = latents()
    terms = {}
    got = loss(f, h, hT, rollout, None, tau, reduce, terms)
    want = reference_loss(f, h, hT, rollout, no_padding(), tau, reduce)
    assert got.item() == pytest.approx(want, rel=1e-10)
    assert terms["l_joint"] == pytest.approx(
        reference_copy(f, h, hT, no_padding(), 0, tau), rel=1e-10)


def test_the_reduction_comes_from_the_config_key_without_an_argument():
    f, h, hT, rollout = latents()
    got = joint_teacher_term(
        f, h, spec(0.5, train_rollout_reduce="harmonic"), None,
        teacher_original_latent=hT, rollout_latents=rollout)
    want = reference_loss(f, h, hT, rollout, no_padding(), 0.5, "harmonic")
    assert got.item() == pytest.approx(want, rel=1e-10)


def test_depth_zero_is_the_l_joint_term_at_weight_one():
    f, h, hT, _ = latents()
    terms = {}
    got = loss(f, h, hT, None, None, 0.5, "sum", terms)
    assert got.item() == terms["l_joint"]
    assert got.item() == pytest.approx(
        reference_copy(f, h, hT, no_padding(), 0, 0.5), rel=1e-10)


@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_a_depth_left_out_builds_no_copy(reduce):
    """--train-rollout-distances 1,3: depth 1 is None."""
    f, h, hT, (_, f2) = latents()
    got = loss(f, h, hT, [None, f2], None, 1.0, reduce)
    want = reference_loss(f, h, hT, [None, f2], no_padding(), 1.0, reduce)
    assert got.item() == pytest.approx(want, rel=1e-10)


def test_a_place_weight_with_a_depth_left_out_raises():
    f, h, hT, (_, f2) = latents()
    with pytest.raises(ValueError, match="place in the list"):
        loss(f, h, hT, [None, f2], None, 1.0, "harmonic")


# ---------------------------------------------------------------------------
# 2. Left padding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reduce", ["sum", "harmonic"])
@pytest.mark.parametrize("tau", [1.0, 0.5])
def test_the_value_equals_the_reference_with_left_padding(tau, reduce):
    f, h, hT, rollout = latents(seed=1)
    pad = left_padding()
    got = loss(f, h, hT, rollout, pad, tau, reduce)
    want = reference_loss(f, h, hT, rollout, pad, tau, reduce)
    assert got.item() == pytest.approx(want, rel=1e-10)


def test_the_padded_positions_do_not_move_the_value():
    f, h, hT, rollout = latents(seed=2)
    pad = left_padding()
    want = loss(f, h, hT, rollout, pad, 0.5, "sum")
    noise = torch.Generator().manual_seed(9)

    def moved(x):
        fill = 50.0 * torch.randn(x.shape, generator=noise,
                                  dtype=x.dtype)
        return torch.where(pad.unsqueeze(-1), fill, x)
    got = loss(moved(f), moved(h), moved(hT), [moved(r) for r in rollout],
               pad, 0.5, "sum")
    assert torch.equal(got, want)


def test_an_all_false_mask_is_no_padding():
    f, h, hT, rollout = latents(seed=3)
    assert torch.equal(loss(f, h, hT, rollout, None, 0.5, "sum"),
                       loss(f, h, hT, rollout, no_padding(), 0.5, "sum"))


# ---------------------------------------------------------------------------
# 3. Gradients
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pad", [None, left_padding()],
                         ids=["no_padding", "left_padding"])
def test_no_gradient_reaches_the_teacher(pad):
    f, h, _, rollout = latents(seed=4, grad=True)
    source = torch.randn(B, T, C, H, dtype=torch.float64,
                         requires_grad=True)
    teacher = 2.0 * source                # a teacher on a graph
    loss(f, h, teacher, rollout, pad, 0.5, "sum").backward()
    assert source.grad is None
    for x in (f, h, *rollout):
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0


def test_the_gradients_match_finite_differences_with_padding():
    """At t = 0 every row of channel 0 is padding, so the pooled negatives
    of that (t, c) are -inf. The backward must stay finite there."""
    f, h, hT, rollout = latents(seed=6, grad=True)
    pad = left_padding()

    def value(f, h, *rollout):
        return loss(f, h, hT, list(rollout), pad, 0.5, "harmonic")
    assert torch.autograd.gradcheck(value, (f, h, *rollout))


def test_a_teacher_leaf_with_a_gradient_flag_gets_none():
    f, h, hT, rollout = latents(seed=5, grad=True)
    loss(f, h, hT, rollout, None, 1.0, "harmonic").backward()
    assert hT.grad is None
    assert all(r.grad.abs().sum() > 0 for r in rollout)


# ---------------------------------------------------------------------------
# The rules of the function
# ---------------------------------------------------------------------------

def test_no_teacher_raises():
    f, h, _, _ = latents()
    with pytest.raises(ValueError, match="EMA teacher"):
        joint_teacher_term(f, h, spec(), None, teacher_original_latent=None)


def test_a_second_tau_raises_and_the_same_tau_does_not():
    f, h, hT, _ = latents()
    with pytest.raises(ValueError, match="one tau"):
        joint_teacher_term(
            f, h, spec(0.5, contrastive_divergence_temperature_rep=1.0),
            None, teacher_original_latent=hT)
    same = joint_teacher_term(
        f, h, spec(0.5, contrastive_divergence_temperature_rep=0.5), None,
        teacher_original_latent=hT)
    assert same.item() == pytest.approx(
        reference_copy(f, h, hT, no_padding(), 0, 0.5), rel=1e-10)


def test_tau_override_wins_over_the_config_key():
    f, h, hT, _ = latents()
    got = joint_teacher_term(f, h, spec(1.0), None,
                             teacher_original_latent=hT, tau_override=0.5)
    assert got.item() == pytest.approx(
        reference_copy(f, h, hT, no_padding(), 0, 0.5), rel=1e-10)


def test_a_depth_without_an_anchor_raises():
    f, h, hT, rollout = latents(depth=T - 1)
    with pytest.raises(ValueError, match="time steps"):
        loss(f, h, hT, rollout, None, 1.0, "sum")


def test_contrastive_latent_loss_points_to_the_joint_function():
    f, h, hT, _ = latents()
    with pytest.raises(NotImplementedError, match="joint_teacher_term"):
        contrastive_latent_loss((f, h), False, spec(),
                                teacher_original_latent=hT)


# ---------------------------------------------------------------------------
# 4. The start-up refusals of train.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_py():
    s = importlib.util.spec_from_file_location("train_py_joint", TRAIN_PY)
    module = importlib.util.module_from_spec(s)
    s.loader.exec_module(module)
    return module


SHAPE = ("--weight-decay", "0.1", "--loss-shape", JOINT_TEACHER_SHAPE)
BASE = SHAPE + ("--ema-embedding", "--ema-encoder")

# id, command line, environment, the words of the refusal.
REFUSALS = [
    ("no_teacher", SHAPE, {}, "no EMA teacher"),
    ("multi_patch", BASE + ("--multi-patch-sizes", "8,16"), {},
     "--multi-patch-sizes"),
    ("ddp", BASE, {"WORLD_SIZE": "2"}, "WORLD_SIZE > 1"),
    ("floor", BASE + ("--subtract-contrastive-floor",), {},
     "--subtract-contrastive-floor"),
    ("pos_in_denom", BASE + ("--pos-in-denominator",), {},
     "--pos-in-denominator"),
    ("moco_negatives", BASE + ("--moco-negatives",), {}, "--moco-negatives"),
    ("moco_rep_keys", BASE + ("--moco-rep-keys",), {}, "--moco-rep-keys"),
    ("rep_weight", BASE + ("--rep-loss-weight", "0.5"), {},
     "--rep-loss-weight other than 1"),
    ("rep_schedule", BASE + ("--rep-loss-weight-end", "0.0"), {},
     "--rep-loss-weight-end"),
    ("pred_weight", BASE + ("--pred-loss-weight", "0.5"), {},
     "--pred-loss-weight other than 1"),
    ("tau_rep", BASE + ("--tau", "0.5", "--tau-rep", "1.0"), {}, "--tau-rep"),
    ("tau_rep_vs_default", BASE + ("--tau-rep", "1.0"), {}, "--tau-rep"),
    ("cpc", BASE + ("--forecaster-kind", "cpc"), {}, "--forecaster-kind"),
    ("fused", BASE, {"XSHH_ALLT_FUSED": "1"}, "XSHH_ALLT_FUSED"),
    ("shard", BASE, {"XSHH_ALLT_SHARD": "1"}, "XSHH_ALLT_SHARD"),
]


def clean_env(monkeypatch, env=None):
    for key in ("WORLD_SIZE", "XSHH_ALLT_FUSED", "XSHH_ALLT_SHARD"):
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize("flags,env,needle", [r[1:] for r in REFUSALS],
                         ids=[r[0] for r in REFUSALS])
def test_the_trainer_refuses(train_py, monkeypatch, flags, env, needle):
    clean_env(monkeypatch, env)
    with pytest.raises(SystemExit, match=re.escape(needle)):
        train_py.check_joint_teacher(train_py.parse_args(list(flags)))


@pytest.mark.parametrize("flags", [
    BASE, SHAPE + ("--ema-embedding",), SHAPE + ("--ema-encoder",),
    BASE + ("--tau", "0.5", "--tau-rep", "0.5"), BASE + ("--tau-rep", "0.07"),
    BASE + ("--rep-loss-weight", "1.0", "--pred-loss-weight", "1.0"),
    BASE + ("--align-loss-weight", "1.0", "--align-target", "teacher"),
    BASE + ("--gift-pretrain", "--skip-nan-samples")])
def test_the_trainer_takes(train_py, monkeypatch, flags):
    clean_env(monkeypatch)
    args = train_py.parse_args(list(flags))
    train_py.check_joint_teacher(args)
    assert train_py.gift_contrastive_gap(args) is None


def test_the_check_reads_only_the_joint_shape(train_py):
    args = train_py.parse_args(["--weight-decay", "0.1", "--loss-shape",
                                "cosine_similarity_batch_rep_only"])
    train_py.check_joint_teacher(args)


def test_the_shape_consumes_the_rollout_depths(train_py):
    args = train_py.parse_args([*BASE, "--align-loss-weight", "0",
                                "--train-rollout-depth", "2"])
    assert train_py.main_term_depth_gap(args) is None
    assert not train_py.rollout_depth_has_no_consumer(args)


def run_train(save_dir, *flags):
    """One run of the trainer on the CPU."""
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="4")
    for key in ("WORLD_SIZE", "XSHH_ALLT_FUSED", "XSHH_ALLT_SHARD",
                "CF_NAN_DEBUG"):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, "-u", str(TRAIN_PY), "--device", "cpu",
         "--weight-decay", "0.1", "--save-dir", str(save_dir),
         "--run-name", "r", *flags],
        capture_output=True, text=True, env=env, timeout=900)


def test_main_refuses_before_the_model(tmp_path):
    r = run_train(tmp_path, "--loss-shape", JOINT_TEACHER_SHAPE,
                  "--moco-negatives")
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert "no EMA teacher" in out and "--moco-negatives" in out
    assert "Device:" not in out


# ---------------------------------------------------------------------------
# 5. Three steps of train.py on the CPU
# ---------------------------------------------------------------------------

JOINT_RUN = ("--loss-shape", JOINT_TEACHER_SHAPE, "--ema-embedding",
             "--ema-encoder", "--ema-tau", "0.9", "--align-loss-weight", "0",
             "--train-rollout-depth", "2", "--train-rollout-reduce", "sum",
             "--total-steps", "3", "--log-every", "1",
             "--save-every", "1000000", "--no-latent-drift-probe",
             "--num-layers", "1", "--num-encoder-layers", "1",
             "--n-heads", "2", "--lr", "1e-3", "--seed", "42")
# The trainer's AUC diagnostic reads one channel only.
SYNTH = ("--mix-ratio", "1.0", "--synth-kind", "periodic",
         "--hf-repo", "none", "--hf-path", "none", "--t-raw", "256",
         "--n-channels", "1", "--d-model", "32", "--batch-size", "4",
         "--tau", "0.5")
GIFT = ("--gift-pretrain", "--freq-vocab", "v2", "--t-raw", "4096",
        "--n-channels", "1", "--d-model", "16", "--batch-size", "4",
        "--rev-norm-kind", "ewma", "--rev-norm-span", "128",
        "--synth-kind", "forked-arma", "--mix-ratio", "0.25",
        "--freq-emb-dim", "3", "--seasonality-emb-dim", "3", "--tau", "1.0")


def gift_corpus(root):
    """The #419 test corpus, with short series that the stream pads."""
    pytest.importorskip("pyarrow")
    from tests.test_419_gift_pretrain import build_corpus, write_index
    folder, index, _ = build_corpus(root / "corpus")
    return ("--gift-pretrain-root", str(folder),
            "--gift-pretrain-index", str(write_index(root, index)))


JOINT_COPIES = ["l_joint_d0", "l_joint_d1", "l_joint_d2"]


def joint_rows(save_dir):
    """The rows of the joint run. ``l_joint`` comes last before the copy of
    each depth."""
    with open(save_dir / "r_losses.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames[-4:] == ["l_joint", *JOINT_COPIES]
        return list(reader)


def check_joint_rows(rows):
    """Three steps of the joint run: the term, its copies, the reference and
    the per-depth errors are finite, and the other term columns are blank."""
    assert [r["step"] for r in rows] == ["1", "2", "3"]
    for r in rows:
        values = [float(r[k]) for k in ("loss", "l_joint", "loss_tau_ref",
                                        "cos_err_d0", "cos_err_d1",
                                        "cos_err_d2", *JOINT_COPIES)]
        assert all(math.isfinite(v) for v in values)
        # The sum of three copies, each 0 or more, holds the depth-0 copy.
        assert float(r["loss"]) >= float(r["l_joint"]) > 0.0
        assert r["l_joint"] == r["l_joint_d0"]
        assert float(r["loss"]) == pytest.approx(
            sum(float(r[c]) for c in JOINT_COPIES), rel=1e-5)
        assert r["l_rep"] == r["l_pred"] == r["l_align"] == r["rep_w"] == ""


def test_three_steps_on_the_synthetic_stream(tmp_path):
    r = run_train(tmp_path, *JOINT_RUN, *SYNTH)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "NaN/Inf DETECTED" not in r.stdout
    check_joint_rows(joint_rows(tmp_path))


@pytest.mark.parametrize("extra", [(), ("--skip-nan-samples",)],
                         ids=["padded", "padded_by_row"])
def test_three_steps_on_the_padded_stream(tmp_path, extra):
    r = run_train(tmp_path / "save", *JOINT_RUN, *GIFT, *extra,
                  *gift_corpus(tmp_path))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "Salesforce/GiftEvalPretrain (#419)" in r.stdout
    assert "NaN/Inf DETECTED" not in r.stdout
    check_joint_rows(joint_rows(tmp_path / "save"))
