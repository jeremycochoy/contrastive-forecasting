"""Three opt-in training changes, each off by default:

1. SIGReg on f (--sigreg-forecast) reads f before the latent sphere
   (--latent-sphere), as it reads e and h. TransformerBlock.forward returns
   that raw output under ``return_raw_forecast``.
2. --ema-tau-shape log: α = start + (end − start)·log(1 + t)/log(1 + span).
3. --align-loss-kind l2_raw: L_align = (1/K)·Σ_k (f_k − h_k)² on f before the
   sphere and the target as it is, at depth 0 and at each rollout depth.
"""
import csv
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.forecasting_head import rollout_forecaster_latents  # noqa: E402
from src.loss import align_loss, masked_mean  # noqa: E402
from src.models import (SCHEDULE_SHAPES, ConfigurableModel,  # noqa: E402
                        ema_tau_at_step, log_schedule_at_step)
from tests.test_ema_tau_cosine import run_train  # noqa: E402
from tests.test_latent_sphere import encoder_input, model  # noqa: E402

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
BASE_COMMIT = "2713097d"


@pytest.fixture(scope="module")
def train():
    spec = importlib.util.spec_from_file_location("train_raw_forecast", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def norms(x):
    return x.float().norm(dim=-1)


# --- 1. the raw forecaster output ------------------------------------------


def test_f_is_the_same_with_and_without_the_raw_output():
    net = model(latent_sphere=True)
    with torch.no_grad():
        f, h = net.transformer(encoder_input(net))
        f2, h2, e, raw = net.transformer(encoder_input(net), return_embed=True,
                                         return_raw_forecast=True)
    assert torch.equal(f, f2) and torch.equal(h, h2)
    assert torch.equal(net.transformer.sphere(raw), f)
    assert not torch.allclose(norms(raw), norms(f))


def test_without_the_sphere_the_raw_output_is_f():
    net = model()
    with torch.no_grad():
        f, _, raw = net.transformer(encoder_input(net), return_raw_forecast=True)
    assert torch.equal(raw, f)


def test_the_default_tuples_keep_their_length():
    net = model(latent_sphere=True)
    with torch.no_grad():
        assert len(net.transformer(encoder_input(net))) == 2
        assert len(net.transformer(encoder_input(net), return_embed=True)) == 3


def test_the_rollout_feeds_back_the_same_copies_and_returns_the_raw_ones():
    net = model(latent_sphere=True)
    f0 = torch.randn(2, 8, 1, 16)
    with torch.no_grad():
        plain = rollout_forecaster_latents(net, f0, 3)
        depths, raw = rollout_forecaster_latents(net, f0, 3, return_raw=True)
    for a, b, r in zip(plain, depths, raw):
        assert torch.equal(a, b)
        assert torch.equal(net.transformer.sphere(r), b)
        assert not torch.allclose(norms(r), norms(b))


def test_the_sigreg_f_term_reads_the_raw_output(train, monkeypatch):
    """The term is the statistic of f_raw, with the projections of one seed."""
    args = train.parse_args(["--weight-decay", "0.1", "--sigreg-forecast",
                             "--sigreg-forecast-weight", "1.0"])
    monkeypatch.setattr(train, "main_contrastive_term",
                        lambda lat, args, rep_w, tau, pad, terms: lat.o.new_zeros(()))
    torch.manual_seed(0)
    raw = 5.0 * torch.randn(4, 6, 1, 8)
    lat = SimpleNamespace(f=F.normalize(raw, dim=-1) * math.sqrt(8), f_raw=raw,
                          o=torch.randn(4, 6, 1, 8), teacher=None, e=None,
                          rollout=[])
    torch.manual_seed(1)
    _, values = train.contrastive_terms(None, lat, args, 0.0, None, None)
    torch.manual_seed(1)
    want_raw = train.sigreg_term(raw, args, None).item()
    torch.manual_seed(1)
    on_sphere = train.sigreg_term(lat.f, args, None).item()
    assert values["sigreg_f"] == pytest.approx(want_raw)
    assert values["sigreg_f"] != pytest.approx(on_sphere)


def test_the_training_latents_carry_the_raw_output(train):
    args = train.parse_args(["--weight-decay", "0.1", "--sigreg-forecast"])
    net = model(latent_sphere=True)
    x = torch.randn(2, 64, 1, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        lat = train.group_forward(net, net.rev_norm(x, mode="norm"), {}, 8, args)
    assert torch.equal(net.transformer.sphere(lat.f_raw), lat.f)
    grid = train.common_grid({8: lat}, {8: None})
    assert torch.equal(grid.f_raw, lat.f_raw)


# --- 2. the log shape of the EMA schedule ----------------------------------


START, END, SPAN = 0.9, 1.0, 99  # log1p(9) = log1p(99) / 2


def test_the_log_shape_ends_and_midpoint():
    at = lambda step: log_schedule_at_step(step, 1000, START, END, SPAN)  # noqa: E731
    assert at(0) == START
    assert at(9) == pytest.approx(0.95, rel=1e-12)
    assert at(SPAN) == END
    assert all(at(step) == END for step in (SPAN + 1, 500, 5000))
    assert at(-3) == START
    assert at(30) == pytest.approx(START + 0.1 * math.log1p(30) / math.log1p(SPAN))


def test_the_log_shape_follows_the_conventions_of_the_other_shapes():
    assert "log" in SCHEDULE_SHAPES
    assert log_schedule_at_step(50, 100, START, None) == START
    assert log_schedule_at_step(0, 0, START, END) == END
    assert log_schedule_at_step(9, SPAN, START, END) == pytest.approx(0.95)
    assert ema_tau_at_step(9, 1000, START, END, SPAN, shape="log") == \
        log_schedule_at_step(9, 1000, START, END, SPAN)


def test_the_log_shape_without_an_end_value_is_refused(tmp_path):
    res, save_dir = run_train(tmp_path, "lognoend", ["--ema-tau-shape", "log"])
    out = res.stdout + res.stderr
    assert res.returncode != 0
    assert "--ema-tau-shape log" in out and "--ema-tau-end" in out


# --- 3. the l2_raw L_align -------------------------------------------------


def align_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    f = 3.0 * torch.randn(2, 5, 1, 4, generator=g)
    h = torch.randn(2, 5, 1, 4, generator=g)
    depth1 = 2.0 * torch.randn(2, 5, 1, 4, generator=g)
    return f, h, depth1


def hand_l2(f, target, keep=None):
    """(1/K)·Σ_k (f_t,k − target_{t+1,k})², averaged over the pairs."""
    pairs = ((f[:, :-1] - target[:, 1:]) ** 2).sum(-1) / f.shape[-1]
    return pairs.mean() if keep is None else pairs[keep].mean()


def test_l2_raw_at_depth_zero_is_the_hand_formula():
    f, h, _ = align_inputs()
    got = align_loss(f, h, 0.5, kind="l2_raw")
    assert got.item() == pytest.approx(0.5 * hand_l2(f, h).item(), rel=1e-6)


def test_l2_raw_at_depth_one_reads_the_target_two_patches_ahead():
    f, h, f1 = align_inputs()
    teacher = h + 0.3
    got = align_loss(f, h, 2.0, target_latent=teacher, rollout_latents=[f1],
                     kind="l2_raw")
    depth1 = hand_l2(f1[:, :4], teacher[:, 1:])  # f^(1)_t against target_{t+2}
    want = 2.0 * hand_l2(f, teacher) + 2.0 * depth1
    assert got.item() == pytest.approx(want.item(), rel=1e-6)
    mean = align_loss(f, h, 2.0, target_latent=teacher, rollout_latents=[f1],
                      depth_reduce="mean", kind="l2_raw")
    assert mean.item() == pytest.approx(want.item() / 2, rel=1e-6)


def test_l2_raw_skips_the_padded_anchors():
    f, h, _ = align_inputs()
    pad = torch.zeros(2, 5, 1, dtype=torch.bool)
    pad[0, :2] = True
    got = align_loss(f, h, 1.0, pad_patches=pad, kind="l2_raw")
    keep = ~pad[:, :-1]
    assert got.item() == pytest.approx(hand_l2(f, h, keep).item(), rel=1e-6)


def test_l2_raw_records_its_depth_zero_value_and_stops_the_target_gradient():
    f, h, f1 = align_inputs()
    f.requires_grad_(True)
    h.requires_grad_(True)
    terms = {}
    align_loss(f, h, 3.0, rollout_latents=[f1], term_out=terms,
               kind="l2_raw").backward()
    assert terms["l_align"] == pytest.approx(hand_l2(f, h).item(), rel=1e-6)
    assert h.grad is None and f.grad.abs().sum() > 0


def old_align_loss(f, target, weight, pad=None):
    """The depth-0 cosine L_align before --align-loss-kind, as written then."""
    fore_norm = F.normalize(f, p=2, dim=-1)
    targ_norm = F.normalize(target, p=2, dim=-1)
    cos = (fore_norm[:, :-1] * targ_norm[:, 1:].detach()).sum(dim=-1)
    pairs = 2.0 - 2.0 * cos
    return weight * (pairs.mean() if pad is None else masked_mean(pairs, ~pad[:, :-1]))


def test_the_cosine_kind_keeps_the_old_values():
    f, h, _ = align_inputs()
    pad = torch.zeros(2, 5, 1, dtype=torch.bool)
    pad[1, :3] = True
    assert torch.equal(align_loss(f, h, 0.7), align_loss(f, h, 0.7, kind="cosine"))
    assert align_loss(f, h, 0.7).item() == pytest.approx(
        old_align_loss(f, h, 0.7).item(), rel=1e-6)
    assert align_loss(f, h, 0.7, pad_patches=pad).item() == pytest.approx(
        old_align_loss(f, h, 0.7, pad).item(), rel=1e-6)


def test_an_unknown_kind_is_refused():
    f, h, _ = align_inputs()
    with pytest.raises(ValueError, match="l1"):
        align_loss(f, h, kind="l1")


def test_the_l2_raw_flag_needs_the_align_term(train):
    assert train.parse_args(["--weight-decay", "0.1"]).align_loss_kind == "cosine"
    with pytest.raises(SystemExit):
        train.parse_args(["--weight-decay", "0.1", "--align-loss-kind", "l1"])


def test_l2_raw_runs_beside_the_main_loss_on_the_raw_latents(train, monkeypatch):
    """The main loss gets its cosine add-on off, and L_align reads f_raw and
    the raw depths."""
    args = train.parse_args(["--weight-decay", "0.1", "--align-loss-weight", "1.0",
                             "--align-loss-kind", "l2_raw"])
    seen = {}

    def main_loss(pair, **kwargs):
        seen["align_loss_weight"] = kwargs["align_loss_weight"]
        return torch.zeros(())
    monkeypatch.setattr(train, "contrastive_latent_loss", main_loss)
    f, h, f1 = align_inputs()
    lat = SimpleNamespace(f=F.normalize(f, dim=-1), f_raw=f, o=h, teacher=None,
                          rollout=[F.normalize(f1, dim=-1)], rollout_raw=[f1])
    args.train_rollout_depth = 1
    terms = {}
    loss = train.main_contrastive_term(lat, args, 1.0, None, None, terms)
    assert seen["align_loss_weight"] == 0.0
    want = align_loss(f, h, 1.0, rollout_latents=[f1], kind="l2_raw")
    assert loss.item() == pytest.approx(want.item(), rel=1e-6)
    assert terms["l_align"] == pytest.approx(hand_l2(f, h).item(), rel=1e-6)


# --- the default runs train as on the base commit --------------------------


def test_a_default_run_writes_the_losses_of_the_base_commit(tmp_path):
    from tests.losses_csv import rows_without_q_random
    from tests.test_412_ours_moirai import git_tree, run
    flags = ("--total-steps", "3", "--batch-size", "2", "--lr", "1e-3",
             "--weight-decay", "0.1", "--mix-ratio", "1.0", "--synth-kind",
             "periodic", "--t-raw", "64", "--n-channels", "1", "--d-model", "16",
             "--n-heads", "2", "--num-layers", "1", "--num-encoder-layers", "1",
             "--log-every", "1", "--save-every", "1000", "--seed", "42",
             "--hf-repo", "none", "--hf-path", "none",
             "--loss-shape", "cosine_similarity_batch_split_pred_rep",
             "--align-loss-weight", "1.0", "--align-target", "teacher",
             "--ema-embedding", "--ema-encoder", "--ema-tau", "0.9",
             "--ema-tau-end", "1.0", "--ema-tau-shape", "cosine",
             "--train-rollout-depth", "2", "--latent-sphere",
             "--sigreg-encoding")
    old = run(git_tree(BASE_COMMIT, tmp_path), tmp_path / "old", *flags)
    new = run(REPO_ROOT, tmp_path / "new", *flags)
    assert old.returncode == 0, old.stderr[-2000:]
    assert new.returncode == 0, new.stderr[-2000:]
    assert (rows_without_q_random(tmp_path / "new" / "r_losses.csv")
            == rows_without_q_random(tmp_path / "old" / "r_losses.csv"))
    assert len(list(csv.reader(open(tmp_path / "new" / "r_losses.csv")))) == 4
