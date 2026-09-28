"""Tests for #417: one patch encoder and one value head per patch size.

#415's value-space model reads every series in patches of 16 with one GRU
patch encoder. #417 gives it one GRU patch encoder and one value head per
patch size, 8 to 128, and picks the size of a series from its frequency, as
Moirai 1.0 does. The body, the normaliser, the embeddings, the data and the
recipe do not change. Five groups of guards follow.

1. The frequency table and its two rules: a range per frequency in
   training, one size per frequency at inference.
2. The model: each size has its own encoder and its own head, and the
   switch off leaves the model as it was.
3. The objective on a batch that mixes sizes.
4. The trainer flag: the refusals, a tiny run and its resume.
5. The eval: the size per frequency, and A2 on a multi-patch model.
"""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.forecasting_head as fh  # noqa: E402
from src.checkpoint import (  # noqa: E402
    load_backbone_from_checkpoint,
    multi_patch_sizes_of,
    prepare_backbone_state_dict,
)
from src.forecasting_head import (  # noqa: E402
    QUANTILE_LEVELS,
    ValueHeadForecaster,
    forecast_A2,
    median_quantile_index,
    multi_patch_value_objective,
    native_value_head,
    patch_value_targets,
    quantile_loss,
    value_patches_to_series,
    value_space_forward,
    value_space_objective,
)
from src.freq_embedding import FREQ_NAME_TO_ID  # noqa: E402
from src.models import ConfigurableModel  # noqa: E402
from src.patch_size import (  # noqa: E402
    PATCH_SIZES,
    check_patch_sizes,
    draw_patch_sizes,
    frequency_class,
    patch_size_choices,
)

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
EVAL_PY = (REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
           / "eval_gift_eval_official.py")

Q = len(QUANTILE_LEVELS)
SIZES = (8, 16, 32, 64, 128)
T_RAW = 512
# The frequency and seasonality embeddings widen every patch by 3 + 3.
TAIL = 6


def model_config(sizes=SIZES, **kw):
    """A small ConfigurableModel config that runs on the CPU in milliseconds."""
    cfg = dict(
        C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=2.0, activation="gelu", depthwise_conv=3, dropout=0.0,
        rev_norm_kind="ewma", rev_norm_span=16, num_encoder_layers=1,
        freq_emb_dim=3, seasonality_emb_dim=3, value_head_quantiles=Q,
        multi_patch_sizes=sizes)
    cfg.update(kw)
    return cfg


def tiny_model(sizes=SIZES, **kw):
    return ConfigurableModel(**model_config(sizes, **kw))


def batch(model, B=4, seed=0):
    """A normalised batch and its label ids."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T_RAW, 1, generator=gen).cumsum(1)
    ids = dict(freq_ids=torch.randint(0, 10, (B,), generator=gen),
               seasonality_ids=torch.randint(0, 10, (B,), generator=gen))
    return model.rev_norm(x, mode="norm"), ids


# ---------------------------------------------------------------------------
# 1. The frequency table
# ---------------------------------------------------------------------------

# Every frequency string of the 97 GIFT-Eval configs, in the spelling of
# pandas before 2.2 and in the spelling of pandas 2.2 and later.
GIFT_EVAL_FREQS = [
    ("A-DEC", "Y"), ("YE-DEC", "Y"), ("Y", "Y"), ("AS-JAN", "Y"),
    ("Q-DEC", "Q"), ("QE-DEC", "Q"), ("QS", "Q"),
    ("M", "M"), ("ME", "M"), ("MS", "M"),
    ("W-SUN", "W"), ("W-WED", "W"), ("W-THU", "W"), ("W-FRI", "W"),
    ("W-TUE", "W"), ("W", "W"),
    ("D", "D"), ("B", "B"),
    ("H", "H"), ("h", "H"), ("6H", "H"),
    ("15T", "T"), ("10T", "T"), ("5T", "T"), ("15min", "T"), ("min", "T"),
    ("10S", "S"), ("10s", "S"),
]


@pytest.mark.parametrize("freq,cls", GIFT_EVAL_FREQS)
def test_every_gift_eval_frequency_has_a_class(freq, cls):
    assert frequency_class(freq) == cls


def test_the_frequency_ids_map_to_their_class():
    """The trainer knows a sample's frequency only by its embedding id."""
    want = {"10s": "S", "1min": "T", "5min": "T", "10min": "T",
            "15min": "T", "30min": "T", "1h": "H", "1d": "D", "1w": "W"}
    for name, cls in want.items():
        assert frequency_class(FREQ_NAME_TO_ID[name]) == cls
    assert frequency_class(FREQ_NAME_TO_ID["unknown"]) is None


@pytest.mark.parametrize("freq", [None, 0, "unknown", "ms", "us", "", "?"])
def test_a_series_with_no_label_has_no_class(freq):
    assert frequency_class(freq) is None


# The owner's table: uni2ts DefaultPatchSizeConstraints, with Q and Y at 8.
TRAIN_TABLE = [
    ("10S", (64, 128)), ("5T", (32, 64, 128)), ("H", (32, 64)),
    ("D", (16, 32)), ("B", (16, 32)), ("W-SUN", (16, 32)),
    ("M", (8, 16, 32)), ("Q-DEC", (8,)), ("A-DEC", (8,)),
    (None, SIZES),
]


@pytest.mark.parametrize("freq,sizes", TRAIN_TABLE)
def test_training_draws_inside_the_moirai_range(freq, sizes):
    assert patch_size_choices(freq, SIZES, training=True) == sizes


INFERENCE_TABLE = [("A-DEC", 8), ("Q-DEC", 8), ("M", 16), ("W-SUN", 16),
                   ("D", 16), ("H", 32), ("15T", 64), ("10S", 128)]


@pytest.mark.parametrize("freq,size", INFERENCE_TABLE)
def test_inference_reads_one_size_per_frequency(freq, size):
    assert patch_size_choices(freq, SIZES, training=False) == (size,)


def test_inference_refuses_a_frequency_with_no_class():
    """An eval must never pick a size for a frequency it cannot read."""
    with pytest.raises(ValueError):
        patch_size_choices("unknown", SIZES, training=False)


def test_the_draw_is_uniform_over_the_range():
    torch.manual_seed(0)
    hourly = torch.full((4000,), FREQ_NAME_TO_ID["1h"])
    drawn = draw_patch_sizes(hourly, SIZES, 4000)
    assert set(drawn.tolist()) == {32, 64}
    assert 0.45 < (drawn == 32).float().mean().item() < 0.55
    unlabelled = draw_patch_sizes(None, SIZES, 4000)
    assert set(unlabelled.tolist()) == set(SIZES)
    for size in SIZES:
        assert 0.15 < (unlabelled == size).float().mean().item() < 0.25


def test_each_sample_draws_from_its_own_frequency():
    ids = torch.tensor([FREQ_NAME_TO_ID["10s"], FREQ_NAME_TO_ID["1d"]] * 500)
    drawn = draw_patch_sizes(ids, SIZES, 1000)
    assert set(drawn[0::2].tolist()) == {64, 128}
    assert set(drawn[1::2].tolist()) == {16, 32}


def test_the_draw_follows_the_seed():
    ids = torch.arange(10).repeat(20)
    torch.manual_seed(3)
    first = draw_patch_sizes(ids, SIZES, 200)
    torch.manual_seed(3)
    assert torch.equal(first, draw_patch_sizes(ids, SIZES, 200))


@pytest.mark.parametrize("sizes", [
    (16, 32, 64, 128),         # Q and Y have no size
    (8, 16, 32, 64),           # S reads at 128 at inference
    (8, 16, 32, 64, 128, 256),  # 256 is outside the table
    (8, 32, 64, 128),          # no base size 16
])
def test_a_size_set_that_leaves_a_frequency_out_is_refused(sizes):
    with pytest.raises(ValueError):
        check_patch_sizes(sizes, base_size=16)


def test_the_full_size_set_covers_every_frequency():
    check_patch_sizes(PATCH_SIZES, base_size=16)
    assert PATCH_SIZES == SIZES


# ---------------------------------------------------------------------------
# 2. The model
# ---------------------------------------------------------------------------

def test_each_size_has_its_own_encoder_and_value_head():
    model = tiny_model()
    sd = model.state_dict()
    for size in SIZES:
        assert sd[f"encoder.encoders.{size}.skip.weight"].shape == (16, size + TAIL)
        assert sd[f"value_heads.{size}.weight"].shape == (Q * size, 16)
    assert "encoder.skip.weight" not in sd and "value_head.weight" not in sd
    encoders = {id(e) for e in model.encoder.encoders.values()}
    heads = {id(h) for h in model.value_heads.values()}
    assert len(encoders) == len(heads) == len(SIZES)


@pytest.mark.parametrize("size", SIZES)
def test_a_forward_pass_works_at_every_size(size):
    torch.manual_seed(0)
    model = tiny_model()
    x_norm, ids = batch(model)
    f_lat, o_lat, v_hat = value_space_forward(model, x_norm, patch_size=size,
                                              **ids)
    T = T_RAW // size
    assert f_lat.shape == o_lat.shape == (4, T, 1, 16)
    assert v_hat.shape == (4, T, 1, Q, size)
    assert torch.isfinite(v_hat).all()


@pytest.mark.parametrize("size", [8, 128])
def test_a_size_trains_only_its_own_encoder_and_head(size):
    torch.manual_seed(0)
    model = tiny_model()
    x_norm, ids = batch(model)
    loss, _, _, _ = value_space_objective(model, x_norm, depth=1,
                                          patch_size=size, **ids)
    loss.backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None}
    for other in SIZES:
        mine = other == size
        assert any(n.startswith(f"encoder.encoders.{other}.") for n in grads) == mine
        assert any(n.startswith(f"value_heads.{other}.") for n in grads) == mine
    assert any(n.startswith("transformer.layers.") for n in grads)


def test_a_size_the_model_lacks_is_refused():
    x_norm, ids = batch(tiny_model())
    with pytest.raises(ValueError):
        value_space_forward(tiny_model(), x_norm, patch_size=24, **ids)
    single = tiny_model(sizes=())
    with pytest.raises(ValueError):
        value_space_forward(single, x_norm, patch_size=32, **ids)


def test_the_model_refuses_a_set_without_its_base_size():
    with pytest.raises(ValueError):
        tiny_model(sizes=(8, 32))


def test_the_model_refuses_patch_stats_with_several_sizes():
    """The patch statistics read the normaliser's whole batch, and a
    multi-patch batch runs one group per size."""
    with pytest.raises(ValueError):
        tiny_model(patch_stats_kind="diff")


def test_the_switch_off_leaves_the_model_unchanged():
    """No flag builds the model #415 trains: the same keys, the same weights
    for the same seed, and the same forecast."""
    cfg = model_config()
    cfg.pop("multi_patch_sizes")
    torch.manual_seed(7)
    old = ConfigurableModel(**cfg).eval()
    torch.manual_seed(7)
    new = tiny_model(sizes=()).eval()
    a, b = old.state_dict(), new.state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert "encoder.skip.weight" in b and "value_head.weight" in b
    x_norm, ids = batch(old)
    assert torch.equal(value_space_forward(old, x_norm, **ids)[2],
                       value_space_forward(new, x_norm, **ids)[2])


# ---------------------------------------------------------------------------
# 3. The objective on a mixed batch
# ---------------------------------------------------------------------------

def group_objective(model, x_norm, ids, rows, size, depth):
    """The plain objective on the rows of one size."""
    part = {k: v[rows] for k, v in ids.items()}
    return value_space_objective(model, x_norm[rows], depth=depth,
                                 patch_size=size, **part)


def weighted_groups(model, x_norm, ids, depth):
    """The loss and the per-depth terms of the batch [8, 32, 32, 128],
    built by hand from one plain objective per group."""
    groups = [(torch.tensor([0]), 8), (torch.tensor([1, 2]), 32),
              (torch.tensor([3]), 128)]
    parts = [(len(r) / 4, group_objective(model, x_norm, ids, r, s, depth))
             for r, s in groups]
    loss = sum(w * p[0] for w, p in parts)
    per_depth = [sum(w * p[3][j] for w, p in parts) for j in range(depth + 1)]
    return loss, per_depth


def test_a_mixed_batch_loss_is_the_mean_over_samples():
    """Each group's loss is the mean over its samples, so a weight of
    (group size / batch size) gives the mean over the whole batch."""
    torch.manual_seed(0)
    model = tiny_model().eval()
    x_norm, ids = batch(model)
    with torch.no_grad():
        loss, _, _, per_depth = multi_patch_value_objective(
            model, x_norm, torch.tensor([8, 32, 32, 128]), depth=2, **ids)
        want_loss, want_depths = weighted_groups(model, x_norm, ids, 2)
    assert torch.isfinite(loss)
    assert torch.allclose(loss, want_loss)
    for got, want in zip(per_depth, want_depths, strict=True):
        assert torch.allclose(got, want)


def test_a_batch_of_one_size_is_the_plain_objective():
    torch.manual_seed(0)
    model = tiny_model().eval()
    x_norm, ids = batch(model)
    with torch.no_grad():
        mixed = multi_patch_value_objective(
            model, x_norm, torch.full((4,), 64), depth=1, **ids)
        plain = value_space_objective(model, x_norm, depth=1, patch_size=64,
                                      **ids)
    assert torch.allclose(mixed[0], plain[0])
    assert torch.allclose(mixed[1], plain[1])


@pytest.mark.parametrize("size", [8, 128])
def test_the_rollout_advances_one_patch_of_its_size_per_depth(size):
    """Depth 1 re-reads the median forecast of depth 0, laid out P values
    per patch, and is supervised against the patch P values further on."""
    torch.manual_seed(0)
    model = tiny_model().eval()
    x_norm, ids = batch(model)
    with torch.no_grad():
        _, _, _, per_depth = value_space_objective(
            model, x_norm, depth=1, patch_size=size, **ids)
        v0 = value_space_forward(model, x_norm, patch_size=size, **ids)[2]
        rolled = value_patches_to_series(v0, median_quantile_index())
        v1 = value_space_forward(model, rolled, patch_size=size, **ids)[2]
        targets, t_valid = patch_value_targets(x_norm, size, shift=1)
    assert t_valid == T_RAW // size - 2
    assert torch.allclose(per_depth[1], quantile_loss(v1[:, :t_valid], targets))


def test_the_diagnostics_read_the_base_size_or_the_largest_group():
    torch.manual_seed(0)
    model = tiny_model().eval()
    x_norm, ids = batch(model)
    with torch.no_grad():
        with_base = multi_patch_value_objective(
            model, x_norm, torch.tensor([8, 8, 8, 16]), **ids)
        without = multi_patch_value_objective(
            model, x_norm, torch.tensor([8, 32, 32, 128]), **ids)
    assert with_base[1].shape[:2] == (1, T_RAW // 16)
    assert without[1].shape[:2] == (2, T_RAW // 32)


def test_every_size_in_the_batch_takes_gradient():
    torch.manual_seed(0)
    model = tiny_model()
    x_norm, ids = batch(model)
    loss, _, _, _ = multi_patch_value_objective(
        model, x_norm, torch.tensor([8, 32, 32, 128]), depth=1, **ids)
    loss.backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None}
    for size in SIZES:
        present = size in (8, 32, 128)
        assert any(n.startswith(f"value_heads.{size}.") for n in grads) == present
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)


# ---------------------------------------------------------------------------
# 4. The trainer
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_py():
    spec = importlib.util.spec_from_file_location("train_py_417", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_trainer(*extra, timeout=900):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.run(
        [sys.executable, str(TRAIN_PY), "--device", "cpu",
         "--weight-decay", "0.1", *extra],
        capture_output=True, text=True, env=env, timeout=timeout)


# The smallest multi-patch run the CPU trains in seconds. The periodic synth
# labels its rows with frequency ids, so the draw reads real ranges, and
# mixup at p = 1 mixes the label embeddings of every step.
TINY_RUN = (
    "--value-space-objective", "--multi-patch-sizes", "8,16,32,64,128",
    "--t-raw", "512", "--n-channels", "1", "--d-model", "16",
    "--n-heads", "2", "--num-layers", "1", "--num-encoder-layers", "1",
    "--batch-size", "4", "--mix-ratio", "1.0", "--synth-kind", "periodic",
    "--freq-emb-dim", "3", "--seasonality-emb-dim", "3", "--mixup-p", "1.0",
    "--train-rollout-depth", "2", "--train-rollout-reduce", "sum",
    "--lr", "1e-3", "--lr-final", "0", "--lr-cosine-steps", "8",
    "--lr-warmup-steps", "2", "--grad-clip", "1.0",
    "--log-every", "1", "--save-every", "1000000")


def test_the_flag_is_on_the_value_space_allowlist(train_py):
    assert "multi_patch_sizes" in train_py.VALUE_SPACE_FLAGS
    clash = train_py.value_space_conflicts(
        ["--value-space-objective", "--weight-decay", "0.1",
         "--multi-patch-sizes", "8,16,32,64,128"])
    assert clash == []


def test_no_flag_gives_one_patch_size(train_py):
    args = train_py.parse_args(["--weight-decay", "0.1",
                                "--value-space-objective"])
    assert train_py.parse_multi_patch_sizes(args) == ()


def test_the_trainer_refuses_the_flag_without_the_value_objective(tmp_path):
    never = tmp_path / "never"
    r = run_trainer("--multi-patch-sizes", "8,16,32,64,128",
                    "--total-steps", "1", "--save-dir", str(never))
    assert r.returncode != 0
    assert "--value-space-objective" in r.stdout + r.stderr
    assert not never.exists()


@pytest.mark.parametrize("flags,words", [
    (["--multi-patch-sizes", "16,32,64,128"], "no training size"),
    (["--multi-patch-sizes", "8,16,32,64,128", "--t-raw", "256"], "at least"),
])
def test_the_trainer_refuses_sizes_it_cannot_train(tmp_path, flags, words):
    never = tmp_path / "never"
    r = run_trainer("--value-space-objective", "--total-steps", "1",
                    "--train-rollout-depth", "2", "--save-dir", str(never),
                    *flags)
    assert r.returncode != 0
    assert words in r.stdout + r.stderr
    assert not never.exists()


@pytest.fixture(scope="module")
def tiny_leg(tmp_path_factory):
    """Three steps of the tiny multi-patch run: its save dir and its result."""
    root = tmp_path_factory.mktemp("multi_patch_leg")
    result = run_trainer(*TINY_RUN, "--total-steps", "3",
                         "--save-dir", str(root), "--run-name", "mp")
    return root, result


def test_a_tiny_multi_patch_run_trains(tiny_leg):
    root, first = tiny_leg
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    assert "Multi-patch (#417)" in first.stdout
    assert "NaN/Inf DETECTED" not in first.stdout
    sd = torch.load(root / "mp_final.pth", map_location="cpu",
                    weights_only=True)
    assert multi_patch_sizes_of(sd) == SIZES
    assert all(f"value_heads.{s}.weight" in sd for s in SIZES)


def test_the_tiny_run_resumes_on_the_same_curve(tiny_leg):
    """The resume restores the weights of every size, the optimizer state
    and the Moirai schedule, and trains steps 4 and 5."""
    root, first = tiny_leg
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    resumed = run_trainer(*TINY_RUN, "--total-steps", "5", "--resume",
                          str(root / "mp_final.pth"), "--save-dir", str(root),
                          "--run-name", "mp")
    out = resumed.stdout
    assert resumed.returncode == 0, out[-3000:] + resumed.stderr[-3000:]
    assert "Restored optimizer" in out and "[lr] WARNING" not in out
    assert "[      4]" in out and "[      1]" not in out
    assert "NaN/Inf DETECTED" not in out


# ---------------------------------------------------------------------------
# 5. The eval
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq,size", INFERENCE_TABLE + [("5T", 64),
                                                         ("10T", 64)])
def test_the_eval_reads_each_frequency_at_its_size(freq, size):
    model = tiny_model().eval()
    head = native_value_head(model, freq)
    assert head.patch_size == head.forecast_len == size
    assert head.value_head is model.value_heads[str(size)]


def test_a_single_patch_model_keeps_its_one_head():
    """#415's checkpoints score as before, whatever the frequency."""
    torch.manual_seed(0)
    model = tiny_model(sizes=()).eval()
    head = native_value_head(model, "10S")
    assert head.patch_size is None and head.forecast_len == 16
    assert head.value_head is model.value_head
    ctx = torch.randn(T_RAW, 1).cumsum(0)
    before = forecast_A2(model, ValueHeadForecaster(model), ctx, 40, "cpu")
    after = forecast_A2(model, head, ctx, 40, "cpu")
    assert (before == after).all()


@pytest.mark.parametrize("freq,size", [("A-DEC", 8), ("H", 32), ("10S", 128)])
def test_a2_on_a_multi_patch_model_steps_by_its_size(monkeypatch, freq, size):
    seen = []
    real = fh.extract_forecaster_latents

    def spy(backbone, x, **kw):
        seen.append(kw.get("patch_size"))
        return real(backbone, x, **kw)

    monkeypatch.setattr(fh, "extract_forecaster_latents", spy)
    torch.manual_seed(0)
    model = tiny_model().eval()
    out = forecast_A2(model, native_value_head(model, freq),
                      torch.randn(T_RAW, 1).cumsum(0), 40, "cpu")
    assert out.shape == (Q, 40, 1)
    assert torch.isfinite(torch.as_tensor(out)).all()
    assert seen == [size] * math.ceil(40 / size)


def test_the_checkpoint_names_its_patch_sizes():
    assert multi_patch_sizes_of(tiny_model().state_dict()) == SIZES
    assert multi_patch_sizes_of(tiny_model(sizes=()).state_dict()) == ()


def test_a_multi_patch_checkpoint_loads_strictly():
    """The native eval keeps the value heads; every other reader drops them
    and still loads the encoder bank."""
    sd = tiny_model().state_dict()
    kept = prepare_backbone_state_dict(sd, keep_value_head=True)
    tiny_model().load_state_dict(kept)
    dropped = prepare_backbone_state_dict(sd)
    assert not any(k.startswith("value_head") for k in dropped)
    tiny_model(value_head_quantiles=0).load_state_dict(dropped)


def test_the_backbone_loader_builds_the_encoder_bank(tmp_path):
    path = tmp_path / "bb.pth"
    torch.save(tiny_model(ffn_mult=4.0).state_dict(), path)
    backbone, cfg = load_backbone_from_checkpoint(
        str(path), "cpu", C=1, H=16, W=16, nhead=2, num_layers=1,
        rev_norm_span=16)
    assert cfg["multi_patch_sizes"] == SIZES
    assert backbone.multi_patch_sizes == SIZES


def load_eval_module():
    """The GIFT-Eval script as a fresh module: its config is a global."""
    pytest.importorskip("gluonts")
    pytest.importorskip("gift_eval")
    spec = importlib.util.spec_from_file_location("eval_417", EVAL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eval_args(module, monkeypatch, backbone, *extra):
    monkeypatch.setattr(sys, "argv", [
        "eval", "--backbone-path", str(backbone), "--strategy", "A2",
        "--device", "cpu", "--t-raw", "4096", "--n-channels", "1",
        "--d-model", "16", "--n-heads", "2", "--num-layers", "1",
        "--encoder-type", "gru", "--rev-norm-kind", "ewma",
        "--rev-norm-span", "128", *extra])
    return module.parse_args()


def test_the_eval_script_scores_a_multi_patch_checkpoint(tmp_path,
                                                         monkeypatch):
    path = tmp_path / "bb.pth"
    torch.save(tiny_model(ffn_mult=4.0).state_dict(), path)
    module = load_eval_module()
    args = eval_args(module, monkeypatch, path, "--native-value-head")
    backbone, head = module.load_models(args, torch.device("cpu"))
    assert backbone.multi_patch_sizes == SIZES
    assert isinstance(head, ValueHeadForecaster)
    assert native_value_head(backbone, "H").patch_size == 32


def test_the_eval_script_refuses_a_multi_patch_checkpoint_without_its_heads(
        tmp_path, monkeypatch):
    path = tmp_path / "bb.pth"
    torch.save(tiny_model(ffn_mult=4.0).state_dict(), path)
    module = load_eval_module()
    args = eval_args(module, monkeypatch, path, "--head-path",
                     str(tmp_path / "head.pth"))
    with pytest.raises(SystemExit):
        module.load_models(args, torch.device("cpu"))
