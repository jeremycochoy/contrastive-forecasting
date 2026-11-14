"""--head-block-sizes: a bank of forecast heads by block size, on the trunk
of one transformer quantile head (src/block_head.py).

Head P decodes k = P / 16 consecutive latents together into the P values of
their block. The frequency of a row selects P, as Moirai picks its inference
patch size. The backbone is unchanged. Groups, all on the CPU:

1. The block size of each frequency, and the size list.
2. The head: its keys, k = 1 against the standard head on the same weights,
   the blocks of each size, the loss and the gradients of a row.
3. The default path: the standard head, the trainer and the eval with no
   flag.
4. The head trainer: a bank trains for a few steps on the CPU and its loss
   is finite, a bank of 16 writes the loss rows of the standard head, the
   shared trainer, and the refusals.
5. The scorer: whole blocks cut to the horizon, the per-window and the
   batched paths agree, and the eval loads the bank off its keys.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.forecasting_head as fh  # noqa: E402
from src.block_head import (BLOCK_SIZES, BlockHeadMember,  # noqa: E402
                            BlockSizeHead, block_head_loss, block_head_sizes,
                            block_size_of, block_targets, block_trunk_state,
                            check_block_sizes, row_block_sizes)
from src.forecasting_head import (QUANTILE_LEVELS,  # noqa: E402
                                  ForecastingHeadBank,
                                  LinearQuantileForecastingHead,
                                  TransformerQuantileForecastingHead,
                                  compute_valid_targets, decode_forecast_head,
                                  extract_encoder_latents,
                                  extract_forecaster_latents, forecast_B4,
                                  forecast_B4_windows, forecast_head_input,
                                  masked_quantile_loss, quantile_loss,
                                  rollout_latent, rollout_steps,
                                  valid_target_keep)
from src.freq_embedding import FREQ_NAMES_V2  # noqa: E402
from src.freq_family import FrequencyFamilyHead, family_layout_of  # noqa: E402
from src.models import ConfigurableModel  # noqa: E402
from src.patch_size import INFERENCE_SIZE  # noqa: E402

GIFT_SCRIPTS = REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
HEAD_PY = GIFT_SCRIPTS / "train_forecasting_head.py"
SHARED_PY = GIFT_SCRIPTS / "train_forecasting_heads_shared.py"
EVAL_PY = GIFT_SCRIPTS / "eval_gift_eval_official.py"
Q = len(QUANTILE_LEVELS)
W, H = 16, 16
T_RAW = 512                        # 32 patches
PATCHES = T_RAW // W
CPU = torch.device("cpu")
TOL = dict(rtol=1e-4, atol=1e-3)   # the tolerance of the batched B4 tests


def v2(name):
    """The id of a frequency name in the vocabulary v2."""
    return FREQ_NAMES_V2.index(name)


def tiny_backbone(**changes):
    """A backbone with one patch size, the EWMA, zero padding and the
    vocabulary v2, at d_model 16."""
    torch.manual_seed(0)
    config = dict(
        C=1, H=H, W=W, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        rev_norm_kind="ewma", rev_norm_span=128, num_encoder_layers=1,
        freq_emb_dim=3, seasonality_emb_dim=3, num_freqs=len(FREQ_NAMES_V2),
        rev_norm_skip_leading_zeros=True)
    return ConfigurableModel(**dict(config, **changes)).eval()


def standard_head(seed=None, dropout=0.0):
    if seed is not None:
        torch.manual_seed(seed)
    return TransformerQuantileForecastingHead(
        H=H, num_layers=2, nhead=2, forecast_len=W, dropout=dropout)


def bank_of(sizes=BLOCK_SIZES, seed=1, dropout=0.0):
    torch.manual_seed(seed)
    return BlockSizeHead(standard_head(dropout=dropout), sizes)


def batch(n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 50 + torch.randn(n, T_RAW, 1, generator=g).cumsum(1)


def layout(backbone, x, name="e_then_f"):
    """``(seq, mask, start, x_norm)`` of the head input of ``x``."""
    f_bc, x_norm = extract_forecaster_latents(backbone, x)
    seq, mask, start = forecast_head_input(backbone, x, f_bc, name)
    return seq, mask, start, x_norm


def n_params(module):
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# 1. The block size of a frequency
# ---------------------------------------------------------------------------

def test_the_bank_has_the_four_sizes_of_a_latent_or_more():
    assert BLOCK_SIZES == (16, 32, 64, 128)


@pytest.mark.parametrize("freq,size", [
    ("10S", 128), ("4S", 128), ("T", 64), ("5T", 64), ("15T", 64),
    ("10min", 64), ("H", 32), ("6H", 32), ("D", 16), ("B", 16),
    ("W-SUN", 16), ("W-FRI", 16), ("M", 16), ("MS", 16),
    # Q and Y read at 8 in Moirai: below one latent of 16 values.
    ("Q-DEC", 16), ("A-DEC", 16), ("YE-DEC", 16),
    # No frequency label.
    (None, 16), ("ms", 16)])
def test_each_frequency_has_one_block_size(freq, size):
    assert block_size_of(freq) == size


def test_the_block_size_is_the_inference_size_with_a_floor_at_16():
    for cls, size in INFERENCE_SIZE.items():
        assert block_size_of(cls) == max(size, W), cls


def test_a_frequency_id_of_the_stream_gives_the_size_of_its_name():
    expected = {"unknown": 16, "10s": 128, "4s": 128, "1min": 64, "5min": 64,
                "10min": 64, "15min": 64, "30min": 64, "1h": 32, "6h": 32,
                "1d": 16, "1w": 16, "1M": 16, "1Q": 16, "1Y": 16}
    assert set(expected) == set(FREQ_NAMES_V2)
    for name, size in expected.items():
        assert block_size_of(v2(name)) == size, name


def test_a_smaller_list_sends_the_other_classes_to_16():
    assert block_size_of("10S", (16, 64)) == 16
    assert block_size_of("5T", (16, 64)) == 64
    assert block_size_of("H", (16, 64)) == 16
    for freq in ("10S", "5T", "H", "D", "Q-DEC", None):
        assert block_size_of(freq, (16,)) == 16


def test_each_row_gets_the_size_of_its_frequency_with_no_draw():
    ids = torch.tensor([v2(f) for f in ("1h", "1d", "5min", "1M", "10s",
                                        "1Y", "unknown", "1h")])
    torch.manual_seed(3)
    before = torch.get_rng_state()
    sizes = row_block_sizes(ids, len(ids))
    assert sizes.tolist() == [32, 16, 64, 16, 128, 16, 16, 32]
    assert sizes.dtype == torch.long and sizes.device.type == "cpu"
    assert torch.equal(torch.get_rng_state(), before)
    assert row_block_sizes(None, 5).tolist() == [16] * 5
    assert row_block_sizes(ids, len(ids), (16, 64)).tolist() == [
        16, 16, 64, 16, 16, 16, 16, 16]


@pytest.mark.parametrize("sizes,why", [
    ((8, 16), "not in"), ((16, 24), "not in"), ((16, 256), "not in"),
    ((32, 64), "no size 16"), ((), "no size 16")])
def test_a_wrong_size_list_is_refused(sizes, why):
    with pytest.raises(ValueError, match=why):
        check_block_sizes(sizes)


def test_the_sizes_are_kept_in_increasing_order():
    assert check_block_sizes([128, 16, 64, 32]) == (16, 32, 64, 128)
    assert check_block_sizes(("16",)) == (16,)


# ---------------------------------------------------------------------------
# 2. The head
# ---------------------------------------------------------------------------

def test_the_bank_has_its_own_keys():
    """No key starts with `heads.` or `freq_`: the code of a head bank
    (#412) and of a frequency family does not read a block head."""
    sd = bank_of().state_dict()
    assert all(re.match(r"block_body\.(transformer|norm)\.|"
                        r"block_out\.(16|32|64|128)\.(weight|bias)$", k)
               for k in sd)
    assert fh.head_bank_sizes(sd) == () and family_layout_of(sd) is None
    assert block_head_sizes(sd) == BLOCK_SIZES
    assert block_head_sizes(bank_of((16, 64)).state_dict()) == (16, 64)
    assert block_head_sizes(standard_head(0).state_dict()) == ()
    assert block_head_sizes(FrequencyFamilyHead(
        standard_head, (16, 32), "shared").state_dict()) == ()
    assert block_head_sizes(ForecastingHeadBank({
        p: TransformerQuantileForecastingHead(H=H, num_layers=1, nhead=2,
                                              forecast_len=p)
        for p in (8, 16)}).state_dict()) == ()


def test_the_output_layer_of_p_reads_k_latents():
    bank = bank_of()
    for p in BLOCK_SIZES:
        k = p // W
        assert bank.block_out[str(p)].weight.shape == (p * Q, k * H), p
    extra = sum(p // W * H * p * Q + p * Q for p in BLOCK_SIZES[1:])
    assert n_params(bank) == n_params(standard_head(0)) + extra
    assert n_params(bank_of((16,))) == n_params(standard_head(0))


def test_the_trunk_and_head_16_are_the_standard_head_with_one_seed():
    """The layers are built in increasing order of P, so the trunk and the
    output layer of 16 have the weights of the standard head, bit for bit."""
    standard = standard_head(seed=7).state_dict()
    trunk = block_trunk_state(bank_of(seed=7).state_dict())
    assert trunk.keys() == standard.keys()
    assert all(torch.equal(trunk[k], standard[k]) for k in standard)
    head = standard_head(0)
    head.load_state_dict(trunk)


def test_k_1_is_the_standard_head_on_the_same_weights():
    """Head 16 decodes each latent alone, as the standard head does: the
    same output at each position, and the same loss."""
    backbone, x = tiny_backbone(), batch()
    seq, mask, start, x_norm = layout(backbone, x)
    standard, bank = standard_head(seed=7).eval(), bank_of(seed=7).eval()
    want = decode_forecast_head(standard, seq, mask, start)
    got = bank.member(16).decode_from(seq, start, mask)
    assert got.shape == want.shape == (3, PATCHES, Q, W)
    assert torch.allclose(got, want, atol=1e-6)
    assert torch.allclose(bank(seq, 16, mask, start), want, atol=1e-6)
    targets, t_valid = compute_valid_targets(x_norm, W=W, forecast_len=W)
    rows = torch.full((3,), 16)
    features = bank.features(seq, mask)[:, start:]
    loss = block_head_loss(bank, features, rows, x_norm)
    assert torch.allclose(loss, quantile_loss(want[:, :t_valid], targets),
                          atol=1e-6)
    keep = ~backbone.rev_norm.pad_mask
    kept = valid_target_keep(backbone.rev_norm.pad_mask, W, W)
    loss = block_head_loss(bank, features, rows, x_norm, keep)
    assert torch.allclose(loss, masked_quantile_loss(want[:, :t_valid],
                                                     targets, kept), atol=1e-6)


@pytest.mark.parametrize("size", BLOCK_SIZES)
def test_each_block_decodes_p_values_from_k_latents(size):
    backbone, x = tiny_backbone(), batch()
    seq, mask, start, x_norm = layout(backbone, x)
    bank, k = bank_of().eval(), size // W
    out = bank(seq, size, mask, start)
    assert out.shape == (3, PATCHES // k, Q, size)
    targets, n_blocks = block_targets(x_norm, size)
    assert n_blocks == (PATCHES - 1) // k
    assert targets.shape == (3, n_blocks, size)
    for b in range(n_blocks):
        first = (b * k + 1) * W
        assert torch.equal(targets[:, b], x_norm[:, first:first + size, 0])
    if k == 1:
        want, t_valid = compute_valid_targets(x_norm, W=W, forecast_len=W)
        assert n_blocks == t_valid and torch.equal(targets, want)


def test_the_blocks_of_a_window_follow_the_row_order_of_the_latents():
    """Two channels: the latents hold one row for each channel of each
    sample, sample by sample. The targets do too."""
    x = 50 + torch.randn(2, T_RAW, 2, generator=torch.Generator().manual_seed(1))
    targets, n_blocks = block_targets(x, 32)
    assert targets.shape == (4, (PATCHES - 1) // 2, 32)
    for b in range(2):
        for c in range(2):
            assert torch.equal(targets[2 * b + c, 0], x[b, W:W + 32, c])


def test_a_window_too_short_for_a_block_is_refused():
    with pytest.raises(ValueError, match="no whole block"):
        block_targets(torch.zeros(1, 128, 1), 128)
    _, n_blocks = block_targets(torch.zeros(1, 144, 1), 128)
    assert n_blocks == 1


def test_a_latent_moves_its_own_block_and_the_later_ones_only():
    """The trunk is causal. A change at latent t reaches the block of t and
    the blocks after it, and no block before it."""
    backbone, x = tiny_backbone(), batch()
    seq, mask, start, _ = layout(backbone, x)
    bank, size, t = bank_of().eval(), 64, 10
    moved = seq.clone()
    moved[:, start + t] += 5.0
    before, after = bank(seq, size, mask, start), bank(moved, size, mask, start)
    block = t // (size // W)
    assert torch.equal(before[:, :block], after[:, :block])
    assert not torch.allclose(before[:, block], after[:, block])


def test_a_row_trains_the_trunk_and_its_own_output_layer():
    backbone, x = tiny_backbone(), batch(n=4)
    seq, mask, start, x_norm = layout(backbone, x)
    bank = bank_of()
    rows = torch.tensor([16, 64, 16, 64])
    block_head_loss(bank, bank.features(seq, mask)[:, start:], rows,
                    x_norm).backward()
    for name, p in bank.named_parameters():
        trained = name.startswith("block_body.") or name.split(".")[1] in (
            "16", "64")
        assert (p.grad is not None) == trained, name


def test_the_group_losses_add_up_by_their_share_of_the_rows():
    backbone, x = tiny_backbone(), batch(n=4)
    seq, mask, start, x_norm = layout(backbone, x)
    bank = bank_of().eval()
    features = bank.features(seq, mask)[:, start:]
    rows = torch.tensor([128, 16, 16, 128])
    keep = ~backbone.rev_norm.pad_mask
    loss = block_head_loss(bank, features, rows, x_norm, keep)
    parts = []
    for size, picked in ((16, [1, 2]), (128, [0, 3])):
        targets, n_blocks = block_targets(x_norm, size)
        kept, _ = block_targets(keep.float(), size)
        preds = bank.decode_blocks(features[picked], size)[:, :n_blocks]
        parts.append(masked_quantile_loss(preds, targets[picked],
                                          kept[picked] > 0.5))
    assert torch.allclose(loss, 0.5 * parts[0] + 0.5 * parts[1], atol=1e-6)
    assert loss.isfinite()


@pytest.mark.parametrize("size", BLOCK_SIZES)
def test_a_member_is_a_quantile_head_of_p_values_with_no_patch_size(size):
    bank = bank_of().eval()
    member = bank.member(size)
    assert isinstance(member, BlockHeadMember)
    assert isinstance(member, TransformerQuantileForecastingHead)
    assert member.forecast_len == member.block_size == size
    assert member.latents_per_block == size // W
    assert getattr(member, "patch_size", None) is None
    assert member.transformer is bank.block_body.transformer
    assert member.forecast_head is bank.block_out[str(size)]
    assert not member.training
    latents = torch.randn(3, 24, H)
    assert member(latents).shape == (3, 24 // (size // W), Q, size)
    assert bank.for_frequency("15T").block_size == 64
    assert bank.for_frequency("A-DEC").block_size == 16


def test_the_trunk_is_a_transformer_quantile_head_of_16():
    with pytest.raises(ValueError, match="transformer"):
        BlockSizeHead(LinearQuantileForecastingHead(H=H, forecast_len=W))
    with pytest.raises(ValueError, match="16"):
        BlockSizeHead(TransformerQuantileForecastingHead(
            H=H, num_layers=1, nhead=2, forecast_len=32))
    with pytest.raises(ValueError, match="no size 16"):
        BlockSizeHead(standard_head(0), (32, 64))


def test_the_mode_of_the_bank_is_the_mode_of_its_members():
    bank = bank_of(dropout=0.1)
    seq = torch.randn(2, 8, H)
    assert bank.training and bank.member(32).training
    assert not torch.equal(bank(seq, 32), bank(seq, 32))
    bank.eval()
    assert not bank.member(32).training
    assert torch.equal(bank(seq, 32), bank(seq, 32))


# ---------------------------------------------------------------------------
# 3. The default path
# ---------------------------------------------------------------------------

def test_the_standard_head_is_its_output_layer_on_its_features():
    """`features` is the trunk of the standard head: the same operations,
    so the output of the head is unchanged, bit for bit."""
    head = standard_head(seed=3).eval()
    seq = torch.randn(2, 2 * PATCHES, H)
    mask = fh.build_e_then_f_mask(PATCHES, PATCHES)
    for kwargs in ({}, {"src_mask": mask}):
        flat = head.forecast_head(head.features(seq, **kwargs))
        assert torch.equal(head(seq, **kwargs), flat.view(2, -1, Q, W))
    assert standard_head(seed=3).state_dict().keys() == {
        k for k in head.state_dict()}


def test_a_standard_head_rolls_out_one_latent_per_patch():
    head, bank = standard_head(0), bank_of()
    for horizon in (1, 16, 17, 48, 200, 900):
        assert rollout_steps(head, horizon, W) == math.ceil(horizon / W)
        for size in BLOCK_SIZES:
            k = size // W
            assert rollout_steps(bank.member(size), horizon, W) == (
                math.ceil(horizon / size) * k)


def load_head_trainer():
    spec = importlib.util.spec_from_file_location("head_trainer_blocks",
                                                  HEAD_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_trainer_default_is_the_standard_head(tmp_path):
    trainer = load_head_trainer()
    bb = tmp_path / "bb.pth"
    torch.save(tiny_backbone().state_dict(), bb)
    args = trainer.parse_args(["--backbone-path", str(bb), *HEAD_PROTOCOL,
                               "--save-dir", str(tmp_path / "head")])
    assert args.head_block_sizes is None
    assert trainer.block_sizes_of_flags(args, False, True) is None
    job = trainer.HeadJob(args)
    assert job.block_sizes is None
    assert type(job.head) is TransformerQuantileForecastingHead


# ---------------------------------------------------------------------------
# 4. The head trainer
# ---------------------------------------------------------------------------

# The flags of a B4 head at the tiny shape, on the CPU.
TINY_SHAPE = ("--d-model", "16", "--n-heads", "2", "--num-layers", "1")
STEPS, BATCH = 4, 8
HEAD_PROTOCOL = (
    "--device", "cpu", "--quantile-head", "--grad-clip", "1.0",
    "--forecast-len", "16", "--batch-size", str(BATCH), "--lr", "1e-3",
    "--total-steps", str(STEPS), "--save-every", "1000000",
    "--log-every", "1", "--seed", "20260722",
    "--hf-repo", "jeremycochoy/gift-pretrain-full-4096",
    "--hf-path", "small_v1", "--head-arch", "transformer",
    "--head-num-layers", "2", "--head-nhead", "2", "--head-ffn-mult", "4.0",
    "--head-causal", "true", "--head-train-input", "e_then_f",
    "--head-dropout", "0.1", "--t-raw", "4096", "--n-channels", "1",
    *TINY_SHAPE, "--encoder-type", "gru", "--rev-norm-kind", "ewma",
    "--rev-norm-span", "128", "--freq-emb-dim", "3",
    "--seasonality-emb-dim", "3")

# The frequency of each source of the test corpus, and its block size.
SOURCES = {"sec": "10S", "min5": "5T", "hour": "H", "day": "D",
           "year": "A-DEC"}
ARMS = {"control": (), "blocks16": ("--head-block-sizes", "16"),
        "blocks": ("--head-block-sizes", "16,32,64,128")}


def trainer_env():
    return dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="",
                OMP_NUM_THREADS="2")


def job_argv(folder, backbone, corpus, arm, *extra):
    return ["--backbone-path", backbone, *HEAD_PROTOCOL, *corpus,
            "--save-dir", str(folder / arm), "--run-name", arm,
            *ARMS[arm], *extra]


def train_solo(argv):
    return subprocess.run([sys.executable, str(HEAD_PY), *argv],
                          capture_output=True, text=True, env=trainer_env(),
                          timeout=900)


def run_files(folder, arm):
    """The loss rows and the final head of one run."""
    rows = list(csv.reader(open(folder / arm / f"{arm}_losses.csv")))
    head = torch.load(folder / arm / f"{arm}_final.pth", map_location="cpu",
                      weights_only=True)
    return rows, head


def losses(rows):
    return np.array([float(row[1]) for row in rows[1:]])


def size_rows(stdout):
    """The count of rows of each block size that a run prints at its end."""
    line = re.search(r"rows of each block size.*", stdout)
    assert line, stdout[-2000:]
    return {int(k): int(v) for k, v in re.findall(r"(\d+): (\d+)",
                                                  line.group(0))}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A corpus in the layout of GiftEvalPretrain with five frequencies:
    the stream of a zero-padding backbone, with a label on each window."""
    pytest.importorskip("pyarrow")
    from tests.test_419_gift_pretrain import (index_sources, series,
                                              write_index, write_source)
    root = tmp_path_factory.mktemp("blocks_corpus")
    rng = np.random.default_rng(0)
    for name, freq in SOURCES.items():
        length = 40 if name == "year" else 1500
        rows = [{"target": series(rng, length, 10.0)} for _ in range(4)]
        write_source(root / "corpus", name, rows, freq)
    index = index_sources(root / "corpus", dict.fromkeys(SOURCES, 1.0))
    return ("--gift-pretrain-root", str(root / "corpus"),
            "--gift-pretrain-index", str(write_index(root, index)))


@pytest.fixture(scope="module")
def backbone_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("blocks_bb") / "bb.pth"
    torch.save(tiny_backbone().state_dict(), path)
    return str(path)


@pytest.fixture(scope="module")
def solo_runs(tmp_path_factory, backbone_path, corpus):
    """Each arm trained alone: the control, a bank of 16 and the bank."""
    folder = tmp_path_factory.mktemp("blocks_solo")
    return folder, {arm: train_solo(job_argv(folder, backbone_path, corpus,
                                             arm)) for arm in ARMS}


def test_a_block_head_trains_on_the_cpu_and_its_loss_is_finite(solo_runs):
    folder, runs = solo_runs
    r = runs["blocks"]
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "Block head: sizes (16, 32, 64, 128)" in r.stdout
    rows, head = run_files(folder, "blocks")
    assert len(rows) == STEPS + 1
    assert np.isfinite(losses(rows)).all() and (losses(rows) > 0).all()
    assert block_head_sizes(head) == BLOCK_SIZES
    assert fh.head_bank_sizes(head) == () and family_layout_of(head) is None
    counts = size_rows(r.stdout)
    assert sum(counts.values()) == STEPS * BATCH
    assert sum(1 for p in BLOCK_SIZES if counts[p] > 0) >= 3
    assert (folder / "blocks" / "blocks_final_optimizer.pth").exists()


def test_a_bank_of_16_writes_the_loss_rows_of_the_standard_head(solo_runs):
    """Every row trains head 16, which decodes each latent alone: the loss
    of the standard head at each step, and its weights at the end, up to
    floating-point noise (the output layer runs on the forecaster positions
    only, so its kernel reads another shape)."""
    folder, runs = solo_runs
    for arm in ("control", "blocks16"):
        assert runs[arm].returncode == 0, runs[arm].stderr[-3000:]
    want_rows, want_head = run_files(folder, "control")
    rows, head = run_files(folder, "blocks16")
    assert len(rows) == STEPS + 1 and [r[0] for r in rows] == [
        r[0] for r in want_rows]
    np.testing.assert_allclose(losses(rows), losses(want_rows), rtol=1e-4)
    assert block_head_sizes(head) == (16,)
    trunk = block_trunk_state(head)
    assert trunk.keys() == want_head.keys()
    for key in want_head:
        assert torch.allclose(trunk[key], want_head[key], rtol=1e-4,
                              atol=1e-5), key
    assert size_rows(runs["blocks16"].stdout) == {16: STEPS * BATCH}
    assert "rows of each block size" not in runs["control"].stdout
    assert block_head_sizes(want_head) == ()


def test_the_bank_trains_other_weights_than_the_standard_head(solo_runs):
    folder, runs = solo_runs
    assert all(r.returncode == 0 for r in runs.values())
    control, blocks = losses(run_files(folder, "control")[0]), losses(
        run_files(folder, "blocks")[0])
    assert not np.allclose(control, blocks)


def test_a_shared_run_gives_the_bank_the_steps_of_its_solo_run(
        tmp_path, solo_runs, backbone_path, corpus):
    """The three arms in one process on one data stream: each arm writes
    the losses and the head of its solo run, bit for bit."""
    folder, runs = solo_runs
    assert all(r.returncode == 0 for r in runs.values())
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text("".join(
        json.dumps(job_argv(tmp_path, backbone_path, corpus, arm)) + "\n"
        for arm in ARMS))
    r = subprocess.run([sys.executable, str(SHARED_PY), "--jobs", str(jobs)],
                       capture_output=True, text=True, env=trainer_env(),
                       timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for arm in ARMS:
        rows, head = run_files(tmp_path, arm)
        want_rows, want_head = run_files(folder, arm)
        assert rows == want_rows, arm
        assert head.keys() == want_head.keys()
        assert all(torch.equal(head[k], want_head[k]) for k in head), arm
    line = re.search(r"\[blocks\] .*rows of each block size.*", r.stdout)
    assert line and size_rows(line.group(0)) == size_rows(
        runs["blocks"].stdout)


def head_args(module, *flags):
    return module.parse_args(["--backbone-path", "unused.pth",
                              *HEAD_PROTOCOL, *flags])


@pytest.mark.parametrize("flags,why", [
    (("--head-block-sizes", "16,32", "--freq-family", "heads"),
     "--freq-family"),
    (("--head-block-sizes", "16,32", "--reconstruction", "encoder"),
     "--reconstruction"),
    (("--head-block-sizes", "16,32", "--mixed-rollout", "4"),
     "--mixed-rollout"),
    (("--head-block-sizes", "16,32", "--head-arch", "gru"), "transformer"),
    (("--head-block-sizes", "16,32", "--head-arch", "transformer-gaussian"),
     "transformer"),
    (("--head-block-sizes", "16,32", "--forecast-len", "128"),
     "--forecast-len 16"),
    (("--head-block-sizes", "8,16"), "not in"),
    (("--head-block-sizes", "16,24"), "not in"),
    (("--head-block-sizes", "32,64"), "no size 16")])
def test_the_trainer_refuses_a_bank_that_it_cannot_train(flags, why):
    trainer = load_head_trainer()
    with pytest.raises(SystemExit, match=why):
        trainer.block_sizes_of_flags(head_args(trainer, *flags), False, True)


def test_a_bank_needs_a_quantile_head_a_one_size_backbone_and_labels():
    trainer = load_head_trainer()
    args = head_args(trainer, "--head-block-sizes", "16,32")
    assert trainer.block_sizes_of_flags(args, False, True) == (16, 32)
    with pytest.raises(SystemExit, match="head bank"):
        trainer.block_sizes_of_flags(args, True, True)
    with pytest.raises(SystemExit, match="frequency label"):
        trainer.block_sizes_of_flags(args, False, False)
    flags = [f for f in HEAD_PROTOCOL if f != "--quantile-head"]
    args = trainer.parse_args(["--backbone-path", "unused.pth", *flags,
                               "--head-block-sizes", "16,32"])
    with pytest.raises(SystemExit, match="--quantile-head"):
        trainer.block_sizes_of_flags(args, False, True)


def test_the_trainer_refuses_a_bank_on_a_head_bank_backbone(tmp_path,
                                                            corpus):
    """A backbone with the mean/std scaling trains a head bank (#412)."""
    bb = tmp_path / "bb.pth"
    torch.save(tiny_backbone(rev_norm_kind="meanstd",
                             rev_norm_span=None).state_dict(), bb)
    r = train_solo(["--backbone-path", str(bb), *HEAD_PROTOCOL, *corpus,
                    "--save-dir", str(tmp_path / "head"),
                    "--head-block-sizes", "16,32,64,128"])
    assert r.returncode != 0
    assert "head bank" in r.stdout + r.stderr
    assert not (tmp_path / "head").exists() or not list(
        (tmp_path / "head").glob("*_final.pth"))


# ---------------------------------------------------------------------------
# 5. The scorer
# ---------------------------------------------------------------------------

def walk(n, seed, level=60.0, scale=1.0):
    rng = np.random.default_rng(seed)
    return (level + scale * rng.standard_normal(n).cumsum()).astype(np.float32)


def contexts():
    """``(5, T_RAW, 1)`` raw windows: a plain walk, another level and scale,
    left zero padding that ends inside a patch, a window of one real patch,
    and a window of padding only."""
    rows = [walk(T_RAW, 0), walk(T_RAW, 1, level=1e3, scale=20.0),
            walk(T_RAW, 2), walk(T_RAW, 3), np.zeros(T_RAW, np.float32)]
    rows[2][:100] = 0.0
    rows[3][:T_RAW - W] = 0.0
    return torch.from_numpy(np.stack(rows))[:, :, None]


def rollout_spy(monkeypatch):
    """Records the latent count of each rollout."""
    seen, real = [], fh.rollout_latent

    def spy(backbone, latents, n, feedback=None):
        seen.append(n)
        return real(backbone, latents, n, feedback)
    monkeypatch.setattr(fh, "rollout_latent", spy)
    return seen


@pytest.mark.parametrize("size,horizon,latents", [
    (128, 200, 16), (128, 128, 8), (128, 16, 8), (64, 100, 8), (32, 40, 4),
    (16, 40, 3)])
def test_b4_rolls_out_whole_blocks_and_cuts_to_the_horizon(
        monkeypatch, size, horizon, latents):
    """ceil(h / P) blocks of k latents, decoded whole, then cut to h."""
    seen = rollout_spy(monkeypatch)
    backbone, member = tiny_backbone(), bank_of().eval().member(size)
    context = contexts()[0]
    out = forecast_B4(backbone, member, context, horizon, CPU)
    assert out.shape == (Q, horizon, 1) and np.isfinite(out).all()
    assert seen == [latents]
    # The whole blocks: a longer horizon of the same blocks starts with
    # the same values.
    whole = forecast_B4(backbone, member, context, latents * W, CPU)
    assert np.array_equal(out, whole[:, :horizon])


def test_the_blocks_of_the_rolled_latents_are_decoded_from_the_first_one():
    backbone, bank = tiny_backbone(), bank_of().eval()
    member = bank.member(64)
    with torch.no_grad():
        e_ctx, _ = extract_encoder_latents(backbone, contexts()[:1])
        rolled = rollout_latent(backbone, e_ctx, 12)
        out = fh._b_variant_decode(member, e_ctx, rolled, e_ctx.size(1))
        seq = torch.cat([e_ctx, rolled], dim=1)
        want = bank(seq, 64, start=e_ctx.size(1))
        alone = member.decode_from(seq, e_ctx.size(1))
    assert out.shape == (1, 3, Q, 64) and torch.equal(out, want)
    assert torch.equal(out, alone)


def test_a_bank_of_16_forecasts_as_the_standard_head():
    backbone = tiny_backbone()
    standard, bank = standard_head(seed=7).eval(), bank_of(seed=7).eval()
    for context in contexts()[:3]:
        want = forecast_B4(backbone, standard, context, 45, CPU)
        got = forecast_B4(backbone, bank.member(16), context, 45, CPU)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_two_sizes_give_two_forecasts():
    backbone, bank = tiny_backbone(), bank_of().eval()
    a = forecast_B4(backbone, bank.for_frequency("H"), contexts()[0], 45, CPU)
    b = forecast_B4(backbone, bank.for_frequency("15T"), contexts()[0], 45, CPU)
    assert not np.allclose(a, b)


@pytest.mark.parametrize("size", [16, 32, 128])
def test_a_batch_of_windows_is_the_windows_one_by_one(size):
    backbone, member = tiny_backbone(), bank_of().eval().member(size)
    horizon = 40
    batch_out = forecast_B4_windows(backbone, member, contexts(), horizon, CPU)
    assert batch_out.shape == (5, Q, horizon, 1)
    for i, context in enumerate(contexts()):
        one = forecast_B4(backbone, member, context, horizon, CPU)
        assert one.shape == (Q, horizon, 1) and np.isfinite(one).all()
        np.testing.assert_allclose(batch_out[i], one, **TOL)


def load_eval_module(name="eval_blocks"):
    pytest.importorskip("gluonts")
    pytest.importorskip("gift_eval")
    spec = importlib.util.spec_from_file_location(name, EVAL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The flags that eval_local.sh gives the GIFT-Eval script, at the tiny shape.
EVAL_PROTOCOL = (
    "--strategy", "B4", "--forecast-len", "16", "--device", "cpu",
    "--t-raw", "4096", "--n-channels", "1", *TINY_SHAPE, "--encoder-type",
    "gru", "--rev-norm-kind", "ewma", "--rev-norm-span", "128",
    "--head-nhead", "2", "--head-causal", "true")


def eval_args(module, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["eval", *EVAL_PROTOCOL, *extra])
    return module.parse_args()


def items(n=7):
    """Gluonts items, the edge cases of a config: a series longer than the
    context, one shorter (left padding), missing values inside and at the
    end, a missing start, an all-NaN series, another level and scale, and a
    seventh so that the last batch of four is a partial one."""
    import pandas as pd
    start = pd.Period("2020-01-01 00:00", freq="h")
    series = [walk(1024 + 100, 0), walk(100, 1), walk(1024 + 100, 2),
              walk(1024 + 50, 3), np.full(1024, np.nan, np.float32),
              walk(1024 + 100, 4, level=1e3, scale=20.0), walk(1024, 5)]
    series[2][50:80] = series[2][-3:] = np.nan
    series[3][:1024 - 40] = np.nan
    return [{"target": s, "start": start, "item_id": f"s{i}"}
            for i, s in enumerate(series[:n])]


def predictors(module, backbone, head, batch_size, horizon):
    kwargs = dict(prediction_length=horizon, device=CPU, t_raw=1024,
                  strategy="B4", context_pad="zeros")
    return (module.ContrastiveForecasterPredictor(backbone, head, **kwargs),
            module.BatchedForecastPredictor(backbone, head,
                                            batch_size=batch_size, **kwargs))


@pytest.mark.parametrize("size,horizon", [(64, 48), (128, 200), (16, 48)])
def test_the_batched_predictor_gives_each_window_its_own_forecast(
        size, horizon):
    module = load_eval_module(f"eval_blocks_batched_{size}")
    one_by_one, batched = predictors(module, tiny_backbone(),
                                     bank_of().eval().member(size), 4, horizon)
    assert batched.batch_size == fh.windows_per_pass(
        4, 1024 // W, rollout_steps(batched.head, horizon, W))
    got = list(batched.predict(items()))
    want = [one_by_one.predict_item(it) for it in items()]
    assert len(got) == len(want) == 7
    for g, w in zip(got, want):
        assert (g.start_date, g.item_id, g.forecast_keys) == (
            w.start_date, w.item_id, w.forecast_keys)
        assert w.forecast_array.shape == (1 + Q, horizon)
        assert np.isfinite(w.forecast_array).all()
        np.testing.assert_allclose(g.forecast_array, w.forecast_array, **TOL)


def hourly_item(n=700):
    import pandas as pd
    return {"target": walk(n, 0), "start": pd.Period("2020-01-01 00:00",
                                                     freq="h")}


def forecast_of(module, args, backbone, head, item, horizon=48):
    predictor = module.ContrastiveForecasterPredictor(
        backbone=backbone, head=head, prediction_length=horizon, device=CPU,
        strategy="B4", context_pad=args.context_pad)
    return predictor.predict_item(item).forecast_array


def test_the_eval_loads_a_block_head_off_its_keys(tmp_path, backbone_path,
                                                  monkeypatch, capsys):
    torch.save(bank_of().state_dict(), tmp_path / "blocks.pth")
    module = load_eval_module()
    args = eval_args(module, monkeypatch, "--backbone-path", backbone_path,
                     "--head-path", str(tmp_path / "blocks.pth"))
    backbone, blocks = module.load_models(args, CPU)
    assert isinstance(blocks, BlockSizeHead) and blocks.sizes == BLOCK_SIZES
    assert not blocks.training and args.context_pad == "zeros"
    assert "block head: one trunk, one output layer per block size" in (
        capsys.readouterr().out)
    head = module.block_config_head(blocks, "ett1/15T/short", "15T")
    assert head.block_size == 64 and head.latents_per_block == 4
    assert head.forecast_head is blocks.block_out["64"]
    assert ("ett1/15T/short: frequency 15T, block size 64 (4 latents per "
            "block)" in capsys.readouterr().out)
    forecast = forecast_of(module, args, backbone, head, hourly_item())
    assert forecast.shape == (1 + Q, 48) and np.isfinite(forecast).all()
    other = module.block_config_head(blocks, "ett1/H/short", "H")
    assert other.block_size == 32
    assert not np.allclose(
        forecast, forecast_of(module, args, backbone, other, hourly_item()))
    for freq, size in (("10S", 128), ("D", 16), ("Q-DEC", 16)):
        assert module.block_config_head(blocks, f"x/{freq}/short",
                                        freq).block_size == size


def test_the_eval_forecasts_with_a_trained_block_head(solo_runs,
                                                      backbone_path,
                                                      monkeypatch):
    folder, runs = solo_runs
    assert runs["blocks"].returncode == 0, runs["blocks"].stderr[-3000:]
    module = load_eval_module("eval_blocks_trained")
    args = eval_args(module, monkeypatch, "--backbone-path", backbone_path,
                     "--head-path", str(folder / "blocks" / "blocks_final.pth"))
    backbone, blocks = module.load_models(args, CPU)
    assert blocks.sizes == BLOCK_SIZES
    for freq in ("10S", "15T", "H", "D"):
        head = module.block_config_head(blocks, f"x/{freq}/short", freq)
        forecast = forecast_of(module, args, backbone, head, hourly_item())
        assert forecast.shape == (1 + Q, 48) and np.isfinite(forecast).all()


def test_a_bank_of_16_scores_as_the_standard_head(tmp_path, backbone_path,
                                                  monkeypatch):
    """The eval of a bank of 16 with the weights of a standard head gives
    the forecast of that head, for each frequency."""
    standard = standard_head(seed=5)
    torch.save(standard.state_dict(), tmp_path / "standard.pth")
    torch.save(bank_of((16,), seed=5).state_dict(), tmp_path / "blocks.pth")
    forecasts = {}
    for name in ("standard", "blocks"):
        module = load_eval_module(f"eval_blocks_{name}")
        args = eval_args(module, monkeypatch, "--backbone-path",
                         backbone_path, "--head-path",
                         str(tmp_path / f"{name}.pth"))
        backbone, head = module.load_models(args, CPU)
        if name == "blocks":
            assert head.sizes == (16,)
            heads = [module.block_config_head(head, f"x/{f}/short", f)
                     for f in ("10S", "H", "A-DEC")]
        else:
            assert type(head) is TransformerQuantileForecastingHead
            heads = [head]
        forecasts[name] = [forecast_of(module, args, backbone, h,
                                       hourly_item()) for h in heads]
    for forecast in forecasts["blocks"]:
        np.testing.assert_allclose(forecast, forecasts["standard"][0],
                                   rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("strategy", ["A2", "R", "B1"])
def test_the_eval_scores_a_block_head_under_b4_only(tmp_path, backbone_path,
                                                    monkeypatch, strategy):
    torch.save(bank_of().state_dict(), tmp_path / "blocks.pth")
    module = load_eval_module()
    args = eval_args(module, monkeypatch, "--backbone-path", backbone_path,
                     "--head-path", str(tmp_path / "blocks.pth"),
                     "--strategy", strategy)
    with pytest.raises(SystemExit, match="block head scores under "
                                         "--strategy B4"):
        module.load_models(args, CPU)


def test_the_eval_refuses_a_block_head_on_a_backbone_with_patch_sizes(
        tmp_path, monkeypatch):
    """A backbone with patch sizes is scored with its head bank (#412)."""
    bb = tmp_path / "bb.pth"
    torch.save(tiny_backbone(
        multi_patch_sizes=(8, 16, 32, 64, 128)).state_dict(), bb)
    torch.save(bank_of().state_dict(), tmp_path / "blocks.pth")
    module = load_eval_module()
    args = eval_args(module, monkeypatch, "--backbone-path", str(bb),
                     "--head-path", str(tmp_path / "blocks.pth"))
    with pytest.raises(SystemExit, match="head bank"):
        module.load_models(args, CPU)


def test_the_eval_of_the_other_heads_is_unchanged(tmp_path, backbone_path,
                                                  monkeypatch):
    """A head file with no block key loads as before."""
    torch.save(standard_head(seed=5).state_dict(), tmp_path / "standard.pth")
    torch.manual_seed(5)
    family = FrequencyFamilyHead(standard_head, (16, 64), "shared")
    torch.save(family.state_dict(), tmp_path / "family.pth")
    module = load_eval_module()
    args = eval_args(module, monkeypatch, "--backbone-path", backbone_path,
                     "--head-path", str(tmp_path / "standard.pth"))
    _, head = module.load_models(args, CPU)
    assert type(head) is TransformerQuantileForecastingHead
    args = eval_args(module, monkeypatch, "--backbone-path", backbone_path,
                     "--head-path", str(tmp_path / "family.pth"))
    _, head = module.load_models(args, CPU)
    assert isinstance(head, FrequencyFamilyHead) and head.members == (16, 64)


# ---------------------------------------------------------------------------
# The scripts of the wave: the arm `blocks` of run_wave.sh and the knob
# CF_HEAD_BLOCK_SIZES of head_eval_bb.sh
# ---------------------------------------------------------------------------

from tests.test_freq_family import (  # noqa: E402
    calls, head_eval, recorded, run_wave, stub_checkout, tag_of, wave_box,
    without_names)

BLOCK_FLAGS = ["--head-block-sizes", "16,32,64,128"]


def test_the_block_knob_adds_only_the_block_flag(stub_checkout):
    """Each flag of the standard B4 head stays. The block flag comes after
    them, and the score is B4."""
    tmp_path = stub_checkout[0]
    r = head_eval(stub_checkout, "run_bb200k_h30k_control")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    control = recorded(tmp_path / "root" / "eval" / "run_bb200k_h30k_control"
                       / "head_argv.json")
    assert "--head-block-sizes" not in control
    tag = "run_bb200k_h30k_ff_blocks"
    r = head_eval(stub_checkout, tag, CF_HEAD_BLOCK_SIZES="16,32,64,128")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = tmp_path / "root" / "eval" / tag
    assert without_names(recorded(out / "head_argv.json")) == (
        without_names(control) + BLOCK_FLAGS)
    shard = recorded(out / "gift" / "shard_0" / "argv.json")
    assert shard[shard.index("--strategy") + 1] == "B4"


@pytest.mark.parametrize("tag,knobs", [
    # A block head with the tag of a standard head would take its files.
    ("run_bb200k_h30k_control", dict(CF_HEAD_BLOCK_SIZES="16,32,64,128")),
    ("run_bb200k_h30k_ff_blocks",
     dict(CF_HEAD_BLOCK_SIZES="16,32,64,128", CF_FREQ_FAMILY="heads")),
    ("run_bb200k_h30k_ff_blocks",
     dict(CF_HEAD_BLOCK_SIZES="16,32,64,128", CF_RECONSTRUCTION="encoder"))])
def test_a_wrong_block_call_is_refused(stub_checkout, tag, knobs):
    tmp_path = stub_checkout[0]
    r = head_eval(stub_checkout, tag, **knobs)
    assert r.returncode != 0, r.stdout[-2000:]
    assert "ABORT" in r.stdout + r.stderr
    assert not (tmp_path / "root" / "eval" / tag / "head_argv.json").exists()


def test_a_wave_trains_the_block_arm_beside_the_control(wave_box):
    base = wave_box[1]
    r = run_wave(wave_box, "control", "blocks", FF_SCORE="0")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    (wave,) = [c for c in calls(base) if c[0] == "wave"]
    assert wave[3:] == [tag_of("control"), tag_of("blocks")]


def test_the_block_arm_sets_the_block_knob():
    """arm_env of run_wave.sh gives the arm `blocks` the four block sizes,
    and the control no knob."""
    script = (Path(__file__).resolve().parents[1] / "reports"
              / "2026-10-10_frequency_family_head" / "scripts" / "run_wave.sh")
    body = script.read_text()
    func = body[body.index("arm_env(){"):body.index("\n}\n", body.index("arm_env(){")) + 3]
    for arm, want in (("blocks", "CF_HEAD_BLOCK_SIZES=16,32,64,128"), ("control", "")):
        r = subprocess.run(["bash", "-c", func + f"arm_env {arm}"], capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == want, (arm, r.stdout, r.stderr)
