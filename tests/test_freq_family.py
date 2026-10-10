"""Tests for the frequency family: a family of forecast decoders on a
backbone with one patch size, selected by the frequency of the series.

Each member decodes the quantiles of the 16 values of the next patch, as the
standard head does. The backbone always reads patches of 16. The key of a
member (16, 32, 64, 128) only selects the decoder.

Groups, all on the CPU:

1. The member of a frequency: the fixed member that scores it, the members
   that a row can train under the draw rule, and the member of each of the
   97 GIFT-Eval configs.
2. The head: the state dict of each body, the decoder of each row, the
   gradients, and a family of one member against the standard head.
3. The B4 forecast of a member.
4. The head trainer: a family that sends each row to one member writes the
   loss rows of the standard head. The four arms train and count their rows.
   The shared trainer gives each arm the steps of its solo run. The
   refusals.
5. The eval script: it loads a family, names the member of each config and
   refuses another strategy.
6. `head_eval_bb.sh`: the family flags reach the head trainer.
7. The scripts of the report: one wave, the follower of `abc_gift`, and the
   score tables.
"""

from __future__ import annotations

import collections
import csv
import fcntl
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.forecasting_head as fh  # noqa: E402
from src.forecasting_head import (QUANTILE_LEVELS,  # noqa: E402
                                  ForecastingHeadBank,
                                  LinearQuantileForecastingHead,
                                  TransformerQuantileForecastingHead,
                                  forecast_B4, head_bank_sizes)
from src.freq_embedding import FREQ_NAMES_V2  # noqa: E402
from src.freq_family import (FAMILY_BODIES, FAMILY_MEMBERS,  # noqa: E402
                             FAMILY_RULES, FrequencyFamilyHead,
                             check_family_members, family_layout_of,
                             family_member, family_member_choices,
                             family_member_state, family_row_members)
from src.models import ConfigurableModel  # noqa: E402

GIFT_SCRIPTS = REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
HEAD_PY = GIFT_SCRIPTS / "train_forecasting_head.py"
SHARED_PY = GIFT_SCRIPTS / "train_forecasting_heads_shared.py"
EVAL_PY = GIFT_SCRIPTS / "eval_gift_eval_official.py"
B4_SCRIPTS = REPO_ROOT / "reports" / "2026-08-08_rollout_depth" / "scripts"
SCRIPTS = (REPO_ROOT / "reports" / "2026-10-10_frequency_family_head"
           / "scripts")
Q = len(QUANTILE_LEVELS)
T = 1024
CPU = torch.device("cpu")
BODIES = ("shared", "heads")
ARMS = ("control", "shared_strict", "shared_draw", "heads_strict",
        "heads_draw")


def v2(name):
    """The id of a frequency name in the vocabulary v2."""
    return FREQ_NAMES_V2.index(name)


# ---------------------------------------------------------------------------
# 1. The member of a frequency
# ---------------------------------------------------------------------------

def test_the_family_has_four_members_two_bodies_and_two_rules():
    assert FAMILY_MEMBERS == (16, 32, 64, 128)
    assert set(FAMILY_BODIES) == {"shared", "heads"}
    assert set(FAMILY_RULES) == {"strict", "draw"}


@pytest.mark.parametrize("freq,member", [
    ("10S", 128), ("4S", 128), ("T", 64), ("5T", 64), ("15T", 64),
    ("10min", 64), ("H", 32), ("6H", 32), ("D", 16), ("B", 16),
    ("W-SUN", 16), ("W-FRI", 16), ("M", 16), ("MS", 16),
    # Q and Y have no member of their own: they are 0.02% of the stream.
    ("Q-DEC", 16), ("A-DEC", 16), ("YE-DEC", 16),
    # No frequency label.
    (None, 16), ("ms", 16)])
def test_each_frequency_has_one_fixed_member(freq, member):
    assert family_member(freq) == member


def test_a_frequency_id_of_the_stream_gives_the_member_of_its_name():
    expected = {"unknown": 16, "10s": 128, "4s": 128, "1min": 64, "5min": 64,
                "10min": 64, "15min": 64, "30min": 64, "1h": 32, "6h": 32,
                "1d": 16, "1w": 16, "1M": 16, "1Q": 16, "1Y": 16}
    assert set(expected) == set(FREQ_NAMES_V2)
    for name, member in expected.items():
        assert family_member(v2(name)) == member, name


@pytest.mark.parametrize("freq,choices", [
    ("10S", (64, 128)), ("5T", (32, 64, 128)), ("H", (32, 64)),
    ("D", (16, 32)), ("B", (16, 32)), ("W-SUN", (16, 32)), ("M", (16, 32)),
    # The training range of Q and Y is 8 only, and 8 is no member.
    ("Q-DEC", (16,)), ("A-DEC", (16,)),
    # A row with no label draws from each member, as Moirai does.
    (None, (16, 32, 64, 128))])
def test_the_draw_rule_takes_the_members_of_the_training_range(freq, choices):
    assert family_member_choices(freq) == choices


def test_a_smaller_family_sends_the_other_classes_to_the_base_member():
    for freq in ("10S", "5T", "H", "D", "M", "Q-DEC", "A-DEC", None):
        assert family_member(freq, (16,)) == 16
        assert family_member_choices(freq, (16,)) == (16,)
    assert family_member("5T", (16, 64)) == 64
    assert family_member("H", (16, 64)) == 16
    assert family_member_choices("H", (16, 64)) == (64,)


@pytest.mark.parametrize("members,why", [
    ((16, 24), "not in"), ((32, 64), "base member 16"), ((), "base member")])
def test_a_wrong_member_set_is_refused(members, why):
    with pytest.raises(ValueError, match=why):
        check_family_members(members)


def test_the_members_are_kept_in_increasing_order():
    assert check_family_members([64, 16, 128, 32]) == (16, 32, 64, 128)
    assert check_family_members((8, 16)) == (8, 16)


ROW_FREQS = ("1h", "1d", "5min", "1M", "10s", "1Y", "unknown", "1h")


def test_the_strict_rule_gives_each_row_its_fixed_member_with_no_draw():
    ids = torch.tensor([v2(f) for f in ROW_FREQS])
    torch.manual_seed(3)
    before = torch.get_rng_state()
    members = family_row_members(ids, len(ids), "strict")
    assert members.tolist() == [32, 16, 64, 16, 128, 16, 16, 32]
    assert members.dtype == torch.long and members.device.type == "cpu"
    assert torch.equal(torch.get_rng_state(), before)


def test_a_batch_with_no_label_trains_the_base_member_under_strict():
    assert family_row_members(None, 5, "strict").tolist() == [16] * 5


def test_the_draw_rule_draws_each_row_from_its_range():
    names = ("10s", "5min", "1h", "1d", "1M", "1Q", "1Y", "unknown")
    ids = torch.tensor([v2(f) for f in names]).repeat_interleave(600)
    torch.manual_seed(0)
    members = family_row_members(ids, len(ids), "draw")
    for i, name in enumerate(names):
        choices = family_member_choices(v2(name))
        drawn = members[i * 600:(i + 1) * 600]
        counts = collections.Counter(drawn.tolist())
        assert set(counts) == set(choices), name
        # A uniform draw: each choice has its share of the 600 rows.
        for choice in choices:
            assert abs(counts[choice] / 600 - 1 / len(choices)) < 0.08, name
    torch.manual_seed(0)
    assert torch.equal(members, family_row_members(ids, len(ids), "draw"))


def test_a_range_of_one_member_needs_no_draw():
    """So a family of one member uses the random state of the standard
    head, under each rule."""
    ids = torch.tensor([v2(f) for f in ("1Q", "1Y", "1Y")])
    torch.manual_seed(3)
    before = torch.get_rng_state()
    assert family_row_members(ids, 3, "draw").tolist() == [16, 16, 16]
    every = torch.tensor([v2(f) for f in ROW_FREQS])
    assert family_row_members(every, 8, "draw", (16,)).tolist() == [16] * 8
    assert torch.equal(torch.get_rng_state(), before)


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="rule"):
        family_row_members(None, 2, "uniform")


def load_eval_module(name="eval_freq_family"):
    pytest.importorskip("gluonts")
    pytest.importorskip("gift_eval")
    spec = importlib.util.spec_from_file_location(name, EVAL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The member of each of the 97 GIFT-Eval configs: T 30, H 31, S 6, and
# D 15, W 8, M 5, Q 1, Y 1 for the 16 member.
CONFIG_MEMBERS = {64: 30, 32: 31, 16: 30, 128: 6}


def test_the_97_configs_have_the_planned_members():
    """The config name holds the frequency that the eval gives the family
    (`<dataset>/<frequency>/<term>`), or one of the same class."""
    module = load_eval_module()
    names = [module.get_ds_config_name(ds, term)[0]
             for ds, term in module.get_all_dataset_configs()]
    assert len(names) == 97
    members = collections.Counter(family_member(n.split("/")[1])
                                  for n in names)
    assert members == CONFIG_MEMBERS
    small = {n for n in names if n.split("/")[1] in ("Q", "A")}
    assert small == {"m4_quarterly/Q/short", "m4_yearly/A/short"}


def test_the_frequency_of_each_gift_eval_dataset_gives_the_same_member():
    """With the GIFT-Eval data: the frequency that each dataset holds gives
    the member of its config name."""
    data = os.environ.get("GIFT_EVAL") or os.path.expanduser(
        "~/workspaces/gift-eval-data")
    if not os.path.isdir(data):
        pytest.skip("no GIFT-Eval data on this machine")
    os.environ.setdefault("GIFT_EVAL", data)
    module = load_eval_module()
    members = collections.Counter()
    for ds, term in module.get_all_dataset_configs():
        freq = module.GiftDataset(name=ds, term=term, to_univariate=False).freq
        name = module.get_ds_config_name(ds, term)[0]
        assert family_member(freq) == family_member(name.split("/")[1]), name
        members[family_member(freq)] += 1
    assert members == CONFIG_MEMBERS


# ---------------------------------------------------------------------------
# 2. The head
# ---------------------------------------------------------------------------

def standard_head(seed=None, dropout=0.1):
    if seed is not None:
        torch.manual_seed(seed)
    return TransformerQuantileForecastingHead(
        H=16, num_layers=2, nhead=2, forecast_len=16, dropout=dropout)


def family_of(body, members=FAMILY_MEMBERS, seed=1):
    torch.manual_seed(seed)
    return FrequencyFamilyHead(standard_head, members, body)


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def test_each_body_keeps_its_own_state_dict_keys():
    """No key starts with `heads.`: the code of a head bank (#412) does not
    read a family as a bank."""
    heads = family_of("heads").state_dict()
    assert all(re.match(r"freq_heads\.(16|32|64|128)\.", k) for k in heads)
    shared = family_of("shared").state_dict()
    assert all(re.match(r"freq_body\.|freq_out\.(16|32|64|128)\.", k)
               for k in shared)
    assert {k for k in shared if k.startswith("freq_out.")} == {
        f"freq_out.{m}.{p}" for m in FAMILY_MEMBERS
        for p in ("weight", "bias")}
    assert not any("forecast_head" in k for k in shared)
    for sd in (heads, shared):
        assert head_bank_sizes(sd) == ()


def test_the_layout_of_a_family_is_read_off_its_keys():
    for body in BODIES:
        assert family_layout_of(family_of(body).state_dict()) == (
            body, FAMILY_MEMBERS)
        assert family_layout_of(family_of(body, (16, 64)).state_dict()) == (
            body, (16, 64))
    assert family_layout_of(standard_head(0).state_dict()) is None
    bank = ForecastingHeadBank({
        p: TransformerQuantileForecastingHead(H=16, num_layers=1, nhead=2,
                                              forecast_len=p)
        for p in (8, 16)})
    assert family_layout_of(bank.state_dict()) is None


def test_the_shared_body_has_one_output_layer_for_each_member():
    one, out = n_params(standard_head(0)), 16 * Q * 16 + Q * 16
    assert n_params(family_of("shared")) == one + 3 * out
    assert n_params(family_of("heads")) == 4 * one
    shared = family_of("shared")
    assert shared.member(16).transformer is shared.member(128).transformer
    assert shared.member(16).norm is shared.member(128).norm
    assert shared.member(16).forecast_head is not shared.member(128).forecast_head
    heads = family_of("heads")
    assert heads.member(16).transformer is not heads.member(128).transformer


@pytest.mark.parametrize("body", BODIES)
def test_each_member_decodes_the_16_values_of_the_next_patch(body):
    """The key of a member is no patch size: forecast_B4 reads a member as
    it reads the standard head."""
    family = family_of(body).eval()
    latents = torch.randn(3, 5, 16)
    for key in FAMILY_MEMBERS:
        member = family.member(key)
        assert isinstance(member, TransformerQuantileForecastingHead)
        assert member.forecast_len == 16
        assert getattr(member, "patch_size", None) is None
        assert member(latents).shape == (3, 5, Q, 16)
    with pytest.raises(KeyError):
        family.member(8)


@pytest.mark.parametrize("body", BODIES)
def test_each_config_reads_the_member_of_its_frequency(body):
    family = family_of(body)
    for freq, key in (("10S", 128), ("15T", 64), ("H", 32), ("D", 16),
                      ("M", 16), ("Q-DEC", 16), ("A-DEC", 16), (None, 16)):
        got = family.for_frequency(freq)
        assert got.forecast_head is family.member(key).forecast_head, freq


@pytest.mark.parametrize("body", BODIES)
def test_each_row_is_decoded_by_its_member(body):
    family = family_of(body).eval()
    latents = torch.randn(6, 5, 16)
    members = torch.tensor([64, 16, 128, 16, 32, 64])
    out = family(latents, members)
    assert out.shape == (6, 5, Q, 16)
    for row, key in enumerate(members.tolist()):
        alone = family.member(key)(latents[row:row + 1])
        assert torch.allclose(out[row:row + 1], alone, atol=1e-6), row
    other = family(latents, torch.tensor([64, 16, 128, 16, 32, 32]))
    assert torch.allclose(other[:5], out[:5], atol=1e-6)
    assert not torch.allclose(other[5], out[5])


def test_a_row_trains_its_member_only():
    """Rows of the members 16 and 64. With one head for each member, the
    other two heads get no gradient. With the shared body, the body and the
    two output layers get one, and the other two output layers get none."""
    members = torch.tensor([64, 16, 16, 64])
    for body in BODIES:
        family = family_of(body)
        family(torch.randn(4, 5, 16), members).sum().backward()
        for name, p in family.named_parameters():
            key = int(name.split(".")[1]) if "freq_body" not in name else None
            trained = key in (16, 64) or key is None
            assert (p.grad is not None) == trained, name


@pytest.mark.parametrize("body", BODIES)
def test_the_first_member_starts_as_the_standard_head(body):
    """The members are built in increasing order. So with one seed, the 16
    member starts from the weights of the standard head: with the shared
    body, the body and the output layer of 16."""
    standard = standard_head(seed=7).state_dict()
    torch.manual_seed(7)
    family = FrequencyFamilyHead(standard_head, FAMILY_MEMBERS, body)
    first = family_member_state(family.state_dict(), 16)
    assert first.keys() == standard.keys()
    assert all(torch.equal(first[k], standard[k]) for k in standard)
    other = family_member_state(family.state_dict(), 64)
    assert not torch.equal(other["forecast_head.weight"],
                           standard["forecast_head.weight"])


@pytest.mark.parametrize("body", BODIES)
def test_a_member_state_loads_in_a_standard_head(body):
    family = family_of(body).eval()
    latents = torch.randn(2, 5, 16)
    for key in FAMILY_MEMBERS:
        head = standard_head(0).eval()
        head.load_state_dict(family_member_state(family.state_dict(), key))
        assert torch.equal(head(latents), family.member(key)(latents))


@pytest.mark.parametrize("body", BODIES)
def test_a_family_of_one_member_is_the_standard_head(body):
    """The same seed, the same latents and dropout on: the output and the
    gradients of the family are those of the standard head, bit for bit."""
    latents = torch.randn(6, 5, 16)
    standard = standard_head(seed=7)
    torch.manual_seed(11)
    want = standard(latents)
    want.sum().backward()
    torch.manual_seed(7)
    family = FrequencyFamilyHead(standard_head, (16,), body)
    torch.manual_seed(11)
    got = family(latents, torch.full((6,), 16))
    got.sum().backward()
    assert torch.equal(got, want)
    grads = family_member_state(
        {n: p.grad for n, p in family.named_parameters()}, 16)
    for name, p in standard.named_parameters():
        assert torch.equal(grads[name], p.grad), name


@pytest.mark.parametrize("body", BODIES)
def test_the_mode_of_the_family_is_the_mode_of_its_members(body):
    family = family_of(body)
    latents, members = torch.randn(4, 5, 16), torch.tensor([16, 32, 64, 128])
    family.train()
    assert not torch.equal(family(latents, members), family(latents, members))
    assert not torch.equal(family.member(64)(latents),
                           family.member(64)(latents))
    family.eval()
    assert torch.equal(family(latents, members), family(latents, members))
    assert torch.equal(family.member(64)(latents), family.member(64)(latents))


def test_a_family_loads_its_own_state_dict_only():
    for body in BODIES:
        family, again = family_of(body, seed=1), family_of(body, seed=2)
        again.load_state_dict(family.state_dict())
        latents = torch.randn(2, 3, 16)
        members = torch.tensor([32, 128])
        assert torch.equal(family.eval()(latents, members),
                           again.eval()(latents, members))
    with pytest.raises(RuntimeError):
        family_of("heads").load_state_dict(family_of("shared").state_dict())


def test_the_shared_body_is_the_body_of_the_transformer_head():
    def linear_head():
        return LinearQuantileForecastingHead(H=16, forecast_len=16)
    with pytest.raises(ValueError, match="transformer"):
        FrequencyFamilyHead(linear_head, FAMILY_MEMBERS, "shared")
    family = FrequencyFamilyHead(linear_head, FAMILY_MEMBERS, "heads")
    assert family.member(32)(torch.randn(2, 3, 16)).shape == (2, 3, Q, 16)


def test_an_unknown_body_is_refused():
    with pytest.raises(ValueError, match="body"):
        FrequencyFamilyHead(standard_head, FAMILY_MEMBERS, "bank")


# ---------------------------------------------------------------------------
# 3. The B4 forecast of a member
# ---------------------------------------------------------------------------

def tiny_backbone(**changes):
    """A backbone of the kind of BLK at d_model 16: one patch size, the
    EWMA, zero padding and the vocabulary v2."""
    torch.manual_seed(0)
    config = dict(
        C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        rev_norm_kind="ewma", rev_norm_span=128, num_encoder_layers=1,
        freq_emb_dim=3, seasonality_emb_dim=3, num_freqs=len(FREQ_NAMES_V2),
        rev_norm_skip_leading_zeros=True)
    return ConfigurableModel(**dict(config, **changes)).eval()


def walk(n, seed=0, level=50.0):
    g = torch.Generator().manual_seed(seed)
    return level + torch.randn(n, generator=g).cumsum(0)


@pytest.mark.parametrize("body", BODIES)
@pytest.mark.parametrize("freq", ["10S", "15T", "H", "A-DEC"])
def test_b4_reads_the_context_in_patches_of_16_for_each_member(
        monkeypatch, body, freq):
    """The frozen backbone has one patch size. So the member changes the
    decoder only: the latents and the rollout are those of the standard
    head, and the forecast is that of a standard head with the weights of
    the member."""
    seen, real = [], fh.extract_encoder_latents

    def spy(backbone, x, **kw):
        seen.append(kw.get("patch_size"))
        return real(backbone, x, **kw)

    monkeypatch.setattr(fh, "extract_encoder_latents", spy)
    backbone, family = tiny_backbone(), family_of(body).eval()
    ctx = walk(T)[:, None]
    out = forecast_B4(backbone, family.for_frequency(freq), ctx, 45, "cpu")
    assert out.shape == (Q, 45, 1) and np.isfinite(out).all()
    alone = standard_head(0).eval()
    alone.load_state_dict(family_member_state(family.state_dict(),
                                              family_member(freq)))
    assert np.array_equal(out, forecast_B4(backbone, alone, ctx, 45, "cpu"))
    assert seen == [None, None]


@pytest.mark.parametrize("body", BODIES)
def test_two_members_give_two_forecasts(body):
    backbone, family = tiny_backbone(), family_of(body).eval()
    ctx = walk(T)[:, None]
    a = forecast_B4(backbone, family.for_frequency("H"), ctx, 45, "cpu")
    b = forecast_B4(backbone, family.for_frequency("15T"), ctx, 45, "cpu")
    assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# 4. The head trainer
# ---------------------------------------------------------------------------

# The flags that head_eval_bb.sh gives the head trainer for a B4 head, at the
# tiny shape and on the CPU.
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

# The frequency of each source of the test corpus, and its family member.
SOURCES = {"sec": "10S", "min5": "5T", "hour": "H", "day": "D",
           "year": "A-DEC"}


def arm_flags(arm):
    """The family flags of an arm: `control`, or `<body>_<rule>`, with
    `_m<members>` for another member set."""
    if arm == "control":
        return ()
    body, rule, *members = arm.split("_")
    flags = ("--freq-family", body, "--freq-family-rule", rule)
    if members:
        flags += ("--freq-family-members", members[0][1:].replace("-", ","))
    return flags


def trainer_env():
    return dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="",
                OMP_NUM_THREADS="2")


def job_argv(folder, backbone, corpus, arm, *extra):
    return ["--backbone-path", backbone, *HEAD_PROTOCOL, *corpus,
            "--save-dir", str(folder / arm), "--run-name", arm,
            *arm_flags(arm), *extra]


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


def member_rows(stdout):
    """The count of rows of each member that a run prints at its end."""
    line = re.search(r"rows of each member.*", stdout)
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
    root = tmp_path_factory.mktemp("family_corpus")
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
    path = tmp_path_factory.mktemp("family_bb") / "bb.pth"
    torch.save(tiny_backbone().state_dict(), path)
    return str(path)


@pytest.fixture(scope="module")
def solo_runs(tmp_path_factory, backbone_path, corpus):
    """Each arm trained alone: the control and the four family arms."""
    folder = tmp_path_factory.mktemp("family_solo")
    return folder, {arm: train_solo(job_argv(folder, backbone_path, corpus,
                                             arm)) for arm in ARMS}


def load_head_trainer():
    spec = importlib.util.spec_from_file_location("head_trainer_family",
                                                  HEAD_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stream_labels(tmp_path_factory, backbone_path, corpus):
    """The frequency ids of the batches that the runs read: `STEPS` batches
    of the stream of the control job."""
    trainer = load_head_trainer()
    folder = tmp_path_factory.mktemp("family_stream")
    job = trainer.HeadJob(trainer.parse_args(
        job_argv(folder, backbone_path, corpus, "control")))
    batches = iter(job.data_loader())
    return [next(batches)[1] for _ in range(STEPS)]


@pytest.mark.parametrize("arm", ["shared_strict_m16", "shared_draw_m16",
                                 "heads_strict_m16", "heads_draw_m16"])
def test_a_family_of_one_member_writes_the_loss_rows_of_the_standard_head(
        tmp_path, solo_runs, backbone_path, corpus, arm):
    """A family that sends each row to one member is the standard head: the
    same loss rows, and the same weights after the last step, bit for bit.
    """
    folder, runs = solo_runs
    assert runs["control"].returncode == 0, runs["control"].stderr[-3000:]
    r = train_solo(job_argv(tmp_path, backbone_path, corpus, arm))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    want_rows, want_head = run_files(folder, "control")
    rows, head = run_files(tmp_path, arm)
    assert len(rows) == STEPS + 1 and rows == want_rows
    assert family_layout_of(head) == (arm.split("_")[0], (16,))
    member = family_member_state(head, 16)
    assert member.keys() == want_head.keys()
    assert all(torch.equal(member[k], want_head[k]) for k in want_head)
    assert member_rows(r.stdout) == {16: STEPS * BATCH}


@pytest.mark.parametrize("arm", ARMS[1:])
def test_each_family_arm_trains_and_counts_its_rows(solo_runs, stream_labels,
                                                    arm):
    folder, runs = solo_runs
    r = runs[arm]
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    body, rule = arm.split("_")
    assert f"Frequency family: body {body}, rule {rule}" in r.stdout
    rows, head = run_files(folder, arm)
    assert len(rows) == STEPS + 1
    assert np.isfinite([float(row[1]) for row in rows[1:]]).all()
    assert family_layout_of(head) == (body, FAMILY_MEMBERS)
    counts = member_rows(r.stdout)
    assert sum(counts.values()) == STEPS * BATCH
    strict = collections.Counter()
    for ids in stream_labels:
        strict.update(family_row_members(ids, BATCH, "strict").tolist())
    assert len(strict) >= 3, "the test stream must reach several members"
    if rule == "strict":
        assert counts == {m: strict[m] for m in FAMILY_MEMBERS}
    else:
        # Each row draws inside the training range of its frequency. So
        # only a row with the fixed member 16 can draw 16, and only a row
        # with the fixed member 64 or 128 can draw 128.
        assert counts != {m: strict[m] for m in FAMILY_MEMBERS}
        assert counts[16] <= strict[16]
        assert counts[128] <= strict[64] + strict[128]


def test_the_arms_of_one_seed_start_from_the_same_weights(solo_runs):
    """The 16 member of each arm starts as the control head. So after the
    same steps on the same batches, the heads differ by the family only."""
    folder, runs = solo_runs
    assert all(r.returncode == 0 for r in runs.values())
    losses = {arm: [float(r[1]) for r in run_files(folder, arm)[0][1:]]
              for arm in ARMS}
    assert len({tuple(v) for v in losses.values()}) == len(ARMS)
    assert "rows of each member" not in runs["control"].stdout
    assert family_layout_of(run_files(folder, "control")[1]) is None


def test_a_shared_run_gives_each_arm_the_steps_of_its_solo_run(
        tmp_path, solo_runs, backbone_path, corpus):
    """The five arms in one process on one data stream: each arm writes the
    losses and the head of its solo run, bit for bit. So the draws of an arm
    do not move with the other arms of its wave."""
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
    for arm in ARMS[1:]:
        counts = member_rows(runs[arm].stdout)
        line = re.search(rf"\[{arm}\] .*rows of each member.*", r.stdout)
        assert line and member_rows(line.group(0)) == counts, arm


def refused(tmp_path, backbone, corpus, *flags):
    r = train_solo(["--backbone-path", backbone, *HEAD_PROTOCOL, *corpus,
                    "--save-dir", str(tmp_path / "head"), "--run-name", "q",
                    *flags])
    assert r.returncode != 0, r.stdout[-2000:]
    assert not (tmp_path / "head" / "q_final.pth").exists()
    return r.stdout + r.stderr


@pytest.mark.parametrize("flags,why", [
    (("--freq-family", "heads", "--reconstruction", "encoder"),
     "--reconstruction"),
    (("--freq-family", "heads", "--mixed-rollout", "4"), "--mixed-rollout"),
    (("--freq-family", "heads", "--forecast-len", "128"), "--forecast-len"),
    (("--freq-family", "heads", "--head-arch", "transformer-gaussian"),
     "transformer-gaussian"),
    (("--freq-family", "shared", "--head-arch", "linear"), "transformer"),
    (("--freq-family", "heads", "--freq-family-members", "16,24"), "not in"),
    (("--freq-family", "heads", "--freq-family-members", "32,64"),
     "base member 16"),
    (("--freq-family-rule", "draw"), "--freq-family"),
    (("--freq-family-members", "16"), "--freq-family")])
def test_the_trainer_refuses_a_family_that_it_cannot_train(
        tmp_path, backbone_path, corpus, flags, why):
    assert why in refused(tmp_path, backbone_path, corpus, *flags)


def test_a_family_needs_a_quantile_head(tmp_path, backbone_path, corpus):
    flags = [f for f in HEAD_PROTOCOL if f != "--quantile-head"]
    r = train_solo(["--backbone-path", backbone_path, *flags, *corpus,
                    "--save-dir", str(tmp_path / "head"),
                    "--freq-family", "heads"])
    assert r.returncode != 0
    assert "--quantile-head" in r.stdout + r.stderr


@pytest.mark.parametrize("changes,why", [
    (dict(rev_norm_kind="meanstd", rev_norm_span=None), "head bank"),
    (dict(multi_patch_sizes=(8, 16, 32, 64, 128)), "head bank"),
    (dict(freq_emb_dim=0, seasonality_emb_dim=0, num_freqs=10),
     "frequency label")])
def test_a_family_needs_a_backbone_with_one_size_the_ewma_and_labels(
        tmp_path, corpus, changes, why):
    """The trainer of a head bank (#412) and a stream with no labels would
    train one member only, with no error."""
    bb = tmp_path / "bb.pth"
    torch.save(tiny_backbone(**changes).state_dict(), bb)
    flags = [f for f in HEAD_PROTOCOL]
    if "freq_emb_dim" in changes:
        for name in ("--freq-emb-dim", "--seasonality-emb-dim"):
            flags[flags.index(name) + 1] = "0"
    r = train_solo(["--backbone-path", str(bb), *flags, *corpus,
                    "--save-dir", str(tmp_path / "head"),
                    "--freq-family", "heads"])
    assert r.returncode != 0, r.stdout[-2000:]
    assert why in r.stdout + r.stderr


# ---------------------------------------------------------------------------
# 5. The eval script
# ---------------------------------------------------------------------------

# The flags that eval_local.sh gives the GIFT-Eval script, at the tiny shape.
EVAL_PROTOCOL = (
    "--strategy", "B4", "--forecast-len", "16", "--device", "cpu",
    "--t-raw", "4096", "--n-channels", "1", *TINY_SHAPE, "--encoder-type",
    "gru", "--rev-norm-kind", "ewma", "--rev-norm-span", "128",
    "--head-nhead", "2", "--head-causal", "true")


def eval_args(module, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["eval", *EVAL_PROTOCOL, *extra])
    return module.parse_args()


def hourly_item(n=700):
    import pandas as pd
    return {"target": walk(n).numpy(),
            "start": pd.Period("2020-01-01 00:00", freq="h")}


def forecast_of(module, args, backbone, head, item, horizon=48):
    predictor = module.ContrastiveForecasterPredictor(
        backbone=backbone, head=head, prediction_length=horizon, device=CPU,
        strategy="B4", context_pad=args.context_pad)
    return predictor.predict_item(item).forecast_array


@pytest.mark.parametrize("arm", ARMS[1:])
def test_the_eval_forecasts_with_a_trained_family(solo_runs, backbone_path,
                                                  monkeypatch, capsys, arm):
    folder, runs = solo_runs
    assert runs[arm].returncode == 0, runs[arm].stderr[-3000:]
    module = load_eval_module()
    args = eval_args(module, monkeypatch, "--backbone-path", backbone_path,
                     "--head-path", str(folder / arm / f"{arm}_final.pth"))
    backbone, family = module.load_models(args, CPU)
    assert isinstance(family, FrequencyFamilyHead)
    assert (family.body, family.members) == (arm.split("_")[0],
                                             FAMILY_MEMBERS)
    assert not family.training and args.context_pad == "zeros"
    capsys.readouterr()
    head = module.family_config_head(family, "ett1/15T/short", "15T")
    assert head.forecast_head is family.member(64).forecast_head
    assert ("ett1/15T/short: frequency 15T, family member 64"
            in capsys.readouterr().out)
    forecast = forecast_of(module, args, backbone, head, hourly_item())
    assert forecast.shape == (1 + Q, 48) and np.isfinite(forecast).all()
    other = module.family_config_head(family, "ett1/H/short", "H")
    assert not np.allclose(
        forecast, forecast_of(module, args, backbone, other, hourly_item()))


@pytest.mark.parametrize("body", BODIES)
def test_a_family_that_sends_each_config_to_one_member_scores_as_the_standard_head(
        tmp_path, backbone_path, monkeypatch, body):
    """The eval of a family of one member, with the weights of a standard
    head, gives the forecast of that standard head, bit for bit and for
    each frequency."""
    standard = standard_head(seed=5)
    torch.manual_seed(5)
    family = FrequencyFamilyHead(standard_head, (16,), body)
    torch.save(standard.state_dict(), tmp_path / "standard.pth")
    torch.save(family.state_dict(), tmp_path / "family.pth")
    forecasts = {}
    for name in ("standard", "family"):
        module = load_eval_module(f"eval_freq_family_{name}")
        args = eval_args(module, monkeypatch, "--backbone-path",
                         backbone_path, "--head-path",
                         str(tmp_path / f"{name}.pth"))
        backbone, head = module.load_models(args, CPU)
        if name == "family":
            assert head.members == (16,)
            heads = [module.family_config_head(head, f"x/{f}/short", f)
                     for f in ("10S", "H", "A-DEC")]
        else:
            heads = [head]
        forecasts[name] = [forecast_of(module, args, backbone, h,
                                       hourly_item()) for h in heads]
    for forecast in forecasts["family"]:
        assert np.array_equal(forecast, forecasts["standard"][0])


@pytest.mark.parametrize("strategy", ["A2", "R", "B1"])
def test_the_eval_scores_a_family_under_b4_only(tmp_path, backbone_path,
                                                monkeypatch, strategy):
    torch.save(family_of("shared").state_dict(), tmp_path / "family.pth")
    module = load_eval_module()
    args = eval_args(module, monkeypatch, "--backbone-path", backbone_path,
                     "--head-path", str(tmp_path / "family.pth"),
                     "--strategy", strategy)
    with pytest.raises(SystemExit, match="frequency family scores under "
                                         "--strategy B4"):
        module.load_models(args, CPU)


def test_the_eval_refuses_a_family_on_a_backbone_with_patch_sizes(
        tmp_path, monkeypatch):
    """A backbone with patch sizes is scored with its head bank (#412)."""
    bb = tmp_path / "bb.pth"
    torch.save(tiny_backbone(
        multi_patch_sizes=(8, 16, 32, 64, 128)).state_dict(), bb)
    torch.save(family_of("heads").state_dict(), tmp_path / "family.pth")
    module = load_eval_module()
    args = eval_args(module, monkeypatch, "--backbone-path", str(bb),
                     "--head-path", str(tmp_path / "family.pth"))
    with pytest.raises(SystemExit, match="head bank"):
        module.load_models(args, CPU)


def test_the_eval_of_a_standard_head_and_of_a_bank_is_unchanged(
        tmp_path, backbone_path, monkeypatch):
    """A head file with no family key loads as before: one standard head."""
    torch.save(standard_head(seed=5).state_dict(), tmp_path / "standard.pth")
    module = load_eval_module()
    args = eval_args(module, monkeypatch, "--backbone-path", backbone_path,
                     "--head-path", str(tmp_path / "standard.pth"))
    _, head = module.load_models(args, CPU)
    assert type(head) is TransformerQuantileForecastingHead


# ---------------------------------------------------------------------------
# 6. head_eval_bb.sh: the family flags reach the head trainer
# ---------------------------------------------------------------------------

STUB_HEAD = r'''
import json, os, sys
argv = sys.argv[1:]
out = argv[argv.index("--save-dir") + 1]
name = argv[argv.index("--run-name") + 1]
os.makedirs(out, exist_ok=True)
json.dump(argv, open(os.path.join(out, "head_argv.json"), "w"))
open(os.path.join(out, name + "_final.pth"), "w").write("head")
'''

STUB_EVAL = r'''
import json, os, sys
argv = sys.argv[1:]
out = argv[argv.index("--output-dir") + 1]
os.makedirs(out, exist_ok=True)
json.dump(argv, open(os.path.join(out, "argv.json"), "w"))
with open(os.path.join(out, "all_results.csv"), "w") as f:
    f.write("dataset,model,mase\nm4_yearly/A/short,x,1.0\n")
with open(os.path.join(out, "summary.txt"), "w") as f:
    f.write("Aggregate GM-Relative MASE (1 configs): 0.5000\n")
'''


@pytest.fixture
def stub_checkout(tmp_path):
    """A checkout whose head trainer and eval record their flags, and a PATH
    whose nvidia-smi reports nothing."""
    wt = tmp_path / "wt"
    scripts = wt / "experiments" / "2026-04-13_gift-eval" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "train_forecasting_head.py").write_text(STUB_HEAD)
    (scripts / "eval_gift_eval_official.py").write_text(STUB_EVAL)
    (wt / "experiments" / "hf_token.txt").write_text("hf_test\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    (tmp_path / "gift").mkdir()
    bb = tmp_path / "bb.pth"
    bb.write_text("backbone")
    env = dict(os.environ, WT=str(wt), CF373_ROOT=str(tmp_path / "root"),
               CF_RESULTS=str(tmp_path / "res"), GIFT_EVAL=str(tmp_path / "gift"),
               PATH=f"{bin_dir}:{os.environ['PATH']}",
               GPU_GATE_LOCKDIR=str(tmp_path), HEAD_VRAM_MIB="0",
               CF393_EVAL_SLOTDIR=str(tmp_path / "slots"),
               EVAL_CONFIG_FILTER="^m4_yearly/short$", EVAL_EXPECT_CONFIGS="1")
    for key in list(env):
        if key.startswith(("CF_FREQ_FAMILY", "CF_RECONSTRUCTION", "CF_SKIP",
                           "CF_HEAD_", "HEAD_SAVE_EVERY", "EVAL_STRATEGY",
                           "EVAL_DEVICE")):
            env.pop(key)
    return tmp_path, bb, env


def head_eval(stub, tag, **extra):
    tmp_path, bb, env = stub
    return subprocess.run(
        ["bash", str(B4_SCRIPTS / "head_eval_bb.sh"), tag, str(bb), "student",
         "30000"], capture_output=True, text=True, env=dict(env, **extra),
        timeout=300)


def recorded(path):
    return json.load(open(path))


def without_names(argv):
    """The flags of a head with no folder and no name."""
    argv = list(argv)
    for flag in ("--save-dir", "--run-name"):
        del argv[argv.index(flag):argv.index(flag) + 2]
    return argv


def test_the_family_is_the_only_change_of_the_head_flags(stub_checkout):
    """Each flag of the standard B4 head stays, also the flags with no
    effect. The family flags come after them. The score is B4."""
    tmp_path = stub_checkout[0]
    r = head_eval(stub_checkout, "run_bb200k_h30k_control")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    control = recorded(tmp_path / "root" / "eval" / "run_bb200k_h30k_control"
                       / "head_argv.json")
    assert "--freq-family" not in control
    for body, rule in (("shared", "strict"), ("shared", "draw"),
                       ("heads", "strict"), ("heads", "draw")):
        tag = f"run_bb200k_h30k_ff_{body}_{rule}"
        r = head_eval(stub_checkout, tag, CF_FREQ_FAMILY=body,
                      CF_FREQ_FAMILY_RULE=rule)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        out = tmp_path / "root" / "eval" / tag
        family = ["--freq-family", body, "--freq-family-rule", rule]
        assert without_names(recorded(out / "head_argv.json")) == (
            without_names(control) + family)
        shard = recorded(out / "gift" / "shard_0" / "argv.json")
        assert shard[shard.index("--strategy") + 1] == "B4"
        assert (tmp_path / "res" / f"score_{tag}.txt").read_text().strip() == "0.5000"


def test_the_rule_of_a_family_is_strict_unless_the_caller_asks(stub_checkout):
    tmp_path = stub_checkout[0]
    tag = "run_bb200k_h30k_ff_heads_strict"
    r = head_eval(stub_checkout, tag, CF_FREQ_FAMILY="heads", CF_SKIP_EVAL="1")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    argv = recorded(tmp_path / "root" / "eval" / tag / "head_argv.json")
    assert argv[argv.index("--freq-family-rule") + 1] == "strict"


def test_a_member_set_reaches_the_head_and_the_tag(stub_checkout):
    tmp_path = stub_checkout[0]
    tag = "run_bb200k_h30k_ff_shared_strict_m16-64"
    r = head_eval(stub_checkout, tag, CF_FREQ_FAMILY="shared",
                  CF_FREQ_FAMILY_MEMBERS="16,64", CF_SKIP_EVAL="1")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    argv = recorded(tmp_path / "root" / "eval" / tag / "head_argv.json")
    assert argv[argv.index("--freq-family-members") + 1] == "16,64"


@pytest.mark.parametrize("tag,knobs", [
    # A family head with the tag of a standard head would take its files.
    ("run_bb200k_h30k_control", dict(CF_FREQ_FAMILY="heads")),
    ("run_bb200k_h30k_ff_heads_strict",
     dict(CF_FREQ_FAMILY="heads", CF_FREQ_FAMILY_RULE="draw")),
    ("run_bb200k_h30k_ff_heads_strict",
     dict(CF_FREQ_FAMILY="heads", CF_FREQ_FAMILY_MEMBERS="16")),
    ("run_bb200k_h30k_ff_bank_strict", dict(CF_FREQ_FAMILY="bank")),
    ("run_bb200k_h30k_ff_heads_uniform",
     dict(CF_FREQ_FAMILY="heads", CF_FREQ_FAMILY_RULE="uniform")),
    # A rule with no family would train a standard head, with no error.
    ("run_bb200k_h30k_control", dict(CF_FREQ_FAMILY_RULE="draw")),
    ("run_bb200k_h30k_control", dict(CF_FREQ_FAMILY_MEMBERS="16")),
    ("run_bb200k_h30k_ff_heads_strict_recon",
     dict(CF_FREQ_FAMILY="heads", CF_RECONSTRUCTION="encoder"))])
def test_a_wrong_family_call_is_refused(stub_checkout, tag, knobs):
    tmp_path = stub_checkout[0]
    r = head_eval(stub_checkout, tag, **knobs)
    assert r.returncode != 0, r.stdout[-2000:]
    assert "ABORT" in r.stdout + r.stderr
    assert not (tmp_path / "root" / "eval" / tag / "head_argv.json").exists()


def test_the_argv_mode_hands_the_family_flags_to_a_shared_trainer(
        stub_checkout):
    tmp_path = stub_checkout[0]
    jobs = tmp_path / "jobs.jsonl"
    r = head_eval(stub_checkout, "run_bb200k_h30k_ff_shared_draw",
                  CF_FREQ_FAMILY="shared", CF_FREQ_FAMILY_RULE="draw",
                  CF_HEAD_ARGV_TO=str(jobs))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    (argv,) = [json.loads(line) for line in jobs.read_text().splitlines()]
    assert argv[-4:] == ["--freq-family", "shared",
                         "--freq-family-rule", "draw"]


# ---------------------------------------------------------------------------
# 7. The scripts of the report
# ---------------------------------------------------------------------------

RUNNER_STUB = r'''#!/bin/bash
# A runner that records each call. CF_HEAD_ARGV_TO: the flags of the head.
# Else: the score of a head that exists.
tag="$1"; out="$CF373_ROOT/eval/$tag"; mkdir -p "$out"
family="${CF_FREQ_FAMILY:-none}/${CF_FREQ_FAMILY_RULE:-none}/${CF_FREQ_FAMILY_MEMBERS:-all}"
if [ -n "${CF_HEAD_ARGV_TO:-}" ]; then
  echo "$tag argv $family $4 $2" >>"$CF_RESULTS/calls.log"
  printf '["--save-dir", "%s", "--run-name", "qhead_%s"]\n' "$out" "$tag" \
    >>"$CF_HEAD_ARGV_TO"
  exit 0
fi
echo "$tag score $family ${EVAL_DEVICE:-cpu} ${BB_GPU:-none}" >>"$CF_RESULTS/calls.log"
[ -n "${FF_TEST_SCORE_FAILS:-}" ] && exit 1
echo 1.1000 >"$CF_RESULTS/score_$tag.txt"
'''

TRAINER_STUB = r'''
import json, os, sys
jobs = [json.loads(line) for line in open(sys.argv[sys.argv.index("--jobs") + 1])]
res = os.environ["FF_TEST_RES"]
names = [a[a.index("--run-name") + 1][len("qhead_"):] for a in jobs]
with open(os.path.join(res, "calls.log"), "a") as f:
    f.write("wave " + os.environ.get("CUDA_VISIBLE_DEVICES", "unset") + " "
            + os.environ.get("HF_TOKEN", "no-token") + " "
            + " ".join(names) + "\n")
print("[shared] 1 steps: a stub", flush=True)
if os.environ.get("FF_TEST_TRAIN_FAILS"):
    sys.exit(1)
for a, name in zip(jobs, names):
    out = a[a.index("--save-dir") + 1]
    open(os.path.join(out, f"qhead_{name}_final.pth"), "w").write("head")
'''

COLLECT_STUB = r'''
import os, sys
with open(os.path.join(os.environ["FF_TEST_RES"], "calls.log"), "a") as f:
    f.write("collect " + " ".join(sys.argv[1:]) + "\n")
'''


@pytest.fixture
def wave_box(tmp_path):
    """A base folder with a stub runner, a stub shared trainer, a stub
    collect script and a backbone file."""
    base = tmp_path / "base"
    res = base / "results"
    code = base / "code"
    (code / "experiments").mkdir(parents=True)
    (code / "experiments" / "hf_token.txt").write_text("hf_test\n")
    (code / "DEPLOYED_COMMIT").write_text("abcdef12\n")
    res.mkdir()
    for name, text in (("runner.sh", RUNNER_STUB), ("trainer.py", TRAINER_STUB),
                       ("collect.py", COLLECT_STUB)):
        (tmp_path / name).write_text(text)
    bb = tmp_path / "bb_200k.pth"
    bb.write_text("backbone")
    with open(tmp_path / "seasonal_naive.csv", "wb") as f:
        f.truncate(24831)
    (tmp_path / "gift").mkdir()
    env = dict(os.environ, FF_BASE=str(base),
               FF_RUNNER=str(tmp_path / "runner.sh"),
               FF_TRAINER=str(tmp_path / "trainer.py"),
               FF_COLLECT=str(tmp_path / "collect.py"),
               FF_SN_REF=str(tmp_path / "seasonal_naive.csv"),
               GIFT_EVAL=str(tmp_path / "gift"), FF_WAVE_VRAM_MIB="0",
               FF_TEST_RES=str(res))
    for key in list(env):
        if key.startswith("FF_") and key not in (
                "FF_BASE", "FF_RUNNER", "FF_TRAINER", "FF_COLLECT",
                "FF_SN_REF", "FF_WAVE_VRAM_MIB", "FF_TEST_RES"):
            env.pop(key)
    return tmp_path, base, bb, env


def run_wave(box, *arms, run="blk", stop="200", **extra):
    tmp_path, base, bb, env = box
    return subprocess.run(
        ["bash", str(SCRIPTS / "run_wave.sh"), run, stop, str(bb), *arms],
        capture_output=True, text=True, env=dict(env, **extra), timeout=120)


def calls(base):
    path = base / "results" / "calls.log"
    return [line.split() for line in open(path)] if path.exists() else []


def tag_of(arm, run="blk", stop="200", steps="30k"):
    family = "" if arm == "control" else "ff_"
    return f"{run}_bb{stop}k_h{steps}_{family}{arm}"


def test_a_wave_trains_its_arms_on_one_stream_and_scores_each_head(wave_box):
    tmp_path, base, bb, _ = wave_box
    r = run_wave(wave_box)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    log = calls(base)
    waves = [c for c in log if c[0] == "wave"]
    assert len(waves) == 1
    # One process trains the five arms, on GPU 1, with the token.
    assert waves[0][1:3] == ["1", "hf_test"]
    assert waves[0][3:] == [tag_of(arm) for arm in ARMS]
    argv = {c[0]: c[2:] for c in log if c[1] == "argv"}
    assert argv[tag_of("control")] == ["none/none/all", "30000", str(bb)]
    assert argv[tag_of("shared_draw")] == ["shared/draw/all", "30000", str(bb)]
    assert argv[tag_of("heads_strict")] == ["heads/strict/all", "30000",
                                            str(bb)]
    scores = {c[0]: c[2:] for c in log if c[1] == "score"}
    assert set(scores) == {tag_of(arm) for arm in ARMS}
    assert scores[tag_of("heads_draw")][0] == "heads/draw/all"
    assert [c for c in log if c[0] == "collect"]
    order = [c[0] if c[0] in ("wave", "collect") else c[1] for c in log]
    assert order.index("wave") < order.index("score") < order.index("collect")
    arms = [line.split("\t") for line in open(base / "results" / "arms.tsv")]
    assert [a[:4] for a in arms[1:]] == [
        ["blk", "200", arm, tag_of(arm)] for arm in ARMS]


def test_a_second_start_skips_the_work_that_is_done(wave_box):
    base = wave_box[1]
    assert run_wave(wave_box).returncode == 0
    first = calls(base)
    r = run_wave(wave_box)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    new = calls(base)[len(first):]
    assert [c[0] for c in new] == ["collect"]
    arms = (base / "results" / "arms.tsv").read_text().splitlines()
    assert len(arms) == 1 + len(ARMS)


def test_a_head_with_no_score_gets_its_score_only(wave_box):
    base = wave_box[1]
    r = run_wave(wave_box, FF_SCORE="0")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert not [c for c in calls(base) if "score" in c[:2]]
    assert not [c for c in calls(base) if c[0] == "collect"]
    first = calls(base)
    r = run_wave(wave_box, FF_TRAIN="0")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    new = calls(base)[len(first):]
    assert not [c for c in new if c[0] == "wave" or c[1] == "argv"]
    assert len([c for c in new if c[1] == "score"]) == len(ARMS)


def test_a_wave_trains_the_arms_that_it_gets(wave_box):
    base = wave_box[1]
    r = run_wave(wave_box, "control", "shared_strict", run="abc_gift",
                 stop="40")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    (wave,) = [c for c in calls(base) if c[0] == "wave"]
    assert wave[3:] == [tag_of("control", "abc_gift", "40"),
                        tag_of("shared_strict", "abc_gift", "40")]
    # A later start with one more arm trains that arm only.
    r = run_wave(wave_box, "control", "shared_strict", "heads_draw",
                 run="abc_gift", stop="40")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    waves = [c for c in calls(base) if c[0] == "wave"]
    assert waves[1][3:] == [tag_of("heads_draw", "abc_gift", "40")]


def test_an_arm_of_another_member_set_names_it_in_its_tag(wave_box):
    base = wave_box[1]
    r = run_wave(wave_box, "control", "heads_strict_m16", FF_HEAD_STEPS="500",
                 FF_SCORE="0")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    argv = {c[0]: c[2:] for c in calls(base) if c[1] == "argv"}
    tag = tag_of("heads_strict_m16", steps="500")
    assert argv[tag][:2] == ["heads/strict/16", "500"]
    assert tag_of("control", steps="500") in argv


@pytest.mark.parametrize("arm", ["bank_strict", "shared", "heads_uniform",
                                 "shared_draw_16", "Control"])
def test_an_unknown_arm_is_refused(wave_box, arm):
    base = wave_box[1]
    r = run_wave(wave_box, "control", arm)
    assert r.returncode != 0
    assert "ABORT" in r.stdout + r.stderr and not calls(base)


def test_a_wave_refuses_missing_inputs(wave_box):
    tmp_path, base, bb, env = wave_box
    for knob in ({"FF_SN_REF": str(tmp_path / "none.csv")},
                 {"GIFT_EVAL": str(tmp_path / "none")},
                 {"FF_RUNNER": str(tmp_path / "none.sh")}):
        r = run_wave(wave_box, **knob)
        assert r.returncode != 0 and "ABORT" in r.stdout + r.stderr, knob
    os.remove(bb)
    r = run_wave(wave_box)
    assert r.returncode != 0 and "ABORT" in r.stdout + r.stderr
    assert not calls(base)


def test_a_wave_waits_for_the_stream_and_its_wait_has_an_end(wave_box):
    """One wave reads the stream at a time: the token has a request limit."""
    base = wave_box[1]
    (base / "locks").mkdir()
    with open(base / "locks" / "stream.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        r = run_wave(wave_box, FF_STREAM_WAIT="1")
    assert r.returncode != 0
    assert "stream" in r.stdout + r.stderr
    assert not [c for c in calls(base) if c[0] == "wave"]


def test_a_wave_of_another_base_folder_waits_for_the_named_stream_lock(
        wave_box):
    """The test wave has its own base folder. It names the stream lock of
    the waves, so it reads no stream while a wave trains."""
    tmp_path, base = wave_box[:2]
    with open(tmp_path / "waves_stream.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        r = run_wave(wave_box, FF_STREAM_WAIT="1",
                     FF_STREAM_LOCK=str(tmp_path / "waves_stream.lock"))
    assert r.returncode != 0 and "stream" in r.stdout + r.stderr
    assert not [c for c in calls(base) if c[0] == "wave"]
    r = run_wave(wave_box, FF_STREAM_LOCK=str(tmp_path / "waves_stream.lock"))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def test_a_wave_with_no_head_after_its_trainer_scores_nothing(wave_box):
    base = wave_box[1]
    r = run_wave(wave_box, FF_TEST_TRAIN_FAILS="1")
    assert r.returncode != 0
    assert not [c for c in calls(base) if c[1] == "score"]
    # The next start trains the wave again.
    r = run_wave(wave_box)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert len([c for c in calls(base) if c[0] == "wave"]) == 2


def test_a_score_that_fails_fails_the_wave_and_keeps_the_other_scores(
        wave_box):
    base = wave_box[1]
    r = run_wave(wave_box, FF_TEST_SCORE_FAILS="1")
    assert r.returncode != 0
    assert len([c for c in calls(base) if c[1] == "score"]) == len(ARMS)
    r = run_wave(wave_box)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert len([c for c in calls(base) if c[0] == "wave"]) == 1


def test_the_scores_run_on_the_cpu_unless_the_caller_asks(wave_box):
    """The standard score runs on the CPU, and a GPU of elisa gives another
    fifth digit of the score. So the GPU is a knob, and not the default."""
    base = wave_box[1]
    r = run_wave(wave_box, "control")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert [c for c in calls(base) if c[1] == "score"][0][3:] == ["cpu", "1"]
    r = run_wave(wave_box, "shared_draw", FF_EVAL_DEVICE="cuda", FF_GPU="0")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    log = calls(base)
    assert [c for c in log if c[0] == "wave"][-1][1] == "0"
    assert [c for c in log if c[1] == "score"][-1][3:] == ["cuda", "0"]


WAVE_STUB = r'''#!/bin/bash
# A wave that records its call, and makes the next checkpoint when asked.
echo "wave train=${FF_TRAIN:-1} score=${FF_SCORE:-1} $*" >>"$FF_TEST_RES/calls.log"
[ -z "${FF_TEST_WAVE_FAILS:-}" ] || [ "${FF_TRAIN:-1}" = 0 ] || exit 1
exit 0
'''


def save_checkpoint(folder, name, optimizer=True):
    """A checkpoint that loads, with its optimizer file."""
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"w": torch.zeros(2)}, folder / f"{name}.pth")
    if optimizer:
        torch.save({"step": 1}, folder / f"{name}_optimizer.pth")


@pytest.fixture
def follow_box(tmp_path):
    base, ckpt = tmp_path / "base", tmp_path / "abc" / "ckpt"
    (base / "results").mkdir(parents=True)
    ckpt.mkdir(parents=True)
    (tmp_path / "wave.sh").write_text(WAVE_STUB)
    env = dict(os.environ, FF_BASE=str(base), FF_WAVE=str(tmp_path / "wave.sh"),
               FF_ABC_CKPT=str(ckpt), FF_POLL="0.2", FF_SETTLE="0",
               FF_WAIT_MAX="20", FF_TEST_RES=str(base / "results"))
    for key in ("FF_STOPS", "FF_TRAIN", "FF_SCORE", "FF_RUN"):
        env.pop(key, None)
    return tmp_path, base, ckpt, env


def run_follow(box, *arms, timeout=120, **extra):
    tmp_path, base, ckpt, env = box
    return subprocess.run(
        ["bash", str(SCRIPTS / "follow_abc_gift.sh"), *arms],
        capture_output=True, text=True, env=dict(env, **extra),
        timeout=timeout)


def test_the_follower_starts_one_wave_for_each_stop_in_order(follow_box):
    tmp_path, base, ckpt, _ = follow_box
    for stop in (40, 100):
        save_checkpoint(ckpt, f"abc_gift_{stop}k")
    r = run_follow(follow_box, "control", "shared_strict", FF_STOPS="40 100")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    log = calls(base)
    trains = [c for c in log if c[1] == "train=1"]
    # The heads train one stop after the other. The scores of a stop start
    # after its heads, and do not stop the next wave.
    assert [c[1:3] for c in trains] == [["train=1", "score=0"]] * 2
    assert [c[3:5] for c in trains] == [["abc_gift", "40"], ["abc_gift", "100"]]
    assert all(c[6:] == ["control", "shared_strict"] for c in log)
    scores = [c for c in log if c[1] == "train=0"]
    assert sorted(c[4] for c in scores) == ["100", "40"]
    # The wave reads a copy of the checkpoint in the folder of the family.
    for c in log:
        assert c[5] == str(base / "ckpt" / "abc_gift" / f"abc_gift_{c[4]}k.pth")
        assert os.path.getsize(c[5]) == os.path.getsize(
            ckpt / f"abc_gift_{c[4]}k.pth")


def test_the_follower_waits_for_a_checkpoint_that_comes_later(follow_box):
    tmp_path, base, ckpt, _ = follow_box

    def later():
        time.sleep(1.5)
        save_checkpoint(ckpt, "abc_gift_40k")

    thread = threading.Thread(target=later)
    thread.start()
    r = run_follow(follow_box, FF_STOPS="40")
    thread.join()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "waiting" in r.stdout
    assert [c[4] for c in calls(base) if c[1] == "train=1"] == ["40"]


def test_each_wait_of_the_follower_has_an_end(follow_box):
    tmp_path, base, ckpt, _ = follow_box
    save_checkpoint(ckpt, "abc_gift_40k")
    # 100k has no optimizer file: it is not complete.
    save_checkpoint(ckpt, "abc_gift_100k", optimizer=False)
    save_checkpoint(ckpt, "abc_gift_140k")
    r = run_follow(follow_box, FF_STOPS="40 100 140", FF_WAIT_MAX="2")
    assert r.returncode != 0
    assert "100k" in r.stdout + r.stderr
    # The follower stops at the stop that does not come.
    assert [c[4] for c in calls(base) if c[1] == "train=1"] == ["40"]


def test_the_follower_takes_the_checkpoint_of_the_last_start_that_loads(
        follow_box):
    """After a new start, the trainer names its files `<name>_r2`. A cut
    file does not load, so the follower does not take it."""
    tmp_path, base, ckpt, _ = follow_box
    save_checkpoint(ckpt, "abc_gift_40k")
    save_checkpoint(ckpt, "abc_gift_r2_40k")
    (ckpt / "abc_gift_r3_40k.pth").write_bytes(b"PK\x03\x04 cut")
    (ckpt / "abc_gift_r3_40k_optimizer.pth").write_bytes(b"PK\x03\x04 cut")
    torch.save({"w": torch.ones(5)}, ckpt / "abc_gift_r2_40k.pth")
    r = run_follow(follow_box, FF_STOPS="40")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "abc_gift_r2_40k.pth" in r.stdout
    copy = base / "ckpt" / "abc_gift" / "abc_gift_40k.pth"
    assert torch.equal(torch.load(copy, weights_only=True)["w"],
                       torch.ones(5))


def test_a_wave_that_fails_does_not_stop_the_next_stop(follow_box):
    tmp_path, base, ckpt, _ = follow_box
    for stop in (40, 100):
        save_checkpoint(ckpt, f"abc_gift_{stop}k")
    r = run_follow(follow_box, FF_STOPS="40 100", FF_TEST_WAVE_FAILS="1")
    assert r.returncode != 0
    trains = [c[4] for c in calls(base) if c[1] == "train=1"]
    assert trains == ["40", "100"]
    assert not [c for c in calls(base) if c[1] == "train=0"]


def test_the_default_stops_are_the_abc_stops(follow_box):
    text = (SCRIPTS / "follow_abc_gift.sh").read_text()
    assert "40 100 140 180 200 240 300 360 400 460" in text


# The score tables.

def load_collect():
    spec = importlib.util.spec_from_file_location(
        "collect_scores_family", SCRIPTS / "collect_scores.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_names():
    module = load_eval_module("eval_freq_family_names")
    return [module.get_ds_config_name(ds, term)[0]
            for ds, term in module.get_all_dataset_configs()]


def write_eval(base, run, stop, arm, score, names, mase, members=True,
               wrong=None):
    """The files of one scored arm: its line of the arm table, its score,
    the table of the eval and the member lines of its shard logs."""
    res, tag = base / "results", tag_of(arm, run, str(stop))
    res.mkdir(parents=True, exist_ok=True)
    table = res / "arms.tsv"
    if not table.exists():
        table.write_text("run\tstop_k\tarm\ttag\thead_steps\tbackbone\n")
    with open(table, "a") as f:
        f.write(f"{run}\t{stop}\t{arm}\t{tag}\t30000\t/ckpt/bb.pth\n")
    (res / f"score_{tag}.txt").write_text(f"{score}\n")
    gift = base / "heads" / "eval" / tag / "gift"
    (gift / "shard_0").mkdir(parents=True)
    with open(gift / "all_results.csv", "w") as f:
        f.write("dataset,model,eval_metrics/MSE[mean],eval_metrics/MASE[0.5]\n")
        for name in names:
            f.write(f"{name},contrastive_tiny,1.0,{mase}\n")
    if arm != "control" and members:
        with open(gift / "shard_0" / "shard.log", "w") as f:
            for name in names:
                freq = name.split("/")[1]
                key = (wrong or {}).get(name, family_member(freq))
                f.write(f"  [eval] {name}: frequency {freq}, "
                        f"family member {key}\n  [  1/97] {name}  MASE=1.0\n")


def read_tsv(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


def test_collect_writes_one_row_for_each_scored_arm(tmp_path):
    names = config_names()
    base = tmp_path / "base"
    write_eval(base, "blk", 200, "control", "1.1300", names, 2.0)
    write_eval(base, "blk", 200, "shared_strict", "1.1074", names, 1.5)
    write_eval(base, "abc_gift", 40, "control", "1.2000", names, 3.0)
    # An arm with no score yet has no row.
    with open(base / "results" / "arms.tsv", "a") as f:
        f.write("abc_gift\t40\theads_draw\t" + tag_of("heads_draw", "abc_gift",
                                                      "40")
                + "\t30000\t/ckpt/bb.pth\n")
    collect = load_collect()
    assert collect.main(["--base", str(base)]) == 0
    rows = read_tsv(base / "results" / "scores.tsv")
    assert [(r["run"], r["stop_k"], r["arm"]) for r in rows] == [
        ("abc_gift", "40", "control"), ("blk", "200", "control"),
        ("blk", "200", "shared_strict")]
    family = rows[2]
    assert family["gm_rel_mase"] == "1.1074" and family["configs"] == "97"
    # The family against the control of its wave.
    assert float(family["ratio_to_control"]) == pytest.approx(1.1074 / 1.13,
                                                              abs=1e-4)
    assert rows[1]["ratio_to_control"] == "1.0000"
    assert family["members"] == "16:30 32:31 64:30 128:6"
    assert rows[1]["members"] == ""
    per_config = read_tsv(base / "results" / "config_mase.tsv")
    assert len(per_config) == 3 * 97
    one = [r for r in per_config if r["arm"] == "shared_strict"
           and r["config"] == "ett1/15T/short"]
    assert [(r["run"], r["stop_k"], r["mase"], r["member"]) for r in one] == [
        ("blk", "200", "1.5", "64")]
    assert {r["member"] for r in per_config if r["arm"] == "control"} == {""}


def test_collect_refuses_a_family_score_with_other_members(tmp_path, capsys):
    """The members of the 97 configs are fixed. Another count shows that
    the eval did not select the decoder by the frequency."""
    names = config_names()
    base = tmp_path / "base"
    write_eval(base, "blk", 200, "heads_draw", "1.1000", names, 1.5,
               wrong={"ett1/15T/short": 16})
    collect = load_collect()
    assert collect.main(["--base", str(base)]) != 0
    assert "heads_draw" in capsys.readouterr().out
    base = tmp_path / "base2"
    write_eval(base, "blk", 200, "heads_draw", "1.1000", names, 1.5,
               members=False)
    assert collect.main(["--base", str(base)]) != 0


def test_collect_reads_a_score_of_a_few_configs(tmp_path):
    """A test score on a few configs has no fixed member count."""
    names = config_names()[:4]
    base = tmp_path / "base"
    write_eval(base, "blk", 200, "heads_strict", "0.9000", names, 1.5)
    collect = load_collect()
    assert collect.main(["--base", str(base)]) == 0
    (row,) = read_tsv(base / "results" / "scores.tsv")
    assert row["configs"] == "4" and row["ratio_to_control"] == ""


# The parity script of the test wave.

def load_compare():
    spec = importlib.util.spec_from_file_location(
        "compare_parity_family", SCRIPTS / "compare_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_parity_script_counts_the_identical_loss_rows(tmp_path):
    header = "step,loss,hf_rows_consumed\n"
    (tmp_path / "a.csv").write_text(header + "1,0.5,256\n2,0.4,512\n")
    (tmp_path / "b.csv").write_text(header + "1,0.5,256\n2,0.25,512\n")
    compare = load_compare()
    same = compare.compare_losses("x", tmp_path / "a.csv", tmp_path / "a.csv")
    assert "2 and 2 loss rows, 2 identical" in same
    assert "max |loss difference| 0" in same
    other = compare.compare_losses("x", tmp_path / "a.csv", tmp_path / "b.csv")
    assert "1 identical" in other and "0.15" in other


@pytest.mark.parametrize("body", BODIES)
def test_the_parity_script_reads_the_member_16_of_a_family_head(tmp_path,
                                                                body):
    standard = standard_head(seed=5)
    torch.manual_seed(5)
    family = FrequencyFamilyHead(standard_head, (16,), body)
    torch.save(standard.state_dict(), tmp_path / "control.pth")
    torch.save(family.state_dict(), tmp_path / "family.pth")
    compare = load_compare()
    n = len(standard.state_dict())
    line = compare.compare_heads("x", tmp_path / "family.pth",
                                 tmp_path / "control.pth")
    assert f"{n} tensors in the control head, {n} identical" in line
    with torch.no_grad():
        family.member(16).forecast_head.weight.add_(0.5)
    torch.save(family.state_dict(), tmp_path / "family.pth")
    line = compare.compare_heads("x", tmp_path / "family.pth",
                                 tmp_path / "control.pth")
    assert f"{n - 1} identical" in line and "0.5" in line


def test_the_parity_script_compares_two_eval_tables(tmp_path):
    header = "dataset,model,eval_metrics/MASE[0.5]\n"
    (tmp_path / "a.csv").write_text(header + "c1,m,1.0\nc2,m,2.0\n")
    (tmp_path / "b.csv").write_text(header + "c1,m,1.0\nc2,m,1.0\n")
    compare = load_compare()
    same = compare.compare_tables("x", tmp_path / "a.csv", tmp_path / "a.csv")
    assert "2 with the same MASE" in same and "1.000000" in same
    other = compare.compare_tables("x", tmp_path / "a.csv", tmp_path / "b.csv")
    assert "1 with the same MASE" in other
    # The geometric mean of the ratios 1 and 2.
    assert f"{2 ** 0.5:.6f}" in other


# The code folder of the waves.

def deploy(tmp_path):
    token = tmp_path / "token.txt"
    token.write_text("hf_test\n")
    env = dict(os.environ, FF_BASE=str(tmp_path / "base"),
               FF_HF_TOKEN_FILE=str(token))
    return subprocess.run(["bash", str(SCRIPTS / "deploy.sh")],
                          capture_output=True, text=True, env=env, timeout=120)


def test_the_deploy_puts_the_committed_code_in_the_code_folder(tmp_path):
    """deploy.sh: the code of the last commit, the name of that commit and
    the Hugging Face token, in the code folder that the waves read. A
    restart of elisa keeps that folder."""
    r = deploy(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    code = tmp_path / "base" / "code"
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse",
                           "--short=8", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    assert (code / "DEPLOYED_COMMIT").read_text().strip() == head
    assert (code / "experiments" / "hf_token.txt").read_text() == "hf_test\n"
    report = "reports/2026-10-10_frequency_family_head/scripts/"
    for rel in (report + "run_wave.sh", report + "follow_abc_gift.sh",
                report + "collect_scores.py",
                "reports/2026-08-08_rollout_depth/scripts/head_eval_bb.sh",
                "reports/2026-08-08_rollout_depth/results/config_costs.csv",
                "experiments/2026-04-13_gift-eval/scripts/"
                "train_forecasting_heads_shared.py",
                "src/freq_family.py"):
        assert (code / rel).is_file(), rel
    assert not (code / "tests").exists()
    (code / "stale.txt").write_text("an old file")
    assert deploy(tmp_path).returncode == 0     # a new deploy replaces it
    assert not (code / "stale.txt").exists()


def test_the_deploy_changes_no_code_under_a_process_that_runs_it(tmp_path):
    """bash reads a script while it runs it. So the deploy refuses when a
    process runs a file of the code folder."""
    assert deploy(tmp_path).returncode == 0
    code = tmp_path / "base" / "code"
    (code / "mark.txt").write_text("the code of a wave")
    (code / "wave.py").write_text("import time\ntime.sleep(60)\n")
    wave = subprocess.Popen([sys.executable, str(code / "wave.py")])
    try:
        r = deploy(tmp_path)
    finally:
        wave.kill()
        wave.wait()
    assert r.returncode != 0 and "ABORT" in r.stdout + r.stderr
    assert (code / "mark.txt").exists()
