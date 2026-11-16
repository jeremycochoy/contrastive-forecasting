"""--head-block-trunks separate: a block head with one full head per block
size, with no shared trunk (src/block_head.py).

1. The head: one trunk per size under the keys `block_body.<P>.*`, the same
   start weights in each trunk, and the keys of the shared bank by default.
2. The training path: a row reads the trunk of its own size only, in the
   forward, in the loss and in the gradients.
3. The trainer: the flag, its refusal with no bank, and a run on the CPU.
4. The eval: the mode read off the keys, and the same member outputs after
   a save and a load.
5. The wave scripts and the score tables: the arm `blocks_sep`.
"""

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from src.block_head import (BLOCK_SIZES, BlockSizeHead, block_head_loss,
                            block_head_sizes, block_size_of, block_trunk_mode,
                            block_trunk_state)
from tests.test_freq_family import (calls, config_names, head_eval,  # noqa: F401
                                    load_collect, read_tsv, recorded,
                                    run_wave, stub_checkout, tag_of, wave_box,
                                    without_names, write_eval)
from tests.test_head_block_sizes import (BATCH, CPU, PATCHES, STEPS,  # noqa: F401
                                         H, Q, W, backbone_path, bank_of,
                                         batch, corpus, eval_args,
                                         forecast_of, head_args, hourly_item,
                                         job_argv, layout, load_eval_module,
                                         load_head_trainer, losses, n_params,
                                         run_files, size_rows, standard_head,
                                         tiny_backbone, train_solo)

SEPARATE = ("--head-block-trunks", "separate")
ROW_SIZES = [16, 32, 64, 32, 128, 16]


def separate_bank(sizes=BLOCK_SIZES, seed=1, dropout=0.0):
    torch.manual_seed(seed)
    return BlockSizeHead(standard_head(dropout=dropout), sizes, "separate")


def distinct_bank(seed=1):
    """A separate bank in eval mode whose trunks have different weights, as
    after a training."""
    bank = separate_bank(seed=seed).eval()
    torch.manual_seed(seed + 100)
    with torch.no_grad():
        for p in bank.block_body.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return bank


def shift_trunk(bank, size, by=0.05):
    with torch.no_grad():
        for p in bank.block_body[str(size)].parameters():
            p.add_(by)


def mixed_rows():
    """``(seq, mask, start, x_norm, rows)`` of a batch with every size."""
    x = batch(n=len(ROW_SIZES))
    return (*layout(tiny_backbone(), x), torch.tensor(ROW_SIZES))


# ---------------------------------------------------------------------------
# 1. The head
# ---------------------------------------------------------------------------

def test_the_default_bank_keeps_the_keys_of_the_shared_trunk():
    """The keys of today: `block_body.<trunk key>` and `block_out.<P>.*`."""
    standard = standard_head(0).state_dict()
    want = {f"block_body.{k}" for k in standard
            if not k.startswith("forecast_head.")}
    want |= {f"block_out.{p}.{part}" for p in BLOCK_SIZES
             for part in ("weight", "bias")}
    for bank in (bank_of(), BlockSizeHead(standard_head(0), BLOCK_SIZES,
                                          "shared")):
        assert bank.trunks == "shared"
        assert set(bank.state_dict()) == want
        assert block_trunk_mode(bank.state_dict()) == "shared"


def test_a_separate_bank_has_one_trunk_per_size_under_its_own_keys():
    bank = separate_bank()
    sd = bank.state_dict()
    assert all(re.match(r"block_body\.(16|32|64|128)\.(transformer|norm)\.|"
                        r"block_out\.(16|32|64|128)\.(weight|bias)$", k)
               for k in sd)
    trunk_keys = {k.split(".", 2)[2] for k in sd
                  if k.startswith("block_body.")}
    for p in BLOCK_SIZES:
        assert {k.split(".", 2)[2] for k in sd
                if k.startswith(f"block_body.{p}.")} == trunk_keys
    assert block_head_sizes(sd) == BLOCK_SIZES
    assert block_trunk_mode(sd) == "separate"
    assert list(bank.block_body) == [str(p) for p in BLOCK_SIZES]
    assert block_head_sizes(separate_bank((16, 64)).state_dict()) == (16, 64)
    assert list(separate_bank((16, 64)).block_body) == ["16", "64"]


def test_a_separate_bank_has_the_parameters_of_one_trunk_per_size():
    shared, bank = bank_of(), separate_bank()
    trunk = n_params(shared.block_body)
    assert n_params(bank) == n_params(shared) + (len(BLOCK_SIZES) - 1) * trunk
    trunks = [bank.trunk(p) for p in BLOCK_SIZES]
    assert len({id(t) for t in trunks}) == len(BLOCK_SIZES)
    assert len({id(p) for t in trunks for p in t.parameters()}) == (
        len(BLOCK_SIZES) * len(list(shared.block_body.parameters())))


def test_each_trunk_starts_from_the_weights_of_the_shared_bank():
    """With one seed, each trunk has the weights of the shared trunk, and
    the output layers those of the shared bank, bit for bit."""
    shared = bank_of(seed=7).state_dict()
    sd = separate_bank(seed=7).state_dict()
    for key, value in shared.items():
        if key.startswith("block_out."):
            assert torch.equal(sd[key], value), key
            continue
        for p in BLOCK_SIZES:
            name = key.replace("block_body.", f"block_body.{p}.")
            assert torch.equal(sd[name], value), name


def test_the_trunk_state_of_a_separate_bank_is_the_trunk_of_16():
    standard = standard_head(seed=7).state_dict()
    bank = separate_bank(seed=7)
    shift_trunk(bank, 32)
    trunk = block_trunk_state(bank.state_dict())
    assert trunk.keys() == standard.keys()
    assert all(torch.equal(trunk[k], standard[k]) for k in standard)
    standard_head(0).load_state_dict(trunk)


def test_a_bank_refuses_an_unknown_trunk_mode():
    with pytest.raises(ValueError, match="shared"):
        BlockSizeHead(standard_head(0), BLOCK_SIZES, "each")


# ---------------------------------------------------------------------------
# 2. The training path
# ---------------------------------------------------------------------------

def test_the_shapes_of_the_forward_and_of_the_loss():
    seq, mask, start, x_norm, rows = mixed_rows()
    bank = separate_bank().eval()
    features = bank.features(seq, mask, rows)
    assert features.shape == (len(ROW_SIZES), seq.shape[1], H)
    for size in BLOCK_SIZES:
        out = bank(seq, size, mask, start)
        assert out.shape == (len(ROW_SIZES), PATCHES // (size // W), Q, size)
    loss = block_head_loss(bank, features[:, start:], rows, x_norm)
    assert loss.shape == () and loss.isfinite() and loss > 0


def test_separate_trunks_need_the_size_of_each_row():
    seq, mask, _, _, _ = mixed_rows()
    with pytest.raises(ValueError, match="row_sizes"):
        separate_bank().features(seq, mask)


def test_the_shared_trunk_ignores_the_row_sizes():
    seq, mask, _, _, rows = mixed_rows()
    bank = bank_of().eval()
    assert torch.equal(bank.features(seq, mask, rows),
                       bank.features(seq, mask))


def test_the_features_of_a_row_come_from_the_trunk_of_its_size():
    """Each row keeps its place, under the (T, T) mask of the layout."""
    seq, mask, _, _, rows = mixed_rows()
    assert mask.shape == (seq.shape[1], seq.shape[1])
    bank = distinct_bank()
    features = bank.features(seq, mask, rows)
    for i, size in enumerate(ROW_SIZES):
        want = bank.trunk(size).features(seq[i:i + 1], mask)[0]
        assert torch.allclose(features[i], want, atol=1e-6), (i, size)
        other = bank.trunk(64 if size != 64 else 16).features(
            seq[i:i + 1], mask)[0]
        assert not torch.allclose(features[i], other, atol=1e-4), (i, size)


def test_a_change_of_the_trunk_of_32_moves_the_rows_of_32_only():
    seq, mask, start, x_norm, rows = mixed_rows()
    bank = distinct_bank()
    before = bank.features(seq, mask, rows)
    shift_trunk(bank, 32)
    after = bank.features(seq, mask, rows)
    for i, size in enumerate(ROW_SIZES):
        same = torch.equal(before[i], after[i])
        assert same == (size != 32), (i, size)
    for i, size in enumerate(ROW_SIZES):
        one = torch.tensor([size])
        losses_of = [block_head_loss(bank, f[i:i + 1, start:], one,
                                     x_norm[i:i + 1]) for f in (before, after)]
        assert torch.equal(*losses_of) == (size != 32), (i, size)


def test_a_separate_bank_with_equal_trunks_is_the_shared_bank():
    """Before the first step, the two banks of one seed give one loss."""
    seq, mask, start, x_norm, rows = mixed_rows()
    shared, bank = bank_of(seed=7).eval(), separate_bank(seed=7).eval()
    want = block_head_loss(shared, shared.features(seq, mask)[:, start:],
                           rows, x_norm)
    got = block_head_loss(bank, bank.features(seq, mask, rows)[:, start:],
                          rows, x_norm)
    assert torch.allclose(got, want, atol=1e-6)


def test_a_row_trains_its_own_trunk_and_its_own_output_layer():
    seq, mask, start, x_norm, _ = mixed_rows()
    rows = torch.tensor([16, 64, 16, 64, 64, 16])
    bank = separate_bank()
    block_head_loss(bank, bank.features(seq, mask, rows)[:, start:], rows,
                    x_norm).backward()
    for name, p in bank.named_parameters():
        trained = name.split(".")[1] in ("16", "64")
        assert (p.grad is not None) == trained, name


def test_the_forward_of_a_size_reads_the_trunk_of_that_size():
    seq, mask, start, _, _ = mixed_rows()
    bank = distinct_bank()
    before = {p: bank(seq, p, mask, start) for p in BLOCK_SIZES}
    shift_trunk(bank, 32)
    for p in BLOCK_SIZES:
        assert torch.equal(before[p], bank(seq, p, mask, start)) == (p != 32)


@pytest.mark.parametrize("size", BLOCK_SIZES)
def test_a_member_owns_the_trunk_and_the_output_layer_of_its_size(size):
    bank = distinct_bank()
    member = bank.member(size)
    assert member.transformer is bank.block_body[str(size)].transformer
    assert member.norm is bank.block_body[str(size)].norm
    assert member.forecast_head is bank.block_out[str(size)]
    assert member.forecast_len == size and not member.training
    latents = torch.randn(3, 24, H)
    assert member(latents).shape == (3, 24 // (size // W), Q, size)
    assert torch.equal(member(latents), bank(latents, size))


# ---------------------------------------------------------------------------
# 3. The head trainer
# ---------------------------------------------------------------------------

def test_the_flag_defaults_to_the_shared_trunk():
    trainer = load_head_trainer()
    args = head_args(trainer, "--head-block-sizes", "16,32")
    assert args.head_block_trunks == "shared"
    args = head_args(trainer, "--head-block-sizes", "16,32", *SEPARATE)
    assert args.head_block_trunks == "separate"
    assert trainer.block_sizes_of_flags(args, False, True) == (16, 32)


def test_the_trainer_refuses_separate_trunks_with_no_bank():
    trainer = load_head_trainer()
    with pytest.raises(SystemExit, match="--head-block-sizes"):
        trainer.block_sizes_of_flags(head_args(trainer, *SEPARATE), False,
                                     True)


def test_the_trainer_builds_the_bank_of_the_flag(tmp_path):
    trainer = load_head_trainer()
    bb = tmp_path / "bb.pth"
    torch.save(tiny_backbone().state_dict(), bb)
    kinds = {}
    for trunks in ("shared", "separate"):
        args = head_args(trainer, "--head-block-sizes", "16,32,64,128",
                         "--head-block-trunks", trunks)
        args.backbone_path = str(bb)
        args.save_dir = str(tmp_path / trunks)
        job = trainer.HeadJob(args)
        assert job.head.trunks == trunks
        kinds[trunks] = set(job.head.state_dict())
    assert block_trunk_mode(dict.fromkeys(kinds["shared"])) == "shared"
    assert block_trunk_mode(dict.fromkeys(kinds["separate"])) == "separate"


@pytest.fixture(scope="module")
def separate_run(tmp_path_factory, backbone_path, corpus):
    """The bank with separate trunks, trained alone on the CPU."""
    folder = tmp_path_factory.mktemp("blocks_sep_solo")
    run = train_solo(job_argv(folder, backbone_path, corpus, "blocks",
                              *SEPARATE))
    return folder, run


def test_a_separate_bank_trains_on_the_cpu(separate_run):
    folder, r = separate_run
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "with no shared weight" in r.stdout
    rows, head = run_files(folder, "blocks")
    assert len(rows) == STEPS + 1
    assert np.isfinite(losses(rows)).all() and (losses(rows) > 0).all()
    assert block_head_sizes(head) == BLOCK_SIZES
    assert block_trunk_mode(head) == "separate"
    counts = size_rows(r.stdout)
    assert sum(counts.values()) == STEPS * BATCH
    # The trunks start equal. Each one that read a row has moved alone.
    moved = [p for p in BLOCK_SIZES if counts[p] > 0]
    assert len(moved) >= 3
    key = "norm.weight"
    trunks = [head[f"block_body.{p}.{key}"] for p in moved]
    assert all(not torch.equal(trunks[0], t) for t in trunks[1:])


# ---------------------------------------------------------------------------
# 4. The eval
# ---------------------------------------------------------------------------

def test_the_eval_loads_a_separate_bank_off_its_keys(tmp_path, backbone_path,
                                                     monkeypatch, capsys):
    """A save and a load through the eval give the same member outputs."""
    bank = distinct_bank()
    torch.save(bank.state_dict(), tmp_path / "blocks_sep.pth")
    module = load_eval_module("eval_blocks_sep")
    args = eval_args(module, monkeypatch, "--backbone-path", backbone_path,
                     "--head-path", str(tmp_path / "blocks_sep.pth"))
    backbone, blocks = module.load_models(args, CPU)
    assert isinstance(blocks, BlockSizeHead) and blocks.sizes == BLOCK_SIZES
    assert blocks.trunks == "separate" and not blocks.training
    assert "block head: one trunk per block size, one output layer per " \
        "block size" in capsys.readouterr().out
    loaded, saved = blocks.state_dict(), bank.state_dict()
    assert loaded.keys() == saved.keys()
    assert all(torch.equal(loaded[k], saved[k]) for k in saved)
    latents = torch.randn(3, 24, H)
    for freq, size in (("10S", 128), ("15T", 64), ("H", 32), ("D", 16)):
        head = module.block_config_head(blocks, f"x/{freq}/short", freq)
        assert head.block_size == size
        assert head.transformer is blocks.block_body[str(size)].transformer
        assert head.forecast_head is blocks.block_out[str(size)]
        assert torch.equal(head(latents), bank.member(size)(latents))
        want = forecast_of(module, args, backbone, bank.member(size),
                           hourly_item())
        got = forecast_of(module, args, backbone, head, hourly_item())
        assert got.shape == (1 + Q, 48) and np.isfinite(got).all()
        np.testing.assert_array_equal(got, want)


def test_the_eval_forecasts_with_a_trained_separate_bank(separate_run,
                                                         backbone_path,
                                                         monkeypatch):
    folder, r = separate_run
    assert r.returncode == 0, r.stderr[-3000:]
    module = load_eval_module("eval_blocks_sep_trained")
    head_path = folder / "blocks" / "blocks_final.pth"
    args = eval_args(module, monkeypatch, "--backbone-path", backbone_path,
                     "--head-path", str(head_path))
    backbone, blocks = module.load_models(args, CPU)
    assert blocks.sizes == BLOCK_SIZES and blocks.trunks == "separate"
    for freq in ("10S", "5T", "H", "D"):
        head = module.block_config_head(blocks, f"x/{freq}/short", freq)
        forecast = forecast_of(module, args, backbone, head, hourly_item())
        assert forecast.shape == (1 + Q, 48) and np.isfinite(forecast).all()


def test_the_eval_keeps_the_shared_bank(tmp_path, backbone_path, monkeypatch,
                                        capsys):
    torch.save(bank_of().state_dict(), tmp_path / "blocks.pth")
    module = load_eval_module("eval_blocks_shared_mode")
    args = eval_args(module, monkeypatch, "--backbone-path", backbone_path,
                     "--head-path", str(tmp_path / "blocks.pth"))
    _, blocks = module.load_models(args, CPU)
    assert blocks.trunks == "shared"
    assert "block head: one trunk, one output layer per block size" in (
        capsys.readouterr().out)


# ---------------------------------------------------------------------------
# 5. The wave scripts and the score tables: the arm `blocks_sep`
# ---------------------------------------------------------------------------

SIZES_KNOB = "16,32,64,128"
BLOCK_FLAGS = ["--head-block-sizes", SIZES_KNOB]
SEP_TAG = "run_bb200k_h30k_ff_blocks_sep"


def test_the_trunk_knob_adds_the_trunk_flag_after_the_block_flag(
        stub_checkout):
    tmp_path = stub_checkout[0]
    r = head_eval(stub_checkout, "run_bb200k_h30k_control")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    control = recorded(tmp_path / "root" / "eval" / "run_bb200k_h30k_control"
                       / "head_argv.json")
    assert "--head-block-trunks" not in control
    r = head_eval(stub_checkout, SEP_TAG, CF_HEAD_BLOCK_SIZES=SIZES_KNOB,
                  CF_HEAD_BLOCK_TRUNKS="separate")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = tmp_path / "root" / "eval" / SEP_TAG
    assert without_names(recorded(out / "head_argv.json")) == (
        without_names(control) + BLOCK_FLAGS + list(SEPARATE))
    shard = recorded(out / "gift" / "shard_0" / "argv.json")
    assert shard[shard.index("--strategy") + 1] == "B4"


def test_the_shared_knob_gives_the_flags_of_the_arm_blocks(stub_checkout):
    tmp_path = stub_checkout[0]
    tag = "run_bb200k_h30k_ff_blocks"
    for knobs in ({}, {"CF_HEAD_BLOCK_TRUNKS": "shared"}):
        r = head_eval(stub_checkout, tag, CF_HEAD_BLOCK_SIZES=SIZES_KNOB,
                      **knobs)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        argv = recorded(tmp_path / "root" / "eval" / tag / "head_argv.json")
        assert "--head-block-trunks" not in argv
        assert argv[argv.index("--head-block-sizes") + 1] == SIZES_KNOB


@pytest.mark.parametrize("tag,knobs", [
    # Separate trunks under the tag of the shared bank would take its files.
    ("run_bb200k_h30k_ff_blocks",
     dict(CF_HEAD_BLOCK_SIZES=SIZES_KNOB, CF_HEAD_BLOCK_TRUNKS="separate")),
    (SEP_TAG, dict(CF_HEAD_BLOCK_SIZES=SIZES_KNOB)),
    (SEP_TAG, dict(CF_HEAD_BLOCK_TRUNKS="separate")),
    ("run_bb200k_h30k_control", dict(CF_HEAD_BLOCK_TRUNKS="separate")),
    (SEP_TAG, dict(CF_HEAD_BLOCK_SIZES=SIZES_KNOB,
                   CF_HEAD_BLOCK_TRUNKS="each")),
    (SEP_TAG, dict(CF_HEAD_BLOCK_SIZES=SIZES_KNOB,
                   CF_HEAD_BLOCK_TRUNKS="separate", CF_FREQ_FAMILY="heads"))])
def test_a_wrong_trunk_call_is_refused(stub_checkout, tag, knobs):
    tmp_path = stub_checkout[0]
    r = head_eval(stub_checkout, tag, **knobs)
    assert r.returncode != 0, r.stdout[-2000:]
    assert "ABORT" in r.stdout + r.stderr
    assert not (tmp_path / "root" / "eval" / tag / "head_argv.json").exists()


def test_a_wave_trains_the_separate_arm_beside_the_other_arms(wave_box):
    base = wave_box[1]
    r = run_wave(wave_box, "control", "blocks", "blocks_sep", FF_SCORE="0",
                 CF_HEAD_BLOCK_TRUNKS="separate")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    (wave,) = [c for c in calls(base) if c[0] == "wave"]
    assert wave[3:] == [tag_of("control"), tag_of("blocks"),
                        tag_of("blocks_sep")]
    assert tag_of("blocks_sep").endswith("_ff_blocks_sep")


def test_the_separate_arm_sets_the_two_block_knobs():
    script = (Path(__file__).resolve().parents[1] / "reports"
              / "2026-10-10_frequency_family_head" / "scripts" / "run_wave.sh")
    body = script.read_text()
    start = body.index("arm_env(){")
    func = body[start:body.index("\n}\n", start) + 3]
    knobs = f"CF_HEAD_BLOCK_SIZES={SIZES_KNOB}"
    for arm, want in (("blocks_sep", f"{knobs} CF_HEAD_BLOCK_TRUNKS=separate"),
                      ("blocks", knobs), ("control", "")):
        r = subprocess.run(["bash", "-c", func + f"arm_env {arm}"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == want, (arm, r.stdout)
    r = subprocess.run(["bash", "-c", func + "arm_env blocks_other"],
                       capture_output=True, text=True)
    assert r.returncode != 0


def write_block_lines(base, arm, names, wrong=None):
    """The block-size line of the eval of each config of the arm ``arm``."""
    log = (base / "heads" / "eval" / tag_of(arm, "blk", "200") / "gift"
           / "shard_0" / "shard.log")
    with open(log, "w") as f:
        for name in names:
            freq = name.split("/")[1]
            size = (wrong or {}).get(name, block_size_of(freq))
            f.write(f"  [eval] {name}: frequency {freq}, block size {size} "
                    f"({size // 16} latents per block)\n")


def test_collect_scores_the_separate_arm_beside_the_arm_blocks(tmp_path):
    names = config_names()
    base = tmp_path / "base"
    write_eval(base, "blk", 200, "control", "1.0000", names, 2.0)
    for arm, score in (("blocks_sep", "1.0200"), ("blocks", "1.0500")):
        write_eval(base, "blk", 200, arm, score, names, 2.1, members=False)
        write_block_lines(base, arm, names)
    collect = load_collect()
    assert collect.ARM_ORDER[-2:] == ("blocks", "blocks_sep")
    assert collect.main(["--base", str(base)]) == 0
    scores = read_tsv(base / "results" / "scores.tsv")
    assert [r["arm"] for r in scores] == ["control", "blocks", "blocks_sep"]
    rows = {r["arm"]: r for r in scores}
    assert rows["blocks_sep"]["ratio_to_control"] == "1.0200"
    assert rows["blocks"]["ratio_to_control"] == "1.0500"
    assert rows["blocks_sep"]["members"] == rows["blocks"]["members"] == (
        "16:30 32:31 64:30 128:6")
    per_config = read_tsv(base / "results" / "config_mase.tsv")
    one = [r for r in per_config if r["arm"] == "blocks_sep"
           and r["config"] == "ett1/15T/short"]
    assert [r["member"] for r in one] == ["64"]


def test_collect_refuses_a_separate_score_with_another_block_size(tmp_path,
                                                                  capsys):
    names = config_names()
    base = tmp_path / "base"
    write_eval(base, "blk", 200, "blocks_sep", "1.0200", names, 2.1,
               members=False)
    write_block_lines(base, "blocks_sep", names, wrong={"ett1/15T/short": 16})
    assert load_collect().main(["--base", str(base)]) != 0
    out = capsys.readouterr().out
    assert "blocks_sep" in out and "block size of their frequency" in out
