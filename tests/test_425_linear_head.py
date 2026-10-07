"""Tests for #425: the reconstruction with a LINEAR head.

The linear head is one linear map from the encoder latent e_t to the
quantiles of the values of patch t (``--head-arch linear``). Each other
setting is that of the B4 head, and the score is strategy R with no change.
A linear head of a checkpoint has its own tag (``..._recon_lin``), so it
shares no file with the transformer head of that checkpoint.

Groups, all on the CPU:

1. The head trainer: ``--reconstruction encoder --head-arch linear`` on each
   kind of run of jobs.tsv, and the shared trainer.
2. The score: strategy R decodes with a linear head, and the eval builds the
   linear head from the head file.
3. The shell runner ``head_eval_bb.sh``: ``CF_HEAD_ARCH=linear``.
4. The queue of the box: ``CF425_HEAD_ARCH=linear`` beside the queue of the
   transformer heads.
5. The sync of the box: its own folders, and the roots that the prune gets.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from src.forecasting_head import (ForecastingHeadBank,
                                  LinearQuantileForecastingHead,
                                  reconstruct_windows)
from tests import test_425_encoder_reconstruction as base
from tests.test_425_encoder_reconstruction import (  # noqa: F401
    B4_SCRIPTS, CPU, HEAD_PROTOCOL, OLD_DATA, Q, REPO_ROOT, SCRIPTS, SIZES,
    assert_same_run, bank_model, corpus_flags, ewma_bank_model, ewma_model,
    eval_args, head_eval, job_calls, load_eval_module, meanstd_model,
    oracle_latents, queue_box, recorded, run_queue, saved, shifted,
    stub_checkout, sync_box, train_shared, train_solo, walk, waves)

# The flags of the transformer head in head_eval_bb.sh. A linear head has
# none of them.
TRANSFORMER_FLAGS = ("--head-arch", "--head-num-layers", "--head-nhead",
                     "--head-ffn-mult", "--head-causal", "--head-train-input",
                     "--head-dropout")


def linear_protocol(protocol=HEAD_PROTOCOL):
    """The flags head_eval_bb.sh gives the head trainer under
    CF_HEAD_ARCH=linear: those of the B4 head, with ``--head-arch linear``
    in place of the flags of the transformer head."""
    flags, skip = [], False
    for flag in protocol:
        if skip:
            skip = False
        elif flag in TRANSFORMER_FLAGS:
            skip = True
        else:
            flags.append(flag)
    return (*flags, "--head-arch", "linear")


LINEAR_PROTOCOL = linear_protocol()


def linear_keys(sizes=()):
    """The keys of a linear head file: one weight and one bias, for each
    patch size of a bank."""
    names = ("forecast_head.weight", "forecast_head.bias")
    if not sizes:
        return set(names)
    return {f"heads.{p}.{name}" for p in sizes for name in names}


def train_linear(tmp_path, backbone, *extra):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="2")
    return subprocess.run(
        [sys.executable, str(base.HEAD_PY), "--backbone-path", backbone,
         *LINEAR_PROTOCOL, "--save-dir", str(tmp_path / "head"),
         "--run-name", "qlin", *extra],
        capture_output=True, text=True, env=env, timeout=900)


def final_head(tmp_path, name="qlin"):
    return torch.load(tmp_path / "head" / f"{name}_final.pth",
                      map_location="cpu", weights_only=True)


def losses_of(tmp_path, name="qlin"):
    rows = csv.DictReader(open(tmp_path / "head" / f"{name}_losses.csv"))
    return [float(r["loss"]) for r in rows]


# ---------------------------------------------------------------------------
# 1. The head trainer
# ---------------------------------------------------------------------------

def test_the_linear_protocol_holds_no_flag_of_the_transformer_head():
    assert LINEAR_PROTOCOL[-2:] == ("--head-arch", "linear")
    assert LINEAR_PROTOCOL.count("--head-arch") == 1
    assert not set(TRANSFORMER_FLAGS[1:]) & set(LINEAR_PROTOCOL)
    kept = [f for f in HEAD_PROTOCOL if f in ("--quantile-head", "--lr",
                                              "--batch-size", "--seed",
                                              "--reconstruction")]
    assert all(flag in LINEAR_PROTOCOL for flag in kept)


# One run of each kind of jobs.tsv: the model, the flags of its data stream
# (None: the stream of #419) and the patch sizes of its head bank.
RUN_KINDS = {
    "patch sizes, mean/std (O* runs)": (bank_model, None, SIZES),
    "patch sizes, EWMA (OEF)": (ewma_bank_model, None, SIZES),
    "one size, mean/std (BMS)": (meanstd_model, None, (16,)),
    "one size, EWMA, zero padding (BLK)": (
        lambda: ewma_model(zero_pad=True), None, ()),
    "one size, EWMA, old data": (ewma_model, OLD_DATA, ()),
}


@pytest.mark.parametrize("kind", RUN_KINDS)
def test_the_trainer_trains_a_linear_head_on_each_kind_of_run(
        tmp_path, corpus_flags, kind):
    """The head file holds one linear map for each patch size and nothing
    else: no transformer layer."""
    make, stream, sizes = RUN_KINDS[kind]
    bb = saved(tmp_path, "bb.pth", make())
    r = train_linear(tmp_path, bb, *(corpus_flags if stream is None else stream))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "RECONSTRUCTION mode: encoder" in r.stdout
    assert "linear-probe quantile" in r.stdout
    sd = final_head(tmp_path)
    assert set(sd) == linear_keys(sizes)
    for size in sizes or (16,):
        prefix = f"heads.{size}." if sizes else ""
        assert sd[prefix + "forecast_head.weight"].shape == (Q * size, 16)
        assert sd[prefix + "forecast_head.bias"].shape == (Q * size,)
    losses = losses_of(tmp_path)
    assert len(losses) == 3 and np.isfinite(losses).all()


def test_the_linear_head_learns_from_its_first_step(tmp_path):
    """Each step moves the weights: the head after 3 steps is not the head
    of the same seed after 1 step."""
    bb = saved(tmp_path, "bb.pth", ewma_model())
    heads = {}
    for steps in ("1", "3"):
        r = train_linear(tmp_path / steps, bb, *OLD_DATA, "--total-steps",
                         steps)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        heads[steps] = final_head(tmp_path / steps)
    assert not torch.equal(heads["1"]["forecast_head.weight"],
                           heads["3"]["forecast_head.weight"])


def linear_argv(folder, backbone, name, *extra):
    """The flags of one solo run of a linear head, four steps."""
    return ["--backbone-path", backbone, *LINEAR_PROTOCOL, "--total-steps",
            "4", "--save-dir", str(folder / name), "--run-name", name, *extra]


def solo_then_shared(tmp_path, jobs, *stream):
    """Train each job alone, then all the jobs in one shared run.
    ``jobs``: ``{name: (model, argv builder)}``."""
    argvs = []
    for name, (model, argv_of) in jobs.items():
        bb = saved(tmp_path, f"{name}.pth", model)
        r = train_solo(argv_of(tmp_path / "solo", bb, name, *stream))
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        argvs.append(argv_of(tmp_path / "shared", bb, name, *stream))
    r = train_shared(tmp_path, argvs)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for name in jobs:
        assert_same_run(tmp_path / "solo", tmp_path / "shared", name)
    return r


def test_a_shared_run_gives_each_linear_job_the_steps_of_its_solo_run(
        tmp_path, corpus_flags):
    """A wave of linear heads on the stream of #419, one job of each kind:
    each job writes the losses and the head of its solo run, to the last
    bit."""
    jobs = {"bank": (bank_model(), linear_argv),
            "ewma_bank": (ewma_bank_model(), linear_argv),
            "ewma": (ewma_model(zero_pad=True), linear_argv),
            "meanstd": (meanstd_model(), linear_argv)}
    r = solo_then_shared(tmp_path, jobs, *corpus_flags)
    assert "4 jobs on one stream" in r.stdout


def test_a_shared_run_of_old_linear_jobs_gives_their_solo_steps(tmp_path):
    jobs = {"old_a": (ewma_model(), linear_argv),
            "old_b": (shifted(ewma_model(), 0.01), linear_argv)}
    solo_then_shared(tmp_path, jobs, *OLD_DATA)


def test_a_wave_trains_a_linear_and_a_transformer_head_of_one_checkpoint(
        tmp_path, corpus_flags):
    """The two heads of one checkpoint in one shared run: each one is the
    head of its solo run. Each job has its own pass of the backbone."""
    jobs = {"lin": (bank_model(), linear_argv),
            "tfm": (bank_model(), base.job_argv)}
    solo_then_shared(tmp_path, jobs, *corpus_flags)
    _, lin = base.run_files(tmp_path / "shared", "lin")
    _, tfm = base.run_files(tmp_path / "shared", "tfm")
    assert set(lin) == linear_keys(SIZES)
    assert any(".transformer.layers." in key for key in tfm)


# ---------------------------------------------------------------------------
# 2. The score: strategy R with a linear head
# ---------------------------------------------------------------------------

def identity_head(size, bank_member=True):
    """A linear head that gives each quantile of position t the latent of
    position t: with :func:`oracle_latents`, the exact values of patch t."""
    head = LinearQuantileForecastingHead(H=size, forecast_len=size).eval()
    with torch.no_grad():
        head.forecast_head.weight.copy_(torch.eye(size).repeat(Q, 1))
        head.forecast_head.bias.zero_()
    if bank_member:
        head.patch_size = size
    return head


@pytest.mark.parametrize("make,size", [(bank_model, 8), (bank_model, 64),
                                       (ewma_bank_model, 32),
                                       (meanstd_model, 16)])
def test_a_perfect_linear_head_gives_back_the_true_horizon(monkeypatch, make,
                                                           size):
    """Strategy R reads the output of a linear head as it reads the output
    of the B4 head: quantile q of value l of patch t."""
    import src.forecasting_head as fh
    monkeypatch.setattr(fh, "extract_encoder_latents", oracle_latents)
    ctx = torch.stack([walk(1024, seed=s) for s in range(3)])[..., None]
    future = torch.stack([walk(50, seed=10 + s, level=60.0)
                          for s in range(3)])[..., None]
    out = reconstruct_windows(make(), identity_head(size), ctx, future, CPU)
    assert out.shape == (3, Q, 50, 1)
    want = future.numpy()[:, None].repeat(Q, axis=1)
    assert np.allclose(out, want, rtol=1e-4, atol=1e-3)


def test_the_eval_builds_a_linear_bank_from_the_head_file(tmp_path,
                                                          monkeypatch):
    module = load_eval_module()
    torch.manual_seed(5)
    trained = ForecastingHeadBank(
        {p: LinearQuantileForecastingHead(H=16, forecast_len=p) for p in SIZES})
    bb = saved(tmp_path, "bb.pth", bank_model())
    head = saved(tmp_path, "head.pth", trained)
    args = eval_args(module, monkeypatch, "--backbone-path", bb,
                     "--head-path", head, "--strategy", "R")
    _, bank = module.load_models(args, CPU)
    assert isinstance(bank, ForecastingHeadBank) and bank.sizes == SIZES
    for size in SIZES:
        got = bank.head_for(size)
        assert isinstance(got, LinearQuantileForecastingHead)
        assert got.patch_size == got.forecast_len == size
        assert torch.equal(got.forecast_head.weight,
                           trained.head_for(size).forecast_head.weight)


def test_the_eval_builds_one_linear_head_for_a_run_of_one_size(tmp_path,
                                                               monkeypatch):
    module = load_eval_module()
    bb = saved(tmp_path, "bb.pth", ewma_model(zero_pad=True))
    head = saved(tmp_path, "head.pth",
                 LinearQuantileForecastingHead(H=16, forecast_len=16))
    args = eval_args(module, monkeypatch, "--backbone-path", bb,
                     "--head-path", head, "--strategy", "R")
    _, got = module.load_models(args, CPU)
    assert isinstance(got, LinearQuantileForecastingHead)
    assert got.forecast_len == 16


def test_the_eval_reconstructs_with_a_trained_linear_bank(tmp_path,
                                                          corpus_flags,
                                                          monkeypatch):
    """The trainer and the eval agree on the file of a linear bank: the
    eval loads the head that the trainer saved, and R gives a finite
    forecast of each quantile."""
    import pandas as pd
    from gluonts.dataset.util import forecast_start
    bb = saved(tmp_path, "bb.pth", bank_model())
    r = train_linear(tmp_path, bb, *corpus_flags)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    module = load_eval_module()
    args = eval_args(module, monkeypatch, "--backbone-path", bb,
                     "--head-path", str(tmp_path / "head" / "qlin_final.pth"),
                     "--strategy", "R")
    backbone, bank = module.load_models(args, CPU)
    item = {"target": walk(700).numpy(),
            "start": pd.Period("2020-01-01 00:00", freq="h")}
    label = {"target": walk(48, seed=7).numpy(), "start": forecast_start(item)}
    predictor = module.ReconstructionPredictor(
        backbone=backbone, head=bank.for_frequency(backbone, "h"),
        prediction_length=48, device=CPU, strategy="R",
        context_pad=args.context_pad, labels=[label])
    (forecast,) = list(predictor.predict([item]))
    assert forecast.forecast_array.shape == (1 + Q, 48)
    assert np.isfinite(forecast.forecast_array).all()


# ---------------------------------------------------------------------------
# 3. The shell runner: CF_HEAD_ARCH=linear in head_eval_bb.sh
# ---------------------------------------------------------------------------

LINEAR = dict(CF_RECONSTRUCTION="encoder", CF_HEAD_ARCH="linear")
# The flags of the B4 head in head_eval_bb.sh before the linear head, in
# their order.
B4_ARCH = ["--head-arch", "transformer", "--head-num-layers", "2",
           "--head-nhead", "8", "--head-ffn-mult", "4.0", "--head-causal",
           "true", "--head-train-input", "e_then_f", "--head-dropout", "0.1"]


def b4_argv(bb, out, tag, recon=False):
    """The flags head_eval_bb.sh gave the head trainer before the linear
    head: the B4 forecast head, or the reconstruction head of #425."""
    return ["--backbone-path", str(bb), "--encoder-source", "student",
            "--device", "cuda", "--quantile-head", "--grad-clip", "1.0",
            "--forecast-len", "16", "--batch-size", "256", "--lr", "1e-3",
            "--total-steps", "30000", "--save-every", "5000", "--log-every",
            "500", "--save-dir", str(out), "--run-name",
            f"qhead_{tag}_s20260722", "--seed", "20260722", "--hf-repo",
            "jeremycochoy/gift-pretrain-full-4096", "--hf-path", "small_v1",
            *B4_ARCH, *(["--reconstruction", "encoder"] if recon else []),
            "--t-raw", "4096", "--n-channels", "1", "--d-model", "64",
            "--n-heads", "8", "--num-layers", "3", "--encoder-type", "gru",
            "--rev-norm-kind", "ewma", "--rev-norm-span", "128",
            "--freq-emb-dim", "3", "--seasonality-emb-dim", "3"]


@pytest.mark.parametrize("arch", [{}, {"CF_HEAD_ARCH": "transformer"}])
def test_the_flags_of_the_transformer_head_stay_as_they_were(stub_checkout,
                                                             arch):
    """Unset, or with its own name, the head is the B4 head: each flag in
    its place, in the forecast mode and in the reconstruction mode."""
    tmp_path, bb, _ = stub_checkout
    modes = {"arm_bb40k_h30k_student": {},
             "arm_bb40k_h30k_recon": {"CF_RECONSTRUCTION": "encoder"}}
    for tag, mode in modes.items():
        r = head_eval(stub_checkout, tag, CF_SKIP_EVAL="1", **mode, **arch)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        out = tmp_path / "root" / "eval" / tag
        assert recorded(out / "head_argv.json") == b4_argv(
            bb, out, tag, recon=bool(mode))


def test_the_linear_arch_reaches_the_head_and_the_score(stub_checkout):
    """CF_HEAD_ARCH=linear: one linear map in place of the transformer
    head, each other flag as it was, and the R score under the linear tag."""
    tmp_path, bb, _ = stub_checkout
    tag = "arm_bb40k_h30k_recon_lin"
    r = head_eval(stub_checkout, tag, **LINEAR)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = tmp_path / "root" / "eval" / tag
    want = b4_argv(bb, out, tag, recon=True)
    at = want.index("--head-arch")
    want[at:at + len(B4_ARCH)] = ["--head-arch", "linear"]
    assert recorded(out / "head_argv.json") == want
    assert (out / f"qhead_{tag}_s20260722_final.pth").exists()
    shard = recorded(out / "gift_r" / "shard_0" / "argv.json")
    assert shard[shard.index("--strategy") + 1] == "R"
    assert shard[shard.index("--head-path") + 1] == str(
        out / f"qhead_{tag}_s20260722_final.pth")
    assert (tmp_path / "res" / f"score_{tag}.txt").read_text().strip() == "0.5000"


@pytest.mark.parametrize("tag,arch", [
    ("arm_bb40k_h30k_recon", "linear"),            # the tag of the B4 head
    ("arm_bb40k_h30k_recon_lin", "transformer"),   # the tag of a linear head
    ("arm_bb40k_h30k_recon_lin", None),
])
def test_the_tag_of_a_head_names_its_arch(stub_checkout, tag, arch):
    """The two heads of one checkpoint must not share a head file or a
    score file: a linear head has a tag that ends in _recon_lin, and the
    transformer head a tag that ends in _recon."""
    tmp_path = stub_checkout[0]
    knobs = dict(CF_RECONSTRUCTION="encoder")
    if arch:
        knobs["CF_HEAD_ARCH"] = arch
    r = head_eval(stub_checkout, tag, **knobs)
    assert r.returncode != 0
    assert "_recon_lin" in r.stdout + r.stderr
    assert not (tmp_path / "root" / "eval" / tag / "head_argv.json").exists()


def test_a_linear_head_is_a_reconstruction_head_only(stub_checkout):
    """This runner has no tag rule for a linear forecast head."""
    r = head_eval(stub_checkout, "arm_bb40k_h30k_student",
                  CF_HEAD_ARCH="linear")
    assert r.returncode != 0
    assert "CF_RECONSTRUCTION=encoder" in r.stdout + r.stderr


def test_an_unknown_head_arch_is_refused(stub_checkout):
    r = head_eval(stub_checkout, "arm_bb40k_h30k_recon",
                  CF_RECONSTRUCTION="encoder", CF_HEAD_ARCH="gru")
    assert r.returncode != 0
    assert "CF_HEAD_ARCH" in r.stdout + r.stderr


def test_the_argv_mode_hands_the_linear_flags_to_a_shared_trainer(
        stub_checkout):
    tmp_path = stub_checkout[0]
    jobs = tmp_path / "jobs.jsonl"
    tag = "arm_bb40k_h30k_recon_lin"
    r = head_eval(stub_checkout, tag, CF_HEAD_ARGV_TO=str(jobs), **LINEAR)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    (argv,) = [json.loads(line) for line in jobs.read_text().splitlines()]
    assert argv[argv.index("--head-arch") + 1] == "linear"
    assert argv[argv.index("--reconstruction") + 1] == "encoder"
    assert not (tmp_path / "root" / "eval" / tag / "head_argv.json").exists()


# ---------------------------------------------------------------------------
# 4. The queue of the box: CF425_HEAD_ARCH=linear
# ---------------------------------------------------------------------------

LIN_GOOD = [tag + "_lin" for tag in base.GOOD]


def record_arch(tmp_path):
    """The stub runner of the queue also records the head arch it gets."""
    stub = tmp_path / "runner.sh"
    stub.write_text(base.QUEUE_STUB.replace(
        "$CF_RECONSTRUCTION", "$CF_RECONSTRUCTION ${CF_HEAD_ARCH:-unset}"))


def linear_queue(queue_box):
    """The box of ``queue_box`` for the linear queue: its own results
    folder, and the default folder of its heads under the checkpoint root."""
    tmp_path, _, env = queue_box
    res = tmp_path / "res_lin"
    res.mkdir()
    env = dict(env, CF425_HEAD_ARCH="linear", CF425_RES=str(res),
               CF425_TEST_RES=str(res))
    env.pop("CF425_ROOT")
    return tmp_path, res, env


def test_the_linear_queue_scores_each_job_under_its_linear_tag(queue_box):
    """Each job trains in a wave and gets one score, under the tag
    ..._recon_lin. The runner gets the linear arch for the flags of the
    head and for the score. The heads have their own folder."""
    tmp_path, res, env = linear_queue(queue_box)
    record_arch(tmp_path)
    r = run_queue(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name[6:-4] for p in res.glob("score_*.txt")) == LIN_GOOD
    trained = [tag for wave in waves(res) for tag in wave]
    assert sorted(t for t in trained if "bad" not in t) == LIN_GOOD
    assert {tuple(c[2:4]) for c in job_calls(res)} == {("encoder", "linear")}
    heads = tmp_path / "ckpt" / "cf-425-lin" / "recon" / "eval"
    assert sorted(p.name for p in heads.iterdir() if "bad" not in p.name) == LIN_GOOD
    assert not (tmp_path / "ckpt" / "cf-425").exists()
    assert "QUEUE_END: 4 scores" in r.stdout


def test_the_default_queue_still_trains_the_transformer_heads(queue_box):
    tmp_path, res, env = queue_box
    record_arch(tmp_path)
    r = run_queue(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name[6:-4] for p in res.glob("score_*.txt")) == base.GOOD
    assert {tuple(c[2:4]) for c in job_calls(res)} == {("encoder",
                                                        "transformer")}
    assert "QUEUE_END: 4 scores" in r.stdout


def test_an_unknown_arch_starts_no_queue(queue_box):
    _, res, env = queue_box
    r = run_queue(dict(env, CF425_HEAD_ARCH="gru"))
    assert r.returncode == 2 and "CF425_HEAD_ARCH" in r.stdout
    assert not (res / "calls.log").exists()


# A shared trainer that holds the GPU lock for a moment: it writes the time
# of its start and the time of its first step to one file for all queues.
SLOW_START_TRAINER = base.TRAINER_STUB.replace(
    'print("[shared] 1 steps: a stub", flush=True)',
    '''events = os.environ["CF425_TEST_EVENTS"]
began = time.time()
time.sleep(0.4)
with open(events, "a") as f:
    f.write(f"{began} {time.time()}\\n")
print("[shared] 1 steps: a stub", flush=True)''')


def test_the_two_queues_run_together_and_share_the_gpu_lock(queue_box):
    """The linear queue starts while the queue of the transformer heads
    runs: each has its own queue lock, job locks and folders. One GPU lock
    holds from the memory check of a wave to its first step, so no two
    waves of the two queues start in the same moment."""
    tmp_path, res, env = queue_box
    _, res_lin, env_lin = linear_queue(queue_box)
    (tmp_path / "trainer.py").write_text(SLOW_START_TRAINER)
    events = tmp_path / "events.log"
    queues = [subprocess.Popen(
        ["bash", str(SCRIPTS / "queue.sh")], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
        env=dict(e, CF425_TEST_EVENTS=str(events))) for e in (env, env_lin)]
    outs = [q.communicate(timeout=180)[0] for q in queues]
    assert [q.returncode for q in queues] == [0, 0], "\n".join(outs)
    assert not any("another queue holds" in out for out in outs)
    assert sorted(p.name[6:-4] for p in res.glob("score_*.txt")) == base.GOOD
    assert sorted(p.name[6:-4] for p in res_lin.glob("score_*.txt")) == LIN_GOOD
    starts = sorted(tuple(map(float, line.split())) for line in open(events))
    assert len(starts) >= 6          # the waves of 2 lanes of 2 queues
    assert all(end <= later for (_, end), (later, _)
               in zip(starts, starts[1:])), starts


def real_job_box(tmp_path):
    """The inputs of the real job table, as sparse files of the right size."""
    ck = tmp_path / "ckpt"
    for row in base.job_rows():
        (ck / row[4]).parent.mkdir(parents=True, exist_ok=True)
        with open(ck / row[4], "wb") as f:
            f.truncate(int(row[5]))
    with open(tmp_path / "seasonal_naive.csv", "wb") as f:
        f.truncate(24831)
    env = dict(os.environ, CF425_CK=str(ck), CF425_CODE=str(REPO_ROOT),
               CF425_SN_REF=str(tmp_path / "seasonal_naive.csv"),
               CF425_DRY_RUN="1")
    for key in ("CF425_JOBS", "CF425_ROOT", "CF425_WAVE_SIZE",
                "CF425_OLD_WAVE_SIZE", "CF425_HEAD_ARCH"):
        env.pop(key, None)
    return env


def plan_of(env, res):
    r = run_queue(dict(env, CF425_RES=str(res)))
    assert r.returncode == 0, r.stdout + r.stderr
    return [line.split() for line in r.stdout.splitlines()]


def test_the_plan_of_the_transformer_queue_is_unchanged(tmp_path):
    """The 56 jobs in the waves of round 2: the 9 old-data jobs in 1 wave,
    and the GiftEvalPretrain jobs in waves of 12, 12, 12 and 11."""
    plan = plan_of(real_job_box(tmp_path), tmp_path / "res")
    assert len(plan) == 56 and all(p[5].endswith("_recon") for p in plan)
    sizes = {}
    for stream, wave, *_ in plan:
        sizes[stream, wave] = sizes.get((stream, wave), 0) + 1
    assert sizes == {("old", "1"): 9, ("gift_pretrain", "1"): 12,
                     ("gift_pretrain", "2"): 12, ("gift_pretrain", "3"): 12,
                     ("gift_pretrain", "4"): 11}


def test_the_plan_of_the_linear_queue_holds_the_56_jobs(tmp_path):
    """The linear queue plans each job of the job table one time, under its
    linear tag, with tier 1 first in each lane."""
    env = dict(real_job_box(tmp_path), CF425_HEAD_ARCH="linear")
    plan = plan_of(env, tmp_path / "res_lin")
    tags = [p[5] for p in plan]
    assert len(set(tags)) == 56 and all(t.endswith("_recon_lin") for t in tags)
    for stream in ("old", "gift_pretrain"):
        lane = [p for p in plan if p[0] == stream]
        assert [int(p[1]) for p in lane] == sorted(int(p[1]) for p in lane)
        assert [p[2] for p in lane] == sorted(p[2] for p in lane)
    assert sum(p[0] == "old" for p in plan) == 9


# ---------------------------------------------------------------------------
# 5. The sync of the box: the folders of the linear queue
# ---------------------------------------------------------------------------

# A box with the environment of a real ssh session: no variable of elisa.
CLEAN_SSH = """#!/bin/bash
exec env -i PATH="$PATH" HOME="$HOME" bash -c "${@: -1}"
"""
# A box that keeps each command and runs none.
RECORDING_SSH = """#!/bin/bash
cat >/dev/null
printf '%s\\n' "${@: -1}" >>"$CF425_TEST_COMMANDS"
"""


def test_the_sync_hands_its_roots_to_the_prune_of_the_box(sync_box):
    """The prune runs on the box, with the environment of the box. So the
    sync gives it the folder of the heads and the folder of the scores: a
    sync of other folders must not prune the default ones."""
    tmp_path, box, mirror, env = sync_box
    ssh = tmp_path / "clean_ssh.sh"
    ssh.write_text(CLEAN_SSH)
    env = dict(env, CF425_SSH=f"bash {ssh}")
    env.pop("CF425_PRUNE_ROOT")
    r = subprocess.run(["bash", str(SCRIPTS / "sync_box.sh")],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    left = {str(p.relative_to(box)) for p in box.rglob("*") if p.is_file()}
    assert left == {"recon/eval/done_recon/q_losses.csv",
                    "recon/eval/done_recon/gift_r/summary.txt",
                    "recon/eval/pruned_recon/gift_r/all_results.csv",
                    "recon/eval/train_recon/q_best.pth",
                    "recon/eval/late_recon/q_final.pth"}
    assert (mirror / "recon/eval/done_recon/q_final.pth").read_bytes() == b"F" * 10


def test_the_linear_sync_reads_and_writes_its_own_folders(tmp_path):
    """CF425_HEAD_ARCH=linear: the sync reads the folders of the linear
    queue on the box, writes its own mirror folders on elisa, and gives the
    prune of the linear code the folders of the linear queue. No command
    names a folder of the queue of the transformer heads."""
    import re
    ssh = tmp_path / "ssh.sh"
    ssh.write_text(RECORDING_SSH)
    commands = tmp_path / "commands.log"
    env = dict(os.environ, HOME=str(tmp_path), CF425_HEAD_ARCH="linear",
               CF425_SSH=f"bash {ssh}", CF425_TEST_COMMANDS=str(commands))
    for key in ("CF425_BOX_ROOT", "CF425_RES", "CF425_MIRROR",
                "CF425_RESULTS_MIRROR", "CF425_PRUNE", "CF425_PRUNE_ROOT"):
        env.pop(key, None)
    r = subprocess.run(["bash", str(SCRIPTS / "sync_box.sh")],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    backup = tmp_path / "checkpoints_backup"
    assert (backup / "cf-412" / "vast_lr100x" / "cf-425-lin").is_dir()
    assert (backup / "cf-425-lin" / "box_results").is_dir()
    assert not (backup / "cf-425").exists()
    assert not (backup / "cf-412" / "vast_lr100x" / "cf-425").exists()
    sent = commands.read_text()
    assert "cd '/workspace/ckpt/cf-425-lin'" in sent
    assert "cd '/workspace/results/cf-425-lin'" in sent
    assert "CF425_PRUNE_ROOT='/workspace/ckpt/cf-425-lin'" in sent
    assert "CF425_RES='/workspace/results/cf-425-lin'" in sent
    assert "bash '/workspace/cf-425-lin/reports/" in sent
    assert not re.search(r"cf-425(?!-lin)", sent)


# ---------------------------------------------------------------------------
# 6. The measures of a wave: the report of the shared trainer, the probe and
#    the parity scripts of the box
# ---------------------------------------------------------------------------

def load_shared_trainer():
    import importlib.util
    sys.path.insert(0, str(base.GIFT_SCRIPTS))
    spec = importlib.util.spec_from_file_location("shared_425", base.SHARED_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_report_gives_the_share_of_time_that_the_stream_took(capsys):
    """A wave that waits for its stream is as fast as its stream, and more
    jobs in it cost no time. A wave that does not wait is as fast as its
    GPU. So the report gives the time that the wave waited for a batch."""
    shared = load_shared_trainer()
    shared.report(100, 12, 50.0, (40, 20.0), CPU, 5.0)
    line = capsys.readouterr().out.strip()
    assert line.startswith("[shared] 100 steps: 2.00 steps/s since the start, "
                           "2.00 steps/s over the last 40, 24.0 job steps/s, "
                           "the stream took 25% of that time")


def test_a_shared_run_reports_the_wait_for_its_stream(tmp_path):
    bb = saved(tmp_path, "bb.pth", ewma_model())
    r = train_shared(tmp_path, [linear_argv(tmp_path, bb, "a", *OLD_DATA)])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    reports = [line for line in r.stdout.splitlines()
               if line.startswith("[shared] ") and " steps: " in line]
    assert reports and reports[0].startswith("[shared] 1 steps: ")
    assert all("the stream took " in line and "% of that time" in line
               for line in reports)


def load_parity_compare():
    return base.load_script("parity_compare")


def test_the_parity_script_compares_the_tensors_of_two_heads(tmp_path):
    compare = load_parity_compare()
    torch.manual_seed(0)
    head = LinearQuantileForecastingHead(H=16, forecast_len=8)
    a = saved(tmp_path, "a.pth", head)
    b = saved(tmp_path, "b.pth", head)
    assert compare.compare_heads("OMB 25k", a, b) == (
        "OMB 25k: 2 tensors in the wave head, 2 in the solo head, "
        "2 identical")
    with torch.no_grad():
        head.forecast_head.bias.add_(1e-7)
    c = saved(tmp_path, "c.pth", head)
    assert compare.compare_heads("x", a, c).endswith("1 identical")


def test_the_parity_script_reads_heads_after_its_flag(tmp_path, capsys,
                                                      monkeypatch):
    compare = load_parity_compare()
    head = saved(tmp_path, "a.pth",
                 LinearQuantileForecastingHead(H=16, forecast_len=8))
    monkeypatch.setattr(sys, "argv", ["parity_compare.py", "--heads", "OMB",
                                      head, head])
    compare.main()
    assert capsys.readouterr().out.strip().endswith("2 identical")


FAKE_NVIDIA_SMI = """#!/bin/bash
# The whole GPU: "used, util". The compute apps: the python processes.
case "$*" in
  *query-compute-apps*) for p in $(pgrep -f trainer.py); do echo "$p, 1234"; done ;;
  *memory.free*) echo 30000 ;;
  *) echo "5000, 50" ;;
esac
"""


@pytest.mark.parametrize("arch,suffix", [(None, "_recon"),
                                         ("linear", "_recon_lin")])
def test_the_probe_trains_wave_1_of_each_lane_and_scores_nothing(
        queue_box, arch, suffix):
    """probe.sh: the first wave of each lane, as the queue plans it, for a
    few steps. It gives the report of each wave and the GPU memory of each
    wave process."""
    tmp_path, _, env = queue_box
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text(FAKE_NVIDIA_SMI)
    (bin_dir / "nvidia-smi").chmod(0o755)
    res = tmp_path / "probe_res"
    env = dict(env, CF425_PROBE_RES=str(res),
               CF425_PROBE_ROOT=str(tmp_path / "probe_root"),
               CF425_TEST_RES=str(res / "queue"), CF425_TRIES="1",
               CF425_TEST_TRAIN_SLEEP="1.5", CF425_PROBE_SAMPLE="0.5",
               PATH=f"{bin_dir}:{os.environ['PATH']}")
    if arch:
        env["CF425_HEAD_ARCH"] = arch
    r = subprocess.run(["bash", str(SCRIPTS / "probe.sh"), "7"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "probe: 4 jobs, 7 steps" in r.stdout and "queue rc=0" in r.stdout
    trained = sorted(tag for wave in waves(res / "queue") for tag in wave)
    assert trained == sorted(f"{job}{suffix}" for job in (
        "arm_a_bb10k_h30k", "arm_bad_bb40k_h30k", "arm_c_bb100k_h30k",
        "arm_d_bb50k_h30k"))
    calls = job_calls(res / "queue")
    assert calls and all(c[1] == "argv" and c[3] == "7" for c in calls)
    assert r.stdout.count("peak 1234 MiB") == 2      # one line for each wave
    assert "5000 MiB" in r.stdout                    # the whole GPU
