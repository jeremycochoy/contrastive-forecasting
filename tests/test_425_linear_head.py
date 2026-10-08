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
    eval_args, head_eval, held, job_calls, load_eval_module, meanstd_model,
    oracle_latents, queue_box, recorded, run_queue, saved, shifted,
    stub_checkout, sync_box, train_shared, train_solo, walk, waves,
    waves_train)

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


def test_a_probe_reads_a_step_rate_more_often(stub_checkout):
    """HEAD_LOG_EVERY: the steps between two log lines of the head trainer.
    Unset, 500 (the golden flags above)."""
    tmp_path = stub_checkout[0]
    tag = "arm_bb40k_h30k_recon_lin"
    r = head_eval(stub_checkout, tag, CF_SKIP_EVAL="1", HEAD_LOG_EVERY="50",
                  **LINEAR)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    head = recorded(tmp_path / "root" / "eval" / tag / "head_argv.json")
    assert head[head.index("--log-every") + 1] == "50"


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
               CF425_TEST_RES=str(res), CF425_SCORE_VRAM_MIB="0")
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


# A shared trainer that holds the GPU lock: it writes "start" to one file for
# all queues, it waits for the shell command CF425_TEST_START_UNTIL, and it
# writes "step" at its first step. The order of the lines is the order of
# the events.
SLOW_START_TRAINER = base.TRAINER_STUB.replace(
    'print("[shared] 1 steps: a stub", flush=True)',
    '''def event(name):
    with open(os.environ["CF425_TEST_EVENTS"], "a") as f:
        f.write(name + "\\n")
event("start")
wait(os.environ["CF425_TEST_START_UNTIL"])
event("step")
print("[shared] 1 steps: a stub", flush=True)''')


def test_the_two_queues_run_together_and_share_the_gpu_lock(queue_box):
    """The linear queue starts while the queue of the transformer heads
    runs: each has its own queue lock, job locks and folders. One GPU lock
    holds from the memory check of a wave to its first step, so no two
    waves of the two queues start in the same moment. Each wave holds the
    lock for 0.4 s: with no lock, a second wave starts in that time. With
    the lock, the load of the machine cannot change the order of the lines."""
    tmp_path, res, env = queue_box
    _, res_lin, env_lin = linear_queue(queue_box)
    (tmp_path / "trainer.py").write_text(SLOW_START_TRAINER)
    events = tmp_path / "events.log"
    queues = [subprocess.Popen(
        ["bash", str(SCRIPTS / "queue.sh")], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
        env=dict(e, CF425_TEST_EVENTS=str(events),
                 CF425_TEST_START_UNTIL="sleep 0.4")) for e in (env, env_lin)]
    outs = [q.communicate(timeout=180)[0] for q in queues]
    assert [q.returncode for q in queues] == [0, 0], "\n".join(outs)
    assert not any("another queue holds" in out for out in outs)
    assert sorted(p.name[6:-4] for p in res.glob("score_*.txt")) == base.GOOD
    assert sorted(p.name[6:-4] for p in res_lin.glob("score_*.txt")) == LIN_GOOD
    order = events.read_text().split()
    assert len(order) >= 12          # the waves of 2 lanes of 2 queues
    assert order == ["start", "step"] * (len(order) // 2), order


FREE_NVIDIA_SMI = """#!/bin/bash
# The free memory of the GPU comes from a file of the test. Each call adds
# one line to a second file.
echo >>"$CF425_TEST_FREE.calls"
cat "$CF425_TEST_FREE"
"""


def wait_for(event, timeout=60):
    """Wait until ``event()`` is true. Only a fault comes to the timeout."""
    deadline = time.time() + timeout
    while not event() and time.time() < deadline:
        time.sleep(0.05)


def gated_queue(queue_box, free):
    """The linear queue of ``queue_box`` on a GPU with ``free`` MiB free: a
    wave needs 6,000 MiB, a score 3,000 MiB, and the box has 2 eval slots."""
    tmp_path, res, env = linear_queue(queue_box)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text(FREE_NVIDIA_SMI)
    (bin_dir / "nvidia-smi").chmod(0o755)
    (tmp_path / "free").write_text(f"{free}\n")
    slots = tmp_path / "slots"
    slots.mkdir()
    env = dict(env, PATH=f"{bin_dir}:{os.environ['PATH']}",
               CF425_TEST_FREE=str(tmp_path / "free"),
               CF425_WAVE_VRAM_MIB="6000", CF425_SCORE_VRAM_MIB="3000",
               CF425_EVAL_SLOTS="2", CF425_EVAL_SLOTDIR=str(slots),
               CF425_VRAM_POLL="1", CF425_WAVE_SIZE="3")
    return tmp_path, res, slots, env


def test_a_wave_leaves_room_for_the_scores_that_can_start(queue_box):
    """A score has no memory gate, and the queue of the transformer heads
    scores on the same GPU. So a wave of the linear queue starts only when
    the GPU has its memory and the memory of a score for each eval slot
    with no score. Here 10,000 MiB are free: less than 6,000 and 2 times
    3,000. The wave reads the free memory 3 times and does not start. It
    starts when 12,000 MiB are free."""
    tmp_path, res, _, env = gated_queue(queue_box, 10000)
    queue = subprocess.Popen(["bash", str(SCRIPTS / "queue.sh")], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
    asked = tmp_path / "free.calls"
    wait_for(lambda: asked.exists() and asked.stat().st_size >= 3)
    assert queue.poll() is None and not (res / "calls.log").exists() or not \
        waves(res)
    (tmp_path / "free").write_text("12000\n")
    out = queue.communicate(timeout=120)[0]
    assert queue.returncode == 0, out
    assert "10000 MiB free on the GPU" in out and "6000 stay free" in out
    assert sorted(p.name[6:-4] for p in res.glob("score_*.txt")) == LIN_GOOD


def test_a_score_that_runs_needs_no_room(queue_box):
    """An eval slot with a score holds its memory now, so the free memory
    counts it: with 1 of 2 slots in use, a wave needs 6,000 and 3,000."""
    tmp_path, res, slots, env = gated_queue(queue_box, 9000)
    with held(slots / "slot_0"):
        r = run_queue(dict(env, CF425_SCORE="0"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "stay free" not in r.stdout          # no wave waited
    assert len(waves(res)) == 3


def test_the_queue_gives_its_eval_slots_to_each_score(queue_box):
    """The scores of the two queues count in one set of eval slots."""
    tmp_path, res, env = linear_queue(queue_box)
    (tmp_path / "runner.sh").write_text(base.QUEUE_STUB.replace(
        "$CF_RECONSTRUCTION", "$CF393_EVAL_SLOTDIR $CF393_EVAL_SLOTS"))
    r = run_queue(dict(env, CF425_EVAL_SLOTDIR="/tmp/some_slots",
                       CF425_EVAL_SLOTS="3"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert {tuple(c[2:4]) for c in job_calls(res)} == {("/tmp/some_slots", "3")}
    r = run_queue(dict(env, CF425_DRY_RUN="1"))
    assert r.returncode == 0


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


def stream_waves(stream, size):
    """``{(stream, wave): jobs}`` for the jobs of one stream of the real job
    table, in waves of ``size`` jobs."""
    count = sum(row[6] == stream for row in base.job_rows())
    return {(stream, str(wave + 1)): min(size, count - wave * size)
            for wave in range(-(-count // size))}


def test_the_plan_of_the_transformer_queue_is_unchanged(tmp_path):
    """Each job of the job table in the waves of a queue with no head and no
    score: the old-data jobs in waves of 10, and the GiftEvalPretrain jobs
    in waves of 12."""
    plan = plan_of(real_job_box(tmp_path), tmp_path / "res")
    assert len(plan) == len(base.job_rows())
    assert all(p[5].endswith("_recon") for p in plan)
    sizes = {}
    for stream, wave, *_ in plan:
        sizes[stream, wave] = sizes.get((stream, wave), 0) + 1
    assert sizes == {**stream_waves("old", 10),
                     **stream_waves("gift_pretrain", 12)}


def test_the_plan_of_the_linear_queue_holds_each_job(tmp_path):
    """The linear queue plans each job of the job table one time, under its
    linear tag, with tier 1 first in each lane."""
    env = dict(real_job_box(tmp_path), CF425_HEAD_ARCH="linear")
    plan = plan_of(env, tmp_path / "res_lin")
    tags = [p[5] for p in plan]
    assert len(set(tags)) == len(base.job_rows())
    assert all(t.endswith("_recon_lin") for t in tags)
    for stream in ("old", "gift_pretrain"):
        lane = [p for p in plan if p[0] == stream]
        assert [int(p[1]) for p in lane] == sorted(int(p[1]) for p in lane)
        assert [p[2] for p in lane] == sorted(p[2] for p in lane)
    assert sum(p[0] == "old" for p in plan) == sum(
        row[6] == "old" for row in base.job_rows())


def score_each_job_but(res, suffix, run="ABC"):
    """A score file for each job of the real job table but those of one run:
    the state of a queue that ended before the card got that run."""
    res.mkdir(parents=True, exist_ok=True)
    for code, arm, stop_k, *_ in base.job_rows():
        if code != run:
            (res / f"score_{arm}_bb{stop_k}k_h30k_{suffix}.txt").write_text(
                "0.5000\n")


def abc_wave(plan, suffix):
    """True when a plan is one old-data wave of the jobs of ABC and no other
    job, with the first and the last stop of the run first."""
    first, *between, last = base.CARD["ABC"]
    stops = [f"{stop}k" for stop in (first, last, *between)]
    return ([(p[0], p[1], p[3], p[4]) for p in plan]
            == [("old", "1", "ABC", stop) for stop in stops]
            and all(p[5].endswith(f"lr10x_bb{p[4]}_h30k_{suffix}")
                    for p in plan))


@pytest.mark.parametrize("arch, suffix", [("transformer", "recon"),
                                          ("linear", "recon_lin")])
def test_a_new_start_trains_the_abc_jobs_in_one_wave(tmp_path, arch, suffix):
    """The card got ABC after the two queues ended. Each other job has its
    score. So a new start of a queue trains the 10 jobs of ABC and no other
    job, in one wave."""
    res = tmp_path / "res"
    score_each_job_but(res, suffix)
    plan = plan_of(dict(real_job_box(tmp_path), CF425_HEAD_ARCH=arch), res)
    assert len(plan) == 10 and abc_wave(plan, suffix)


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
    queue on the box, and gives the prune of the linear code these folders.
    On elisa it writes its own folders beside those of the linear queue of
    elisa (queue_elisa.sh), and never the job tree of that queue. No command
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
    assert (backup / "cf-425-lin" / "box_ckpt").is_dir()
    assert (backup / "cf-425-lin" / "box_results").is_dir()
    assert not (backup / "cf-425-lin" / "ckpt").exists()
    assert not (backup / "cf-425").exists() and not (backup / "cf-412").exists()
    sent = commands.read_text()
    assert "cd '/workspace/ckpt/cf-425-lin'" in sent
    assert "cd '/workspace/results/cf-425-lin'" in sent
    assert "CF425_PRUNE_ROOT='/workspace/ckpt/cf-425-lin'" in sent
    assert "CF425_RES='/workspace/results/cf-425-lin'" in sent
    assert "bash '/workspace/cf-425-lin/reports/" in sent
    assert not re.search(r"cf-425(?!-lin)", sent)


def test_the_sync_of_the_box_fallback_keeps_the_files_of_the_queue_of_elisa(
        tmp_path):
    """The queue of elisa and its fallback on the box can run the same job.
    The sync of the fallback writes its own job tree on elisa (``box_ckpt``).
    So the head, the per-config table and the log that the queue of elisa
    wrote for that job stay. The prune reads the manifest of that tree: it
    frees the box only of a head that the sync brought. Two jobs: the head
    of elisa is older than the head of the box, and newer."""
    tags = [f"cf412om_bb{stop}k_h30k_recon_lin" for stop in (10, 25)]
    older, newer = (f"recon/eval/{tag}" for tag in tags)
    lin = tmp_path / "checkpoints_backup" / "cf-425-lin"
    box, res = tmp_path / "box" / "ckpt", tmp_path / "box" / "results"
    # One job on the two machines: a head of one size, and other text files.
    elisa = {"q_final.pth": b"E" * 10, "gift_r/all_results.csv": b"elisa\n",
             "stop.log": b"elisa\n"}
    fallback = {"q_final.pth": b"B" * 10, "stop.log": b"the box\n",
                "gift_r/all_results.csv": b"the box\n"}
    for job in (older, newer):
        for root, files in ((lin / "ckpt", elisa), (box, fallback)):
            for name, data in files.items():
                (root / job / name).parent.mkdir(parents=True, exist_ok=True)
                (root / job / name).write_bytes(data)
    os.utime(lin / "ckpt" / older / "q_final.pth", (1_000_000_000,) * 2)
    os.utime(box / newer / "q_final.pth", (1_000_000_000,) * 2)
    res.mkdir(parents=True)
    for tag in tags:
        (res / f"score_{tag}.txt").write_text("0.9000\n")
    ssh = tmp_path / "fake_ssh.sh"
    ssh.write_text(base.FAKE_SSH)
    env = dict(os.environ, HOME=str(tmp_path), CF425_HEAD_ARCH="linear",
               CF425_SSH=f"bash {ssh}", CF425_BOX_ROOT=str(box),
               CF425_RES=str(res), CF425_PRUNE=str(SCRIPTS / "prune.sh"))
    for key in ("CF425_MIRROR", "CF425_RESULTS_MIRROR", "CF425_PRUNE_ROOT"):
        env.pop(key, None)
    r = subprocess.run(["bash", str(SCRIPTS / "sync_box.sh")],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    for job in (older, newer):
        for name, data in elisa.items():
            assert (lin / "ckpt" / job / name).read_bytes() == data, name
        for name, data in fallback.items():
            assert (lin / "box_ckpt" / job / name).read_bytes() == data, name
        assert not (box / job / "q_final.pth").exists()   # elisa holds its copy
    for tag in tags:
        assert (lin / "box_results" / f"score_{tag}.txt"
                ).read_text() == "0.9000\n"


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
# Each GPU: "index, used, util". GPU 0 has 9366 MiB at first and 16511 MiB
# while a wave trains. The compute apps: the python processes.
case "$*" in
  *query-compute-apps*) for p in $(pgrep -f trainer.py); do echo "$p, 1234"; done ;;
  *memory.free*) echo 30000 ;;
  *) if pgrep -f trainer.py >/dev/null; then echo "0, 16511, 50"; else echo "0, 9366, 50"; fi
     echo "1, 77, 0" ;;
esac
"""

# A shell test for CF425_TEST_TRAIN_UNTIL, before the path of the sample log
# of a probe: the log holds a line of 2 waves, and a line of GPU 0 while a
# wave trains. So a wave trains until the probe has a sample of each wave.
SAMPLED = ("""awk '$2 == "wave" && !seen[$3]++ { n++ } / total 0 16511 / """
           """{ busy = 1 } END { exit !(busy && n >= 2) }' """)


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
               CF425_TEST_TRAIN_UNTIL=SAMPLED + str(res / "gpu.log"),
               CF425_PROBE_SAMPLE="0.5",
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
    assert r.stdout.count("% of one CPU core") == 2
    assert "all processes, GPU 0: 16511 MiB" in r.stdout    # each whole GPU
    assert "all processes, GPU 1: 77 MiB" in r.stdout


# ---------------------------------------------------------------------------
# 7. The lanes of a queue: several GPUs, a stop file, a limit of CPU threads
# ---------------------------------------------------------------------------

# A shared trainer that records the GPU and the CPU threads of each wave.
LANE_TRAINER = base.TRAINER_STUB.replace(
    'f.write("wave " + " ".join(names) + "\\n")',
    'f.write("wave " + " ".join(names) + "\\n")\n'
    '    f.write("on " + os.environ.get("CUDA_VISIBLE_DEVICES", "unset") + " "\n'
    '            + os.environ.get("OMP_NUM_THREADS", "unset") + " "\n'
    '            + os.environ.get("MKL_NUM_THREADS", "unset") + " "\n'
    '            + " ".join(names) + "\\n")')
# A runner that records the GPU and the eval slots of each score.
LANE_RUNNER = base.QUEUE_STUB.replace(
    'echo "$tag score $CF_RECONSTRUCTION"',
    'echo "$tag score $BB_GPU $CF393_EVAL_SLOTDIR"')


def lane_box(queue_box, lanes, **knobs):
    """The linear queue of ``queue_box`` with the lanes ``lanes``."""
    tmp_path, res, env = linear_queue(queue_box)
    (tmp_path / "trainer.py").write_text(LANE_TRAINER)
    (tmp_path / "runner.sh").write_text(LANE_RUNNER)
    env = dict(env, CF425_LANES=lanes, CF425_EVAL_SLOTDIR=str(tmp_path / "slots"),
               **knobs)
    env.pop("OMP_NUM_THREADS", None)
    env.pop("MKL_NUM_THREADS", None)
    return tmp_path, res, env


def wave_gpus(res):
    """``{job tag: GPU of the wave that trained it}``, from the last try."""
    return {tag: c[1] for c in base.calls(res) if c[0] == "on" for tag in c[4:]}


def score_calls(res):
    """``{job tag: (GPU, eval slot folder)}`` of each score call."""
    return {c[0]: (c[2], c[3]) for c in job_calls(res) if c[1] == "score"}


def is_old(tag):
    return tag.startswith(("arm_c", "arm_d"))


def test_each_lane_trains_and_scores_on_its_gpu(queue_box):
    """CF425_LANES gives each lane its streams and its GPU. A wave trains
    on the GPU of its lane, and each job gets its score on the GPU that
    trained it. A lane on another GPU than the GPU of the queue counts its
    scores in its own eval slots."""
    tmp_path, res, env = lane_box(queue_box, "gift_pretrain:0 old:1")
    r = run_queue(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name[6:-4] for p in res.glob("score_*.txt")) == LIN_GOOD
    trained, scored = wave_gpus(res), score_calls(res)
    assert sorted(scored) == LIN_GOOD
    slots = str(tmp_path / "slots")
    for tag in LIN_GOOD:
        gpu = "1" if is_old(tag) else "0"
        assert trained[tag] == gpu
        assert scored[tag] == (gpu, slots + "_gpu1" if is_old(tag) else slots)


def test_two_lanes_of_one_stream_train_two_waves_at_one_time(queue_box):
    """Two lanes can take their waves from one stream: each wave has its
    own jobs, and the two waves train at the same time. No job of the
    old-data stream fails. A job that fails in one lane can get one try too
    many from the other lane: the queue reads the tries of a job before it
    locks the job."""
    _, res, env = lane_box(queue_box, "old:0 old:1", CF425_OLD_WAVE_SIZE="1")
    r = run_queue(dict(env, CF425_TEST_TRAIN_UNTIL=waves_train(res)))
    assert r.returncode == 0, r.stdout + r.stderr
    order = [c[0] for c in base.calls(res) if c[0] in ("wave", "wave_end")]
    assert order[:3] == ["wave", "wave", "wave_end"]
    old = waves(res)
    assert all(len(wave) == 1 for wave in old)
    assert sorted(wave[0] for wave in old) == LIN_GOOD[2:]
    assert {wave_gpus(res)[tag] for tag in LIN_GOOD[2:]} == {"0", "1"}
    assert sorted(p.name[6:-4] for p in res.glob("score_*.txt")) == LIN_GOOD[2:]


def test_a_lane_takes_its_next_stream_when_the_first_has_no_job(queue_box):
    """A lane with the streams old,gift_pretrain trains the old-data jobs,
    then the GiftEvalPretrain jobs. So its GPU does not stay idle."""
    _, res, env = lane_box(queue_box, "old,gift_pretrain:1",
                           CF425_WAVE_SIZE="3")
    r = run_queue(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name[6:-4] for p in res.glob("score_*.txt")) == LIN_GOOD
    first, later = waves(res)[0], waves(res)[1:]
    assert all(is_old(tag) for tag in first)
    assert later and not any(is_old(tag) for wave in later for tag in wave)
    assert set(wave_gpus(res).values()) == {"1"}


@pytest.mark.parametrize("lanes", ["gift_pretrain", "gift_pretrain:x",
                                   "moon:0", "old:0 gift:1"])
def test_a_wrong_lane_starts_no_queue(queue_box, lanes):
    _, res, env = lane_box(queue_box, lanes)
    r = run_queue(env)
    assert r.returncode == 2 and "CF425_LANES" in r.stdout
    assert not (res / "calls.log").exists()


def test_the_trainer_of_a_wave_gets_a_limit_of_cpu_threads(queue_box):
    """CF425_TRAIN_THREADS: the CPU threads of each trainer process, so a
    queue does not take each core of a machine that it shares. Unset, the
    queue sets no limit, as before."""
    _, res, env = lane_box(queue_box, "gift_pretrain:0 old:0",
                           CF425_TRAIN_THREADS="3")
    r = run_queue(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert {tuple(c[2:4]) for c in base.calls(res) if c[0] == "on"} == {("3", "3")}


def test_a_queue_with_no_thread_limit_sets_none(queue_box):
    _, res, env = lane_box(queue_box, "gift_pretrain:0 old:0")
    r = run_queue(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert {tuple(c[2:4]) for c in base.calls(res) if c[0] == "on"} == {
        ("unset", "unset")}


def test_two_lanes_on_two_gpus_start_their_waves_at_one_time(queue_box):
    """The start lock is for one GPU: it counts the free memory of that
    GPU. So lanes on two GPUs do not wait for each other. Each wave holds
    its start lock until two waves have their start."""
    tmp_path, res, env = lane_box(queue_box, "gift_pretrain:0 old:1",
                                  CF425_WAVE_SIZE="3")
    (tmp_path / "trainer.py").write_text(SLOW_START_TRAINER)
    events = tmp_path / "events.log"
    r = run_queue(dict(
        env, CF425_TEST_EVENTS=str(events),
        CF425_TEST_START_UNTIL=f"[ $(grep -c start {events}) -ge 2 ]"))
    assert r.returncode == 0, r.stdout + r.stderr
    # The second wave starts before the first step of the first.
    assert events.read_text().split()[:2] == ["start", "start"]
    assert (tmp_path / "gpu.lock_gpu1").exists()


def run_until_wave(env, res, stop, lanes=1, script="queue.sh"):
    """Start the queue, write the stop file ``stop`` when each of its
    ``lanes`` lanes trains a wave, and return the output of the queue. Each
    wave trains until the stop file exists, so the file comes in a wave."""
    queue = subprocess.Popen(
        ["bash", str(SCRIPTS / script)], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
        env=dict(env, CF425_TEST_TRAIN_UNTIL=f"[ -e {stop} ]"))
    wait_for(lambda: (res / "calls.log").exists()
             and len(waves(res)) >= lanes)
    stop.touch()
    out = queue.communicate(timeout=120)[0]
    assert queue.returncode == 0, out
    return out


def test_a_stop_file_gives_a_gpu_back_at_the_end_of_a_wave(queue_box):
    """stop_gpu<N> in the results folder: each lane of that GPU trains its
    wave to the end, then starts no score and no wave. The lanes of the
    other GPU go on. A new queue, with no stop file, scores the head that
    the lane left and trains the jobs that it did not start."""
    _, res, env = lane_box(queue_box, "gift_pretrain:0 old:1",
                           CF425_WAVE_SIZE="1", CF425_LANE_STAGGER="0")
    out = run_until_wave(env, res, res / "stop_gpu0", lanes=2)
    gift = [wave for wave in waves(res) if not is_old(wave[0])]
    assert gift == [[LIN_GOOD[0]]]                 # one wave, to its end
    scored = sorted(p.name[6:-4] for p in res.glob("score_*.txt"))
    assert scored == LIN_GOOD[2:]                  # the old-data jobs only
    assert "stop file" in out
    heads = res.parent / "ckpt" / "cf-425-lin" / "recon" / "eval"
    assert list((heads / LIN_GOOD[0]).glob("*_final.pth"))
    assert not (res / "failed").exists() or not list((res / "failed").iterdir())

    (res / "stop_gpu0").unlink()
    again = run_queue(dict(env, CF425_TRIES="1"))
    assert again.returncode == 0, again.stdout + again.stderr
    assert sorted(p.name[6:-4] for p in res.glob("score_*.txt")) == LIN_GOOD
    trained = [tag for wave in waves(res) for tag in wave]
    assert trained.count(LIN_GOOD[0]) == 1         # scored, not trained again
    assert not any(is_old(tag) for wave in waves(res)[3:] for tag in wave)


def test_the_stop_file_of_the_queue_ends_each_lane(queue_box):
    _, res, env = lane_box(queue_box, "gift_pretrain:0 old:1",
                           CF425_WAVE_SIZE="1", CF425_OLD_WAVE_SIZE="1",
                           CF425_LANE_STAGGER="0")
    run_until_wave(env, res, res / "stop", lanes=2)
    assert len(waves(res)) == 2 and not list(res.glob("score_*.txt"))


def test_a_wave_that_fails_under_a_stop_file_counts_no_try(queue_box):
    """An owner who needs the GPU now writes the stop file and stops the
    trainer of the lane. The jobs of that wave keep all their tries."""
    _, res, env = lane_box(queue_box, "gift_pretrain:0", CF425_WAVE_SIZE="3")
    out = run_until_wave(env, res, res / "stop_gpu0")
    assert len(waves(res)) == 1 and base.BAD + "_lin" in waves(res)[0]
    assert not (res / "failed").exists() or not list((res / "failed").iterdir())
    assert "counts no try" in out


def test_the_probe_can_take_more_waves_of_each_stream(queue_box):
    """CF425_PROBE_WAVES=2: the first two waves of each stream, for a probe
    of a queue with two lanes on one stream."""
    tmp_path, _, env = queue_box
    res = tmp_path / "probe_res"
    env = dict(env, CF425_PROBE_RES=str(res), CF425_HEAD_ARCH="linear",
               CF425_PROBE_ROOT=str(tmp_path / "probe_root"),
               CF425_TEST_RES=str(res / "queue"), CF425_TRIES="1",
               CF425_PROBE_SAMPLE="0.5", CF425_PROBE_WAVES="2",
               CF425_LANES="gift_pretrain:0 gift_pretrain:0 old:0")
    r = subprocess.run(["bash", str(SCRIPTS / "probe.sh"), "7"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "probe: 5 jobs, 7 steps" in r.stdout
    assert len(waves(res / "queue")) == 3


# ---------------------------------------------------------------------------
# 8. The linear queue on elisa: queue_elisa.sh and deploy_elisa.sh
# ---------------------------------------------------------------------------

ALWAYS_FREE_NVIDIA_SMI = "#!/bin/bash\necho 24000\n"


def elisa_home(tmp_path, rows=None):
    """A home folder as on elisa: the input checkpoints of ``rows`` (the
    real job table when None) as sparse files, the seasonal-naive reference,
    and a GPU tool that reports free memory."""
    home = tmp_path / "home"
    ck = home / "checkpoints_backup" / "cf-412" / "vast_lr100x"
    for row in base.job_rows() if rows is None else rows:
        (ck / row[4]).parent.mkdir(parents=True, exist_ok=True)
        with open(ck / row[4], "wb") as f:
            f.truncate(int(row[5]))
    ref = home / "workspaces" / "gift-eval" / "results" / "seasonal_naive"
    ref.mkdir(parents=True)
    with open(ref / "all_results.csv", "wb") as f:
        f.truncate(24831)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text(ALWAYS_FREE_NVIDIA_SMI)
    (bin_dir / "nvidia-smi").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CF425_")
           and k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "GIFT_EVAL")}
    # The queue waits while a forecast_scores.sh runs on the machine, and a
    # test of an other pytest worker starts one. So this queue looks for a
    # process that does not exist.
    env.update(HOME=str(home), CF425_CODE=str(REPO_ROOT),
               CF425_AFTER=str(tmp_path / "forecast_scores.sh"),
               PATH=f"{bin_dir}:{os.environ['PATH']}")
    return home, env


def run_elisa(env, *args, timeout=180):
    return subprocess.run(["bash", str(SCRIPTS / "queue_elisa.sh"), *args],
                          capture_output=True, text=True, env=env,
                          timeout=timeout)


def test_the_elisa_queue_plans_each_job_under_its_home_folder(tmp_path):
    """queue_elisa.sh: the linear queue with the folders of elisa. Each
    folder is under ~/checkpoints_backup/cf-425-lin, so a restart of elisa
    keeps the heads, the scores and the logs."""
    home, env = elisa_home(tmp_path)
    r = run_elisa(dict(env, CF425_DRY_RUN="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    plan = [line.split() for line in r.stdout.splitlines()]
    assert len({p[5] for p in plan}) == len(base.job_rows())
    assert all(p[5].endswith("_recon_lin") for p in plan)
    lin = home / "checkpoints_backup" / "cf-425-lin"
    assert (lin / "results").is_dir() and (lin / "ckpt" / "recon").is_dir()
    gift = [int(p[1]) for p in plan if p[0] == "gift_pretrain"]
    assert max(gift.count(wave) for wave in set(gift)) <= 8


def test_a_new_start_of_the_elisa_queue_trains_the_abc_jobs_in_one_wave(
        tmp_path):
    """The folder of the linear queue of elisa holds the score of each job
    but those of ABC. A new start trains the 10 jobs of ABC in one wave."""
    home, env = elisa_home(tmp_path)
    score_each_job_but(home / "checkpoints_backup" / "cf-425-lin" / "results",
                       "recon_lin")
    r = run_elisa(dict(env, CF425_DRY_RUN="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    plan = [line.split() for line in r.stdout.splitlines()]
    assert len(plan) == 10 and abc_wave(plan, "recon_lin")


def test_the_elisa_queue_uses_two_gpus_and_keeps_no_state_in_tmp(tmp_path,
                                                                 queue_box):
    """The lanes of elisa train on GPU 0 and GPU 1, each trainer has its
    limit of CPU threads, each job gets its score on the GPU of its wave,
    and the locks and the eval slots are in the results folder."""
    box, _, _ = queue_box
    rows = [row for row in base.job_rows(box / "jobs.tsv")]
    home, env = elisa_home(tmp_path / "e", rows)
    (box / "trainer.py").write_text(LANE_TRAINER)
    (box / "runner.sh").write_text(LANE_RUNNER)
    lin = home / "checkpoints_backup" / "cf-425-lin"
    res = lin / "results"
    env = dict(env, CF425_JOBS=str(box / "jobs.tsv"),
               CF425_RUNNER=str(box / "runner.sh"),
               CF425_TRAINER=str(box / "trainer.py"),
               CF425_TEST_RES=str(res), CF425_LANE_STAGGER="0",
               CF425_GPU_POLL="0.1", CF425_WAVE_SIZE="1")
    r = run_elisa(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name[6:-4] for p in res.glob("score_*.txt")) == LIN_GOOD
    trained, scored = wave_gpus(res), score_calls(res)
    assert set(trained.values()) == {"0", "1"}
    for tag in LIN_GOOD:
        gpu, slots = scored[tag]
        assert gpu == trained[tag] and slots.startswith(str(res))
    threads = {c[2] for c in base.calls(res) if c[0] == "on"}
    assert len(threads) == 1 and 1 <= int(threads.pop()) <= 8
    assert list((lin / "ckpt" / "recon" / "eval").iterdir())
    assert (res / "locks" / "gpu_start.lock").exists()


def test_the_elisa_queue_goes_on_after_a_stop(tmp_path, queue_box):
    """The same command after a restart of elisa or a stop: a job with a
    score is not trained and not scored again, a job with a head gets its
    score, and the other jobs train."""
    box, _, _ = queue_box
    rows = [row for row in base.job_rows(box / "jobs.tsv")
            if "bad" not in row[1]]
    (box / "good.tsv").write_text("".join("\t".join(row) + "\n" for row in rows))
    home, env = elisa_home(tmp_path / "e", rows)
    (box / "trainer.py").write_text(LANE_TRAINER)
    (box / "runner.sh").write_text(LANE_RUNNER)
    res = home / "checkpoints_backup" / "cf-425-lin" / "results"
    env = dict(env, CF425_JOBS=str(box / "good.tsv"),
               CF425_RUNNER=str(box / "runner.sh"),
               CF425_TRAINER=str(box / "trainer.py"),
               CF425_TEST_RES=str(res), CF425_LANE_STAGGER="0",
               CF425_GPU_POLL="0.1", CF425_WAVE_SIZE="1",
               CF425_OLD_WAVE_SIZE="1")
    res.mkdir(parents=True)
    (res / "stop").touch()                      # a stop before the first wave
    r = run_elisa(env)
    assert r.returncode == 0 and not (res / "calls.log").exists()
    (res / "stop").unlink()
    run_until_wave(env, res, res / "stop", script="queue_elisa.sh")
    done = {p.name[6:-4] for p in res.glob("score_*.txt")}
    first = [tag for wave in waves(res) for tag in wave]
    assert 0 < len(first) < 4 and not done      # heads, and no score yet
    (res / "stop").unlink()
    r = run_elisa(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name[6:-4] for p in res.glob("score_*.txt")) == LIN_GOOD
    trained = [tag for wave in waves(res) for tag in wave]
    assert sorted(trained) == LIN_GOOD          # each job trains one time
    calls = job_calls(res)
    again = run_elisa(env)
    assert again.returncode == 0
    assert [tag for wave in waves(res) for tag in wave] == trained
    assert job_calls(res) == calls              # no call for a scored job
    assert len([c for c in calls if c[1] == "score"]) == 4


def deploy(env, *args):
    return subprocess.run(["bash", str(SCRIPTS / "deploy_elisa.sh"), *args],
                          capture_output=True, text=True, env=env, timeout=120)


def test_the_deploy_puts_the_committed_code_in_the_elisa_folder(tmp_path):
    """deploy_elisa.sh: the code of the last commit, the name of that
    commit and the Hugging Face token, in the code folder of the linear
    queue of elisa. A restart of elisa keeps that folder, and its path
    shows cf-425 in each process of the queue."""
    token = tmp_path / "token.txt"
    token.write_text("hf_test\n")
    env = dict(os.environ, HOME=str(tmp_path), CF425_HF_TOKEN_FILE=str(token))
    env.pop("CF425_ELISA_BASE", None)
    r = deploy(env)
    assert r.returncode == 0, r.stdout + r.stderr
    code = tmp_path / "checkpoints_backup" / "cf-425-lin" / "code"
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse",
                           "--short=8", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    assert (code / "DEPLOYED_COMMIT").read_text().strip() == head
    assert (code / "experiments" / "hf_token.txt").read_text() == "hf_test\n"
    for rel in ("reports/2026-10-06_encoder_reconstruction/scripts/queue.sh",
                "reports/2026-08-08_rollout_depth/scripts/head_eval_bb.sh",
                "reports/2026-08-08_rollout_depth/results/config_costs.csv",
                "experiments/2026-04-13_gift-eval/scripts/"
                "train_forecasting_heads_shared.py",
                "src/forecasting_head.py"):
        assert (code / rel).is_file(), rel
    assert not (code / "tests").exists()
    (code / "stale.txt").write_text("an old file")
    assert deploy(env).returncode == 0          # a new deploy replaces it
    assert not (code / "stale.txt").exists()


def test_the_deploy_changes_no_code_under_a_queue_that_runs(tmp_path):
    """bash reads a script while it runs it. So the deploy refuses when a
    queue holds the queue lock of the results folder."""
    token = tmp_path / "token.txt"
    token.write_text("hf_test\n")
    env = dict(os.environ, HOME=str(tmp_path), CF425_HF_TOKEN_FILE=str(token))
    assert deploy(env).returncode == 0
    lin = tmp_path / "checkpoints_backup" / "cf-425-lin"
    (lin / "code" / "mark.txt").write_text("the code of the queue")
    (lin / "results").mkdir()
    with held(lin / "results" / "queue.lock"):
        r = deploy(env)
    assert r.returncode != 0 and "queue" in r.stdout + r.stderr
    assert (lin / "code" / "mark.txt").exists()


# ---------------------------------------------------------------------------
# 9. The report scripts: the linear scores in the tables and in the figures
# ---------------------------------------------------------------------------

# The job tree of each source of the linear scores, in the folder of the
# linear queue: the queue of elisa, and its fallback on the box.
JOB_TREE = {"results": "ckpt", "box_results": "box_ckpt"}


def per_config_of(source):
    """A per-config table of one config, with the name of its source."""
    return f"dataset,eval_metrics/MASE[0.5]\n{source},1.0\n"


def linear_tree(lin, tag, score, folder="results", wave="gift_x"):
    """The files of one scored linear job, as queue_elisa.sh leaves them
    (``results`` and ``ckpt``) or as the sync of the box fallback does
    (``box_results`` and ``box_ckpt``). The per-config table, the stop log
    and the head of a job hold the name of their source: one config with
    that name."""
    res = lin / folder
    (res / "waves" / wave).mkdir(parents=True, exist_ok=True)
    (res / f"score_{tag}.txt").write_text(score)
    (res / "queue.log").write_text(f"queue of {folder}\n")
    (res / "waves" / wave / "train.log").write_text("[shared] done\n")
    job = lin / JOB_TREE[folder] / "recon" / "eval" / tag
    (job / "gift_r").mkdir(parents=True)
    (job / "gift_r" / "all_results.csv").write_text(per_config_of(folder))
    (job / "gift_r" / "summary.txt").write_text("summary\n")
    (job / "stop.log").write_text(f"stop of {folder}\n")
    (job / f"qhead_{tag}_s20260722_final.pth").write_bytes(folder.encode())
    (job / f"qhead_{tag}_s20260722_losses.csv").write_text("step,loss\n1,0.5\n")


def collect_with(tmp_path):
    collect = base.load_script("collect")
    collect.BOX_RESULTS = tmp_path / "box"
    collect.MIRROR = tmp_path / "mirror" / "cf-425"
    collect.LINEAR = tmp_path / "lin"
    collect.ELISA_SNAPSHOTS = tmp_path / "no_snap"
    collect.RESULTS = tmp_path / "results"
    collect.SYNC_LOG = tmp_path / "no_sync.log"
    collect.BOX_RESULTS.mkdir()
    return collect, tmp_path / "results"


def test_collect_gives_the_linear_scores_their_own_table(tmp_path):
    """The linear scores come from the folder of the linear queue: the
    queue of elisa, and the mirror of its box fallback. They get their own
    table, and the table of the transformer heads does not hold them."""
    import gzip
    collect, results = collect_with(tmp_path)
    (collect.BOX_RESULTS / "score_cf412om_bb10k_h30k_recon.txt").write_text(
        "0.3100\n")
    linear_tree(tmp_path / "lin", "cf412om_bb10k_h30k_recon_lin", "0.7100\n")
    linear_tree(tmp_path / "lin", "cf412om_bb25k_h30k_recon_lin", "0.6500\n",
                folder="box_results", wave="gift_y")
    (tmp_path / "lin" / "results" / "score_cf412om_bb50k_h30k_recon_lin.txt"
     ).write_text("")                                         # no score yet
    collect.main()
    assert (results / "recon_trajectories.tsv").read_text() == (
        "cf412om\t10\t0.3100\n")
    assert (results / "recon_lin_trajectories.tsv").read_text() == (
        "cf412om\t10\t0.7100\ncf412om\t25\t0.6500\n")
    for stop in (10, 25):
        tag = f"cf412om_bb{stop}k_h30k_recon_lin"
        assert (results / "scores" / f"score_{tag}.txt").is_file()
        assert (results / "per_config" / f"{tag}.csv").is_file()
        assert (results / "logs" / "jobs" / tag / "summary.txt").is_file()
        assert (results / "logs" / "jobs" / tag / "stop.log").is_file()
        packed = results / "head_losses" / f"{tag}_losses.csv.gz"
        assert gzip.decompress(packed.read_bytes()) == b"step,loss\n1,0.5\n"
    logs = results / "logs" / "linear"
    assert (logs / "results" / "queue.log").read_text() == "queue of results\n"
    assert (logs / "box_results" / "queue.log").is_file()
    assert (logs / "results" / "waves" / "gift_x" / "train.log").is_file()
    assert (logs / "box_results" / "waves" / "gift_y" / "train.log").is_file()
    assert not list(results.rglob("*.pth"))


def test_collect_with_no_linear_score_writes_no_linear_table(tmp_path):
    collect, results = collect_with(tmp_path)
    (collect.BOX_RESULTS / "score_cf412om_bb10k_h30k_recon.txt").write_text(
        "0.3100\n")
    collect.main()
    assert (results / "recon_trajectories.tsv").is_file()
    assert not (results / "recon_lin_trajectories.tsv").exists()
    assert not (results / "logs" / "linear").exists()


def test_collect_with_no_linear_score_removes_the_linear_table_of_an_earlier_run(
        tmp_path):
    """The linear table holds the linear scores that exist now. With no
    linear score, the table of an earlier run goes away. So the table of the
    jobs and the figures show no linear score that no score file holds."""
    collect, results = collect_with(tmp_path)
    tag = "cf412om_bb10k_h30k_recon_lin"
    linear_tree(tmp_path / "lin", tag, "0.7100\n")

    def linear_column():
        rows = (results / "job_scores.tsv").read_text().splitlines()[1:]
        return {row.split("\t")[4] for row in rows}

    collect.main()
    assert (results / "recon_lin_trajectories.tsv").read_text() == (
        "cf412om\t10\t0.7100\n")
    assert linear_column() == {"", "0.7100"}
    (tmp_path / "lin" / "results" / f"score_{tag}.txt").unlink()
    collect.main()
    assert not (results / "recon_lin_trajectories.tsv").exists()
    assert linear_column() == {""}


def test_collect_reads_the_job_tree_of_each_linear_source(tmp_path):
    """The queue of elisa and the box fallback keep the files of their jobs
    in two trees, ``ckpt`` and ``box_ckpt``. A job with a score from the two
    machines keeps the score and the files of elisa. A job with a score from
    the fallback only gets the files of the fallback."""
    collect, results = collect_with(tmp_path)
    both, box_only = (f"cf412om_bb{stop}k_h30k_recon_lin" for stop in (10, 25))
    linear_tree(tmp_path / "lin", both, "0.7100\n")
    linear_tree(tmp_path / "lin", both, "0.9000\n", folder="box_results")
    linear_tree(tmp_path / "lin", box_only, "0.6500\n", folder="box_results")
    collect.main()
    assert (results / "recon_lin_trajectories.tsv").read_text() == (
        "cf412om\t10\t0.7100\ncf412om\t25\t0.6500\n")
    for tag, score, source in ((both, "0.7100\n", "results"),
                               (box_only, "0.6500\n", "box_results")):
        assert (results / "scores" / f"score_{tag}.txt").read_text() == score
        assert (results / "per_config" / f"{tag}.csv").read_text() == (
            per_config_of(source))
        assert (results / "logs" / "jobs" / tag / "stop.log").read_text() == (
            f"stop of {source}\n")
        assert (results / "head_losses" / f"{tag}_losses.csv.gz").is_file()


def test_collect_writes_one_table_of_the_scores_of_each_job(tmp_path):
    """``job_scores.tsv``: one row for each job of jobs.tsv, in its order.
    The columns: the code of the run, the stop, the B4 forecast, the R of the
    transformer head and the R of the linear head. A score that does not
    exist yet is an empty cell. The B4 forecast is the score of #412, or the
    score of this card. For a stop with the two, the score of this card
    wins, as in the figures."""
    collect, results = collect_with(tmp_path)
    collect.JOBS = tmp_path / "jobs.tsv"
    collect.JOBS.write_text(
        "# a comment\n"
        "#code\tarm\tstop_k\ttier\tckpt\tbytes\tdata\n"
        "OMB\tcf412om\t10\t1\ta.pth\t1\tgift_pretrain\n"
        "OMB\tcf412om\t25\t2\tb.pth\t1\tgift_pretrain\n"
        "LOW\tk3_x_lr30x\t665\t1\tc.pth\t1\told\n"
        "OCB\tcf412oc\t40\t1\td.pth\t1\tgift_pretrain\n")
    collect.FORECAST_412 = tmp_path / "gm_trajectories.tsv"
    collect.FORECAST_412.write_text("cf412om\t10\t1.3782\ncf412om\t25\t1.3345\n"
                                    "cf412om\t50\t1.4497\n"
                                    "k3_x_lr30x\t665\t1.2550\n")
    box = collect.BOX_RESULTS
    (box / "score_cf412om_bb10k_h30k_recon.txt").write_text("0.3100\n")
    (box / "score_k3_x_lr30x_bb665k_h30k_recon.txt").write_text("0.0451\n")
    (box / "score_k3_x_lr30x_bb665k_h30k_student.txt").write_text("1.1913\n")
    linear_tree(tmp_path / "lin", "cf412om_bb10k_h30k_recon_lin", "0.7100\n")
    linear_tree(tmp_path / "lin", "cf412om_bb25k_h30k_recon_lin", "0.6500\n",
                folder="box_results")
    collect.main()
    assert (results / "job_scores.tsv").read_text() == (
        "run\tstop_k\tb4_forecast\tr_transformer_head\tr_linear_head\n"
        "OMB\t10\t1.3782\t0.3100\t0.7100\n"
        "OMB\t25\t1.3345\t\t0.6500\n"
        "LOW\t665\t1.1913\t0.0451\t\n"
        "OCB\t40\t\t\t\n")


def test_the_table_of_the_jobs_holds_each_job_of_the_card(tmp_path):
    """The real job table and the real scores of #412: one row for each job,
    in the order of the job table. #412 gives the B4 forecast of each job
    but the 5 stops that the card names: this card scores them
    (``forecast_425.tsv``)."""
    collect = base.load_script("collect")
    collect.RESULTS = tmp_path                      # no table of this card
    rows = collect.job_scores()
    assert len(rows) == len(base.job_rows())
    assert [row[:2] for row in rows] == [[job[0], job[2]]
                                         for job in base.job_rows()]
    assert {(row[0], row[1]) for row in rows if not row[2]} == {
        ("LOW", "665"), ("LNG", "665"), ("MIN", "1080"), ("CYN", "1140"),
        ("TWN", "420")}
    assert all(row[3:] == ["", ""] for row in rows)
    (tmp_path / "forecast_425.tsv").write_text(
        "k3_r100_09_lr56_fix09_dec10k_lr30x\t665\t1.1913\n")
    low, = [row for row in collect.job_scores() if row[0] == "LOW"]
    assert low == ["LOW", "665", "1.1913", "", ""]


def linear_figure(tmp_path, overlay, linear, name="f"):
    plot = base.load_script("plot_recon")
    forecast = {"cf412om": {40000: 1.3782, 100000: 1.3345}}
    fig = plot.draw_figure(plot.GRAPHS["ours_patch_sizes"], forecast, linear,
                           tmp_path / f"{name}_{overlay}.png", overlay,
                           title="R, linear head: ours", linear=True)
    return plot, fig


@pytest.mark.parametrize("overlay", [False, True])
def test_the_linear_head_has_its_own_figures(tmp_path, overlay):
    """No figure mixes the two heads. A linear figure shows the R of the
    linear heads only, with the colour and the line style of each run, and
    the legend gives the first and the last R of that head with their
    ratio. The first line of the key names the linear map. The title and
    the plot hold no fact."""
    pytest.importorskip("matplotlib")
    plot, fig = linear_figure(tmp_path, overlay,
                              {"cf412om": {40000: 0.7100, 100000: 0.6500}})
    ax = fig.axes[-1]                                   # the panel of R
    hue = plot.colour("cf412om")
    curve, = [line for line in ax.get_lines() if line.get_color() == hue]
    assert list(curve.get_ydata()) == [0.71, 0.65]
    assert curve.get_linestyle() == "-"                 # the style of the run
    texts = base.legend_texts(fig)
    assert any(text.endswith("R 0.7100 → 0.6500, ×0.92") for text in texts)
    assert "R: one linear map decodes the horizon from latents" in texts
    assert all(len(axis.texts) == 0 for axis in fig.axes)
    assert fig.axes[0].get_title() == "R, linear head: ours"
    if overlay:
        b4, = [line for line in fig.axes[0].get_lines()
               if line.get_color() == hue]
        assert list(b4.get_ydata()) == [1.3782, 1.3345]


def test_a_linear_figure_holds_no_snapshot_marker(tmp_path):
    """Only a transformer head has a scored earlier snapshot, so a linear
    figure draws none, with no key line. With no linear score, there is no
    linear figure."""
    pytest.importorskip("matplotlib")
    plot = base.load_script("plot_recon")
    points = {"cf412om": {40000: 0.7100, 100000: 0.6500}}
    snapshots = {"cf412om": {40000: (28500, 0.6000)}}
    fig = plot.draw_figure(plot.GRAPHS["ours_patch_sizes"], {}, points,
                           tmp_path / "s.png", False, snapshots=snapshots,
                           linear=True)
    ax, = fig.axes
    assert not [line for line in ax.get_lines()
                if line.get_markerfacecolor() == "white"]
    assert ("The same head at an earlier head step"
            not in base.legend_texts(fig))
    assert plot.draw_figure(plot.GRAPHS["ours_patch_sizes"], {}, {},
                            tmp_path / "none.png", False, linear=True) is None
    assert not (tmp_path / "none.png").exists()


def test_the_two_versions_of_a_graph_share_one_y_range(tmp_path):
    """The y range of a figure also holds the scores of the other head of
    the same graph (y_extra), so a reader compares the two heads at the
    same height. graph_values gives the scores of one head in one graph,
    with the snapshot scores that its charts hold."""
    pytest.importorskip("matplotlib")
    plot = base.load_script("plot_recon")
    points = {"cf412om": {40000: 0.3100, 100000: 0.2900}}
    fig = plot.draw_figure(plot.GRAPHS["ours_patch_sizes"], {}, points,
                           tmp_path / "f.png", False, y_extra=[0.71, 0.29])
    ax, = fig.axes
    low, high = ax.get_ylim()
    assert low < 0.29 and high > 0.71
    assert plot.graph_values(
        plot.GRAPHS["ours_patch_sizes"], points,
        {"cf412om": {40000: (28500, 0.2)}}) == [0.31, 0.29, 0.2]


def test_a_floor_near_a_linear_score_is_on_the_chart(tmp_path):
    """A linear score is an R score of its scaling setup: the chart of the
    linear head holds the floor of the setup when a linear score is near
    it."""
    pytest.importorskip("matplotlib")
    plot = base.load_script("plot_recon")
    floors = [{"setup": "meanstd", "label": "mean/std floor",
               "arms": {"cf412om"}, "score": 1.5721}]
    linear = {"cf412om": {40000: 1.20, 100000: 1.10}}
    fig = plot.draw_figure(plot.GRAPHS["ours_patch_sizes"], {}, linear,
                           tmp_path / "f.png", False, floors, linear=True)
    ax, = fig.axes
    flat = [line for line in ax.get_lines() if len(set(line.get_ydata())) == 1]
    assert [float(line.get_ydata()[0]) for line in flat] == [1.5721]


def test_the_floors_table_names_each_arm_of_the_job_table():
    """A figure reads the floor of a run in ``results/floors.tsv``, by the
    arm of the run. So each arm of the job table is in the row of one
    scaling setup, and an old-data arm is in the row of the old data."""
    plot = base.load_script("plot_recon")
    floors = plot.load_floors(plot.FLOORS)
    arms = [arm for floor in floors for arm in sorted(floor["arms"])]
    assert sorted(arms) == sorted({row[1] for row in base.job_rows()})
    old, = [floor["arms"] for floor in floors if floor["setup"] == "ewma_old"]
    assert old == {row[1] for row in base.job_rows() if row[6] == "old"}


def test_the_figures_read_the_linear_table(tmp_path):
    plot = base.load_script("plot_recon")
    assert plot.LINEAR.name == "recon_lin_trajectories.tsv"
    table = tmp_path / "lin.tsv"
    table.write_text("cf412om\t10\t0.7100\n")
    assert plot.load([table]) == {"cf412om": {40000: 0.71}}    # batch 256: x4


def lin_check_tree(tmp_path, monkeypatch):
    """check_scores.py for the linear jobs: the artefacts of one job."""
    import gzip
    monkeypatch.setenv("CF425_HEAD_ARCH", "linear")
    monkeypatch.setenv("CF425_LINEAR", str(tmp_path / "lin"))
    check = base.load_script("check_scores")
    tag = "cf412om_bb10k_h30k_recon_lin"
    results, mirror = tmp_path / "results", tmp_path / "lin_ckpt"
    logs = results / "logs" / "jobs" / tag
    head = mirror / "recon" / "eval" / tag
    for folder in (results / "scores", results / "per_config", logs,
                   results / "head_losses", head):
        folder.mkdir(parents=True)
    jobs = tmp_path / "jobs.tsv"
    jobs.write_text("#code\tarm\tstop_k\nOMB\tcf412om\t10\t1\tx.pth\t1\tgift\n")
    (results / "scores" / f"score_{tag}.txt").write_text("0.5000\n")
    configs = [f"data_{i}/H/short" for i in range(97)]
    (results / "per_config" / f"{tag}.csv").write_text(
        f"dataset,{check.MASE}\n" + "".join(f"{c},1.0\n" for c in configs))
    (logs / "summary.txt").write_text(
        "Config      MASE  SN_MASE   Relative\n"
        + "".join(f"{c}    1.0000   2.0000     0.5000\n" for c in configs))
    (logs / "stop.log").write_text(
        "[10-07] [x] eval start (97 configs, R, forecast-len 16, cuda)\n")
    with gzip.open(results / "head_losses" / f"{tag}_losses.csv.gz",
                   "wt") as out:
        out.write("step,loss\n1,0.5\n30000,0.1\n")
    (head / "q_final.pth").write_bytes(b"head")
    check.RESULTS, check.JOBS, check.MIRROR = results, jobs, mirror
    return check, results


def test_the_check_reads_the_linear_jobs_under_their_tag(tmp_path,
                                                         monkeypatch):
    """CF425_HEAD_ARCH=linear: the check of each linear job, in its own
    table. The table of the transformer heads stays."""
    check, results = lin_check_tree(tmp_path, monkeypatch)
    assert check.main() == 0
    row, = csv.DictReader(open(results / "checks_lin.tsv"), delimiter="\t")
    assert row["result"] == "ok" and row["score"] == "0.5000"
    assert not (results / "checks.tsv").exists()


def test_the_default_check_reads_the_folder_of_the_linear_heads_of_elisa(
        monkeypatch):
    monkeypatch.setenv("CF425_HEAD_ARCH", "linear")
    monkeypatch.delenv("CF425_MIRROR", raising=False)
    monkeypatch.delenv("CF425_LINEAR", raising=False)
    check = base.load_script("check_scores")
    assert str(check.MIRROR).endswith("checkpoints_backup/cf-425-lin/ckpt")
    assert str(check.ELISA_SCORES).endswith("cf-425-lin/results")
    assert str(check.FALLBACK_SCORES).endswith("cf-425-lin/box_results")
    assert str(check.FALLBACK_HEADS).endswith("cf-425-lin/box_ckpt")
    monkeypatch.delenv("CF425_HEAD_ARCH")
    check = base.load_script("check_scores")
    assert str(check.MIRROR).endswith("cf-412/vast_lr100x/cf-425")


def test_the_check_reads_the_head_of_a_job_in_the_tree_of_its_score(
        tmp_path, monkeypatch):
    """collect.py takes the score of a linear job from the queue of elisa,
    or from the box fallback when elisa has none. The check reads the head
    of the job in the tree of that source. Three jobs: a score from the two
    machines, a score from the fallback only, and a score from the fallback
    whose head is not on elisa, beside a head of elisa with no score."""
    lin = tmp_path / "lin"
    both, box_only, lost = (f"cf412om_bb{stop}k_h30k_recon_lin"
                            for stop in (10, 25, 50))
    linear_tree(lin, both, "0.7100\n")
    linear_tree(lin, both, "0.9000\n", folder="box_results")
    linear_tree(lin, box_only, "0.6500\n", folder="box_results")
    linear_tree(lin, lost, "")                       # elisa: a head, no score
    (lin / "box_results" / f"score_{lost}.txt").write_text("0.8000\n")
    collect, results = collect_with(tmp_path)
    collect.main()
    monkeypatch.setenv("CF425_HEAD_ARCH", "linear")
    monkeypatch.setenv("CF425_LINEAR", str(lin))
    monkeypatch.delenv("CF425_MIRROR", raising=False)
    check = base.load_script("check_scores")
    check.RESULTS, check.JOBS = results, tmp_path / "jobs.tsv"
    check.JOBS.write_text("".join(
        f"OMB\tcf412om\t{stop}\t1\tx.pth\t1\tgift_pretrain\n"
        for stop in (10, 25, 50)))
    check.main()
    rows = csv.DictReader(open(results / "checks_lin.tsv"), delimiter="\t")
    assert {row["stop_k"]: (row["score"], row["head_bytes"]) for row in rows} == {
        "10": ("0.7100", str(len("results"))),
        "25": ("0.6500", str(len("box_results"))),
        "50": ("0.8000", "0")}
