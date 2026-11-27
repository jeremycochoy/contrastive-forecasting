"""#482: the head follower tries a failed head or score again.

``follow_multi.sh`` (reports/2026-10-10_frequency_family_head/scripts) waits
for the checkpoint of each stop, trains the control heads of the stop, then
scores each run with ``run_wave.sh``. On 11-27 the head runner refused an
empty token at the 25k stop of five boxes, and the follower went on to the
next stop: these stops got no score. Now the heads of a stop get up to
FF_HEAD_TRIES tries, each score too, FF_HEAD_RETRY_WAIT s apart. A last pass
gives each stop with no score one more try.

Stubs replace the head runner, the head trainer, the collect script and
nvidia-smi: no GPU and no network. The score step is the ``run_wave.sh`` of
the repository. Each stub writes its calls to ``results/calls.log``, and a
FF_TEST_* knob makes the first calls of a stub fail. FF_TEST_SCRIPTS=<folder>
runs the tests on the scripts of a code folder (the folder of follow_multi.sh).
"""
import fcntl
import os
import re
import subprocess
import time
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(os.environ.get(
    "FF_TEST_SCRIPTS", REPO_ROOT / "reports" / "2026-10-10_frequency_family_head" / "scripts"))
FOLLOW = SCRIPTS / "follow_multi.sh"
DEPLOYED = "a416b3ae"  # the follower of the 15 boxes of #482

# head_eval_bb.sh of the tests. With CF_HEAD_ARGV_TO: the flags of a head job.
# Else: the score of a head. The first FF_TEST_ARGV_FAILS (or
# FF_TEST_SCORE_FAILS) calls of the kind fail.
RUNNER = r'''#!/bin/bash
tag="$1"; out="$CF373_ROOT/eval/$tag"; mkdir -p "$out"
kind=score; [ -z "${CF_HEAD_ARGV_TO:-}" ] || kind=argv
n=$(( $(cat "$CF_RESULTS/calls.log" 2>/dev/null | grep -c "^$kind ") + 1 ))
echo "$kind $tag" >>"$CF_RESULTS/calls.log"
if [ "$kind" = argv ]; then
  [ "$n" -gt "${FF_TEST_ARGV_FAILS:-0}" ] || { echo "ABORT: empty HF_TOKEN" >&2; exit 2; }
  printf '["--save-dir", "%s", "--run-name", "qhead_%s"]\n' "$out" "$tag" >>"$CF_HEAD_ARGV_TO"
  exit 0
fi
[ "$n" -gt "${FF_TEST_SCORE_FAILS:-0}" ] || { echo "the eval stops" >&2; exit 1; }
echo 0.9500 >"$CF_RESULTS/score_$tag.txt"
'''

# The head trainer of the tests. The first FF_TEST_TRAIN_FAILS calls fail and
# write no head, or with FF_TEST_TRAIN_FIRST_ONLY the head of the first job
# only.
TRAINER = r'''
import json, os, sys
jobs = [json.loads(line) for line in open(sys.argv[sys.argv.index("--jobs") + 1])]
log = os.path.join(os.environ["FF_TEST_RES"], "calls.log")
n = 1 + (sum(line.startswith("train ") for line in open(log)) if os.path.exists(log) else 0)
names = [a[a.index("--run-name") + 1][len("qhead_"):] for a in jobs]
with open(log, "a") as f:
    f.write(" ".join(["train", os.environ.get("HF_TOKEN") or "no-token", *names]) + "\n")
fails = n <= int(os.environ.get("FF_TEST_TRAIN_FAILS", "0"))
for i, (a, name) in enumerate(zip(jobs, names)):
    if not fails or (i == 0 and os.environ.get("FF_TEST_TRAIN_FIRST_ONLY")):
        out = a[a.index("--save-dir") + 1]
        open(os.path.join(out, f"qhead_{name}_s20260722_final.pth"), "w").write("head")
print("[shared] 1 steps: a stub", flush=True)
if fails:
    print("RuntimeError: CUDA error: an illegal memory access", flush=True)
    sys.exit(1)
'''

COLLECT = r'''
import os
with open(os.path.join(os.environ["FF_TEST_RES"], "calls.log"), "a") as f:
    f.write("collect\n")
'''

# nvidia-smi of the tests: the free MiB of the GPU. The first FF_TEST_SMI_LOW
# calls give 100 MiB.
SMI = r'''#!/bin/bash
f="$FF_TEST_RES/smi.count"; n=$(( $(cat "$f" 2>/dev/null || echo 0) + 1 )); echo "$n" >"$f"
if [ "$n" -le "${FF_TEST_SMI_LOW:-0}" ]; then echo 100; else echo 24000; fi
'''


def write(path, text, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


def make_box(root):
    """The base folder of a follower, its code folder, the stubs and the
    checkpoint folder of the runs. Returns (root, environment)."""
    base, stubs = root / "base", root / "stubs"
    write(base / "code" / "DEPLOYED_COMMIT", "abcdef12\n")
    write(base / "code" / "experiments" / "hf_token.txt", "hf_test\n")
    write(stubs / "runner.sh", RUNNER, 0o755)
    write(stubs / "trainer.py", TRAINER)
    write(stubs / "collect.py", COLLECT)
    write(stubs / "bin" / "nvidia-smi", SMI, 0o755)
    write(root / "seasonal_naive.csv", "x" * 24831)
    (root / "gift").mkdir()
    (base / "results").mkdir(parents=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("FF_", "CF_", "EVAL_"))}
    env.update(PATH=f"{stubs / 'bin'}:{os.environ['PATH']}",
               FF_BASE=str(base), FF_CODE=str(base / "code"),
               FF_RUNNER=str(stubs / "runner.sh"),
               FF_TRAINER=str(stubs / "trainer.py"),
               FF_COLLECT=str(stubs / "collect.py"),
               FF_SN_REF=str(root / "seasonal_naive.csv"),
               GIFT_EVAL=str(root / "gift"),
               FF_WAVE=str(SCRIPTS / "run_wave.sh"), FF_GPU="0",
               FF_HEAD_RETRY_WAIT="0", FF_VRAM_WAIT="0", FF_POLL="1",
               FF_WAIT_MAX="20", FF_STREAM_WAIT="5",
               FF_TEST_RES=str(base / "results"), CUDA_VISIBLE_DEVICES="")
    return root, env


@pytest.fixture
def box(tmp_path):
    return make_box(tmp_path)


def save_ckpt(box, run, stop):
    """The checkpoint of a run at a stop, and its optimizer file."""
    folder = box[0] / "runs" / run
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"step": stop}, folder / f"{run}_{stop}k.pth")
    torch.save({}, folder / f"{run}_{stop}k_optimizer.pth")


def tag(run="rowa", stop=10):
    return f"{run}_bb{stop}k_h30k_control"


def make_head(box, run, stop):
    """The final head of a run at a stop, as a trainer leaves it."""
    write(box[0] / "base" / "heads" / "eval" / tag(run, stop)
          / f"qhead_{tag(run, stop)}_s20260722_final.pth", "head")


def follow_cmd(box, runs, stops, script, extra):
    root, env = box
    pairs = " ".join(f"{run}={root / 'runs' / run}" for run in runs)
    return ["bash", str(script)], dict(env, FF_MULTI_RUNS=pairs, FF_STOPS=stops,
                                       **extra)


def follow(box, runs, stops, script=FOLLOW, **extra):
    """One start of the follower. Its log is the stdout."""
    cmd, env = follow_cmd(box, runs, stops, script, extra)
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          timeout=600)


def start_follow(box, runs, stops, out, **extra):
    """A start of the follower in the background. Its log goes to ``out``."""
    cmd, env = follow_cmd(box, runs, stops, FOLLOW, extra)
    with open(out, "w") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                env=env)


def wait_for(path, text, timeout=120):
    end = time.time() + timeout
    while text not in Path(path).read_text():
        assert time.time() < end, f"no {text!r} in {Path(path).read_text()}"
        time.sleep(0.1)


def calls(box):
    path = box[0] / "base" / "results" / "calls.log"
    return [line.split() for line in open(path)] if path.exists() else []


def kinds(box):
    return [c[0] for c in calls(box)]


def scores(box):
    res = box[0] / "base" / "results"
    return {p.name[len("score_"):-len(".txt")]: p.read_text().strip()
            for p in res.glob("score_*.txt")}


def heads(box):
    return sorted(p.parent.name for p in
                  (box[0] / "base" / "heads" / "eval").glob("*/*_final.pth"))


def files(box):
    """The files of the base folder, with the time in a wave folder name out."""
    base = box[0] / "base"
    return sorted(re.sub(r"(multi_bb\d+k)_\d{4}_\d{6}_\w{4}", r"\1", str(p.relative_to(base)))
                  for p in base.rglob("*") if p.is_file())


def lines_with(log, text):
    return [line for line in log.splitlines() if text in line]


def deployed_follower(tmp_path):
    """follow_multi.sh of the boxes (a416b3ae), with its code_folder.sh."""
    where = "reports/2026-10-10_frequency_family_head/scripts"
    for name in ("follow_multi.sh", "code_folder.sh"):
        r = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{DEPLOYED}:{where}/{name}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"git holds no {DEPLOYED}")
        write(tmp_path / "deployed" / name, r.stdout)
    return tmp_path / "deployed" / "follow_multi.sh"


# A first try that works: the files of today.

def test_a_first_try_that_works_gives_the_calls_and_files_of_the_deployed_follower(
        tmp_path):
    deployed = deployed_follower(tmp_path)
    seen = {}
    for name, script in (("deployed", deployed), ("new", FOLLOW)):
        b = make_box(tmp_path / name)
        save_ckpt(b, "rowa", 10)
        r = follow(b, ["rowa"], "10", script=script)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        seen[name] = (calls(b), files(b), scores(b))
    assert seen["new"] == seen["deployed"]
    assert seen["new"][0] == [["argv", tag()], ["train", "hf_test", tag()],
                              ["score", tag()], ["collect"]]
    assert seen["new"][2] == {tag(): "0.9500"}


def test_a_stop_with_its_head_trains_no_head_and_gets_its_score(box):
    save_ckpt(box, "rowa", 10)
    make_head(box, "rowa", 10)
    r = follow(box, ["rowa"], "10")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert kinds(box) == ["score", "collect"]
    assert "train SKIP at 10k" in r.stdout
    assert scores(box) == {tag(): "0.9500"}


# The heads of a stop: up to FF_HEAD_TRIES tries.

def test_a_trainer_that_fails_once_gives_one_head_and_one_score(box):
    save_ckpt(box, "rowa", 10)
    r = follow(box, ["rowa"], "10", FF_TEST_TRAIN_FAILS="1")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert kinds(box) == ["argv", "train", "argv", "train", "score", "collect"]
    assert heads(box) == [tag()] and scores(box) == {tag(): "0.9500"}
    (failed,) = lines_with(r.stdout, "try 1 of 3 failed")
    assert "the heads of 10k: try 1 of 3 failed: the trainer ended with rc=1, no final head for rowa" in failed
    assert "RuntimeError: CUDA error" in failed     # the last line of train.log
    assert "The next try in 0 s." in failed
    assert "the heads of 10k: try 2 of 3 worked" in r.stdout
    assert "FAILED" not in r.stdout
    assert r.stdout.splitlines()[-1].endswith("FOLLOW_END rc=0")


def test_a_trainer_that_always_fails_gives_FAILED_after_the_tries(box):
    save_ckpt(box, "rowa", 10)
    r = follow(box, ["rowa"], "10", FF_HEAD_TRIES="2", FF_TEST_TRAIN_FAILS="99")
    assert r.returncode == 1
    # Two tries, then one more at the last pass. No score.
    assert kinds(box) == ["argv", "train"] * 3
    assert not scores(box)
    heads_lines = lines_with(r.stdout, "the heads of 10k")
    assert "try 1 of 2 failed" in heads_lines[0]
    assert "FAILED (try 2 of 2): the trainer ended with rc=1" in heads_lines[1]
    assert "last pass" in heads_lines[2] and "FAILED (try 1 of 1)" in heads_lines[2]
    assert len(lines_with(r.stdout, "FAILED")) == 2
    assert "no score: rowa 10k" in r.stdout
    assert r.stdout.splitlines()[-1].endswith("FOLLOW_END rc=1")


def test_a_runner_that_refuses_once_is_tried_again(box):
    """The 25k stops of 11-27: the runner refused an empty token."""
    save_ckpt(box, "rowa", 10)
    r = follow(box, ["rowa"], "10", FF_TEST_ARGV_FAILS="1")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert kinds(box) == ["argv", "argv", "train", "score", "collect"]
    assert ("the heads of 10k: try 1 of 3 failed: no head flags for rowa "
            "(ABORT: empty HF_TOKEN)") in r.stdout
    assert scores(box) == {tag(): "0.9500"}


def test_a_gpu_with_no_free_memory_is_tried_again(box):
    save_ckpt(box, "rowa", 10)
    r = follow(box, ["rowa"], "10", FF_TEST_SMI_LOW="1",
               FF_WAVE_VRAM_MIB="8000")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "try 1 of 3 failed: GPU 0 has 100 MiB free after 0 s" in r.stdout
    assert kinds(box).count("train") == 1 and scores(box) == {tag(): "0.9500"}


def test_a_stream_wait_that_ends_is_tried_again(box):
    save_ckpt(box, "rowa", 10)
    out = box[0] / "follow.log"
    (box[0] / "base" / "locks").mkdir()
    with open(box[0] / "base" / "locks" / "stream.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = start_follow(box, ["rowa"], "10", out, FF_STREAM_WAIT="1",
                         FF_HEAD_RETRY_WAIT="5")
        wait_for(out, "try 1 of 3 failed: another wave reads the data stream after 1 s")
    assert p.wait(timeout=300) == 0, out.read_text()[-3000:]
    assert kinds(box) == ["argv", "train", "score", "collect"]


def test_each_try_trains_only_the_runs_with_no_final_head(box):
    for run in ("rowa", "rowb"):
        save_ckpt(box, run, 10)
    r = follow(box, ["rowa", "rowb"], "10", FF_TEST_TRAIN_FAILS="1",
               FF_TEST_TRAIN_FIRST_ONLY="1")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    trains = [c[2:] for c in calls(box) if c[0] == "train"]
    assert trains == [[tag("rowa"), tag("rowb")], [tag("rowb")]]
    argv = [c[1] for c in calls(box) if c[0] == "argv"]
    assert argv == [tag("rowa"), tag("rowb"), tag("rowb")]
    assert sorted(scores(box)) == [tag("rowa"), tag("rowb")]


# The score of a run at a stop: up to FF_HEAD_TRIES tries.

def test_a_score_that_fails_once_is_tried_again(box):
    save_ckpt(box, "rowa", 10)
    r = follow(box, ["rowa"], "10", FF_TEST_SCORE_FAILS="1")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert kinds(box) == ["argv", "train", "score", "collect", "score", "collect"]
    assert ("the score of rowa 10k: try 1 of 3 failed: rc=1, score control FAILED"
            in r.stdout)
    assert "the score of rowa 10k: try 2 of 3 worked" in r.stdout
    assert scores(box) == {tag(): "0.9500"} and "FAILED (" not in r.stdout


def test_a_score_that_always_fails_gives_FAILED_after_the_tries(box):
    save_ckpt(box, "rowa", 10)
    r = follow(box, ["rowa"], "10", FF_HEAD_TRIES="2", FF_TEST_SCORE_FAILS="99")
    assert r.returncode == 1
    # Two tries, then one more at the last pass. The head trains once.
    assert kinds(box) == ["argv", "train"] + ["score", "collect"] * 3
    assert "the score of rowa 10k FAILED (try 2 of 2)" in r.stdout
    assert "train SKIP at 10k" in r.stdout      # the last pass trains no head
    assert r.stdout.splitlines()[-1].endswith("FOLLOW_END rc=1")


# The order of the stops, and the last pass.

def test_the_next_stop_goes_on_and_the_last_pass_tries_a_failed_stop_again(box):
    for stop in (10, 25):
        save_ckpt(box, "rowa", stop)
    r = follow(box, ["rowa"], "10 25", FF_HEAD_TRIES="2", FF_TEST_TRAIN_FAILS="2")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    trains = [c[2] for c in calls(box) if c[0] == "train"]
    assert trains == [tag(stop=10), tag(stop=10), tag(stop=25), tag(stop=10)]
    assert sorted(scores(box)) == [tag(stop=10), tag(stop=25)]
    log = r.stdout
    assert (log.index("the heads of 10k FAILED (try 2 of 2)")
            < log.index("train: 1 heads at 25k") < log.index("last pass at 10k: rowa"))
    assert log.splitlines()[-1].endswith("FOLLOW_END rc=0")


def test_a_checkpoint_that_never_comes_ends_the_follow_with_code_1(box):
    save_ckpt(box, "rowa", 10)
    r = follow(box, ["rowa"], "10 25", FF_WAIT_MAX="2")
    assert r.returncode == 1
    assert "STOP: no checkpoint of rowa at 25k" in r.stdout
    assert scores(box) == {tag(stop=10): "0.9500"}
    assert kinds(box).count("train") == 1   # the last pass needs a checkpoint copy
    assert "no score: rowa 25k" in r.stdout


@pytest.mark.parametrize("knobs", [{"FF_HEAD_TRIES": "0"},
                                   {"FF_HEAD_TRIES": "two"},
                                   {"FF_HEAD_RETRY_WAIT": "-1"}])
def test_a_wrong_count_of_tries_or_wait_is_refused(box, knobs):
    save_ckpt(box, "rowa", 10)
    r = follow(box, ["rowa"], "10", **knobs)
    assert r.returncode == 2 and "ABORT" in r.stdout
    assert not calls(box)
