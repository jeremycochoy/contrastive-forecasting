"""#482: the box scripts of the paper ablations
(reports/2026-11-27_paper_ablations/scripts/setup_box.sh and start_box.sh).

setup_box.sh must end with a non-zero exit code at a failed copy or an md5
mismatch, and write the file setup_done_<run> on the box only as its last
step. start_box.sh must refuse a box without this file. The tests replace
ssh and rsync: the "box" is a local folder, and ssh runs its command here.
"""
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "reports" / "2026-11-27_paper_ablations" / "scripts"
RUN, CODE = "g166b_w320e6f6_ab04_instnorm", "code_0123abcd"
REAL_RSYNC = shutil.which("rsync")
# ssh and rsync of the tests: the box is a local folder (fake_box/).
FAKE_BOX = SCRIPTS / "fake_box"

# setup_hp64_box.sh of #425 in the tests: the head code to the box, and the
# lines of its checks. FAKE_PIP_RC sets the exit code that pip reports.
FAKE_BASE_SETUP = r'''#!/bin/bash
mkdir -p "$BOX_H/ff" && cp -r "$HP64/ff/code" "$BOX_H/ff/code"
echo "head code md5 match"
echo "pip rc ${FAKE_PIP_RC:-0}"
echo "gift-eval rc 0"
echo "GIFT-Eval data md5 match"
'''


def write(path, text, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


@pytest.fixture
def env(tmp_path):
    """The folders of elisa (CF, HP64), the box folder and the fake tools.
    Returns the environment of the scripts."""
    cf, hp, box = tmp_path / "cf", tmp_path / "hp64", tmp_path / "box"
    write(cf / CODE / "src" / "models.py", "code\n")
    write(cf / "box_train" / RUN / f"run_{RUN}.sh", "  --encoder-type gru\n")
    write(cf / "box_train" / RUN / f"supervise_{RUN}.sh", "echo supervise\n")
    (cf / "box_train" / RUN / "ckpt").mkdir()
    write(cf / "secrets" / "hf_token_agent.txt", "hf_test\n")
    write(hp / "box" / "setup_hp64_box.sh", FAKE_BASE_SETUP)
    write(hp / "box" / "start_follow_w320e6f6_box.sh", "echo follower\n")
    write(hp / "bench_1123" / "train_heads_tf32.py", "heads\n")
    write(hp / "ff" / "code" / "experiments" / "hf_token.txt", "shared\n")
    heads_md5 = hashlib.md5(b"heads\n").hexdigest()[:8]
    return dict(os.environ, PATH=f"{FAKE_BOX}:{os.environ['PATH']}",
                REAL_RSYNC=REAL_RSYNC, FAKE_LOG=str(tmp_path / "fake.log"),
                CF=str(cf), HP64=str(hp), BOX_H=str(box),
                HEADS_MD5=heads_md5, BOX="root@box", BOX_PORT="22",
                CODE=CODE, RUN=RUN, MAX_HOURS="0")


def run(script, env, **extra):
    return subprocess.run(["bash", str(SCRIPTS / script)], capture_output=True,
                          text=True, env=dict(env, **extra), timeout=300)


def done_file(env):
    return Path(env["BOX_H"]) / f"setup_done_{RUN}"


def test_a_setup_with_no_failure_writes_the_setup_file(env):
    out = run("setup_box.sh", env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert done_file(env).read_text().startswith(f"run {RUN} code {CODE} ")
    checks = [line for line in out.stdout.splitlines()
              if line[:4].isdigit() and " md5 match " in line]
    assert len(checks) == 5          # its own checks, after those of #425
    assert (Path(env["BOX_H"]) / RUN / f"run_{RUN}.sh").exists()


@pytest.mark.parametrize("fault, why", [
    (dict(FAKE_RSYNC_FAIL=CODE), "rsync of"),
    (dict(FAKE_RSYNC_FAIL="hf_token.txt"), "rsync of"),
    (dict(FAKE_RSYNC_CORRUPT=RUN), "md5 MISMATCH"),
    (dict(FAKE_RSYNC_CORRUPT="train_heads_tf32.py"), "md5 MISMATCH"),
    (dict(FAKE_PIP_RC="1"), "no line 'pip rc 0'"),
    (dict(HEADS_MD5="00000000"), "is not md5"),
])
def test_each_failure_stops_the_setup_with_no_setup_file(env, fault, why):
    assert run("setup_box.sh", env).returncode == 0
    out = run("setup_box.sh", dict(env, BOX_H=env["BOX_H"] + "2"), **fault)
    assert out.returncode == 2, out.stdout + out.stderr
    assert why in out.stdout
    assert not (Path(env["BOX_H"] + "2") / f"setup_done_{RUN}").exists()


def test_a_failed_setup_removes_the_setup_file_of_an_earlier_setup(env):
    assert run("setup_box.sh", env).returncode == 0
    out = run("setup_box.sh", env, FAKE_RSYNC_FAIL=CODE)
    assert out.returncode == 2
    assert not done_file(env).exists()


START = dict(SHORT="ab04", LABEL="cf-482-ab04", ID="1")


def stop_the_puller(env):
    """The stop file of the puller: a puller that start_box.sh starts in a
    test ends at once."""
    write(Path(env["CF"]) / "logs" / "stop_pull_ab04", "")


def test_start_refuses_a_box_with_no_setup(env):
    stop_the_puller(env)
    out = run("start_box.sh", env, **START)
    assert out.returncode == 2
    assert "no successful setup_box.sh" in out.stdout
    assert "supervise" not in Path(env["FAKE_LOG"]).read_text()


def test_start_refuses_the_setup_of_another_run(env):
    stop_the_puller(env)
    write(done_file(env), "run other_run code x\n")
    out = run("start_box.sh", env, **START)
    assert out.returncode == 2
    assert "supervise" not in Path(env["FAKE_LOG"]).read_text()


def test_start_runs_after_a_setup(env):
    stop_the_puller(env)
    assert run("setup_box.sh", env).returncode == 0
    out = run("start_box.sh", env, **START)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"supervise_{RUN}.sh" in Path(env["FAKE_LOG"]).read_text()


# A code folder whose create_encoder builds a TimesFM block of one Linear,
# with a layer normalisation or not.
BLOCK = """import torch
def create_encoder(kind, w, h):
    layers = [torch.nn.Linear(w, h)]{}
    return torch.nn.Sequential(*layers)
"""


def code_folder(path, layer_norm):
    norm = "\n    layers.append(torch.nn.LayerNorm(h))" if layer_norm else ""
    write(path / "src" / "__init__.py", "")
    write(path / "src" / "encoders.py", BLOCK.format(norm))
    write(path / "experiments" / "hf_token.txt", "hf_test\n")


def test_row_6_needs_the_timesfm_block_of_its_run_code(env):
    cf = Path(env["CF"])
    write(cf / "box_train" / RUN / f"run_{RUN}.sh", "  --encoder-type timesfm\n")
    code_folder(cf / CODE, layer_norm=False)
    code_folder(cf / "ff_code_norm", layer_norm=True)
    code_folder(cf / "ff_code_same", layer_norm=False)
    out = run("setup_box.sh", env, HEAD_CODE="ff_code_norm")
    assert out.returncode == 2 and "TimesFM block" in out.stdout
    assert not done_file(env).exists()
    out = run("setup_box.sh", env, HEAD_CODE="ff_code_same")
    assert out.returncode == 0, out.stdout + out.stderr
    assert not list(cf.rglob("__pycache__"))
