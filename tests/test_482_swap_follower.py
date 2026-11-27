"""#482: swap_follower.sh moves the head follower of a box to a new score code.

The swap must stop a follower only while it waits for a checkpoint: no head
trains and no score runs. It freezes the followers before it looks for a head
or a score, so that no head starts between the check and the stop. It reads
the stops with no score before it stops a follower. It copies the new folder
next to the score code (the running follower reads $H/ff/code), moves it in
place after the stop, and starts the follower with the start script of the
box for the stops with no score. The tests replace ssh and rsync (fake_box/):
the "box" is a local folder, and each process of its follower has
FF_BASE=<box>/ff. The tests stop only the processes with this FF_BASE.
"""
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "reports" / "2026-11-27_paper_ablations" / "scripts"
FAKE_BOX = SCRIPTS / "fake_box"
RUN, SHORT, FOLDER = "g166b_w320e6f6_ab01_no_synth", "ab01c", "ff_code_1234abcd"
NEW_COMMIT = "a416b3ae + #482 patch 1234abcd"
FOLLOW_DIR = "reports/2026-10-10_frequency_family_head/scripts"

# follow_multi.sh of the tests: a follower that waits for a checkpoint. When
# the file head_now appears in FF_BASE (a checkpoint came), it starts a head
# trainer, and the head writes head_runs.
FOLLOWER = r'''#!/bin/bash
echo "[stub] start: stops $FF_STOPS, code $(cat ../../../DEPLOYED_COMMIT), trainer ${FF_TRAINER:-none}"
while :; do
  if [ -e "$FF_BASE/head_now" ]; then
    rm -f "$FF_BASE/head_now"
    python3 -c 'import pathlib, sys, time; pathlib.Path(sys.argv[1]).touch(); time.sleep(300)' \
      "$FF_BASE/head_runs" --jobs x &
  fi
  sleep 0.2
done
'''

# start_follow_w320e6f6_box.sh of the tests, with the box folder FAKE_BOX_H.
START = r'''#!/bin/bash
H="$FAKE_BOX_H"
cd "$H/ff/code/reports/2026-10-10_frequency_family_head/scripts" || exit 2
FF_BASE="$H/ff" FF_STOPS="$E_STOPS" setsid nohup bash follow_multi.sh >>"$H/ff/results/$E_LOG" 2>&1 </dev/null &
echo "follow_multi.sh started on the box (pid $!)"
'''


def write(path, text, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


def code_folder(path, commit, token):
    write(path / "DEPLOYED_COMMIT", commit + "\n")
    write(path / "experiments" / "hf_token.txt", token + "\n", 0o600)
    write(path / FOLLOW_DIR / "follow_multi.sh", FOLLOWER)


def box_procs(h):
    """(pid, command line) of each process with FF_BASE=<h>/ff."""
    want, out = f"FF_BASE={h}/ff".encode(), []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if want in env:
            out.append((int(pid), [c.decode() for c in cmd if c]))
    return out


def followers(h):
    return [pid for pid, cmd in box_procs(h) if cmd[-1:] == ["follow_multi.sh"]]


def state(pid):
    """The state letter of a process: T when it is stopped (frozen)."""
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]


def start_follower(box, stops, log):
    h, env = box
    subprocess.run(["bash", "start_follow_w320e6f6_box.sh"], cwd=h / "ff", check=True,
                   env=dict(env, E_STOPS=stops, E_LOG=log))


def stop_all(h):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, _ in box_procs(h):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(1)


@pytest.fixture
def box(tmp_path):
    """The elisa folders (CF) and the box folder H, with a follower that waits
    for the checkpoint of 50k. The stops 10k and 25k have their score."""
    cf, h = tmp_path / "cf", tmp_path / "box"
    code_folder(cf / FOLDER, NEW_COMMIT, "hf_card")
    write(cf / "secrets" / "hf_token_agent.txt", "hf_card\n", 0o600)
    code_folder(h / "ff" / "code", "a416b3ae", "hf_old")
    write(h / "ff" / "start_follow_w320e6f6_box.sh", START)
    write(h / "ff" / "train_heads_tf32.py", "heads\n")
    for k in (10, 25):
        write(h / "ff" / "results" / f"score_{RUN}_bb{k}k_h30k_control.txt", "0.95\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FF_", "E_"))}
    env.update(PATH=f"{FAKE_BOX}:{os.environ['PATH']}", REAL_RSYNC=shutil.which("rsync"),
               FAKE_LOG=str(tmp_path / "fake.log"), FAKE_BOX_H=str(h), CF=str(cf),
               BOX_H=str(h), BOX="root@box", BOX_PORT="22", RUN=RUN, SHORT=SHORT,
               FOLDER=FOLDER)
    start_follower((h, env), "10 25 50 75 100 125 150 166", f"follow_{SHORT}_box.log")
    wait_until(lambda: followers(h))
    yield h, env
    stop_all(h)


def wait_until(test, timeout=30):
    end = time.time() + timeout
    while not test():
        assert time.time() < end, "the wait ended"
        time.sleep(0.2)


def swap(box, **extra):
    h, env = box
    return subprocess.run(["bash", str(SCRIPTS / "swap_follower.sh")], capture_output=True,
                          text=True, env=dict(env, **extra), timeout=300)


def head_trainer(box, token):
    """A process of a head trainer of the box, with HF_TOKEN=<token>."""
    h, env = box
    return subprocess.Popen(["python3", "-c", "import time; time.sleep(300)", "--jobs", "x"],
                            env=dict(env, FF_BASE=f"{h}/ff", HF_TOKEN=token))


def box_log(h):
    return (h / "ff" / "results" / f"follow_{SHORT}_box.log").read_text()


def test_a_swap_of_an_idle_box_starts_the_follower_on_the_new_code(box):
    h, _ = box
    (old,) = followers(h)
    r = swap(box)
    assert r.returncode == 0, r.stdout + r.stderr
    assert old not in followers(h)
    code = h / "ff" / "code"
    assert (code / "DEPLOYED_COMMIT").read_text() == NEW_COMMIT + "\n"
    assert (code / "experiments" / "hf_token.txt").read_text() == "hf_card\n"
    (kept,) = h.joinpath("ff").glob("code_*")       # the old code, moved aside
    assert (kept / "DEPLOYED_COMMIT").read_text() == "a416b3ae\n"
    assert not (h / "ff" / FOLDER).exists()
    (new,) = followers(h)
    assert os.readlink(f"/proc/{new}/cwd") == str(code / FOLLOW_DIR)
    log = box_log(h).splitlines()
    assert log[0].startswith("[stub] start: stops 10 25 50")         # the old follower
    assert log[-1] == (f"[stub] start: stops 50 75 100 125 150 166, code {NEW_COMMIT}, "
                       f"trainer {h}/ff/train_heads_tf32.py")
    assert "md5 match" in r.stdout and f"new follower (PID and folder): {new} " in r.stdout


def test_a_box_where_a_head_trains_keeps_its_follower(box):
    h, _ = box
    (old,) = followers(h)
    trainer = head_trainer(box, "hf_card")
    try:
        wait_until(lambda: any(p == trainer.pid for p, _ in box_procs(h)))
        r = swap(box)
    finally:
        trainer.kill()
    assert r.returncode == 3, r.stdout + r.stderr
    assert "a head or a score runs" in r.stdout
    assert followers(h) == [old] and state(old) != "T"
    assert (h / "ff" / "code" / "DEPLOYED_COMMIT").read_text() == "a416b3ae\n"
    assert not list(h.joinpath("ff").glob("code_*"))


def kill_after_head(base):
    """A bash function `kill` for the box script: before its first signal, a
    checkpoint comes and the follower starts a head. The function waits until
    the head runs, then sends the signal."""
    return (f'() {{ if [ ! -e {base}/head_asked ]; then touch {base}/head_asked {base}/head_now; '
            f'until [ -e {base}/head_runs ]; do sleep 0.05; done; fi; builtin kill "$@"; }}')


def test_a_head_that_starts_between_the_check_and_the_stop_keeps_the_follower(box):
    h, _ = box
    base = h / "ff"
    (old,) = followers(h)
    r = swap(box, **{"BASH_FUNC_kill%%": kill_after_head(base)})
    assert r.returncode == 3, r.stdout + r.stderr
    assert "a head or a score runs" in r.stdout
    assert followers(h) == [old] and state(old) != "T"
    assert (base / "code" / "DEPLOYED_COMMIT").read_text() == "a416b3ae\n"
    assert not list(base.glob("code_*"))
    (base / "head_runs").unlink()                    # the follower goes on: a second head
    (base / "head_now").touch()
    wait_until(lambda: (base / "head_runs").exists())


def term_after_stop():
    """A bash function `kill` for the box script: right after its SIGSTOP of
    the followers, the box script gets a SIGTERM, and its trap ends it."""
    return '() { builtin kill "$@"; [ "$1" != -STOP ] || builtin kill -TERM $$; }'


def test_a_sigterm_right_after_the_sigstop_leaves_no_follower_frozen(box):
    h, _ = box
    start_follower(box, "25", f"follow_{SHORT}_25k_box.log")      # the recovery follower
    wait_until(lambda: len(followers(h)) == 2)
    old = sorted(followers(h))
    r = swap(box, **{"BASH_FUNC_kill%%": term_after_stop()})
    assert r.returncode == 2, r.stdout + r.stderr
    assert "the stop of the followers failed (exit code 1)" in r.stdout
    assert sorted(followers(h)) == old
    assert "T" not in [state(pid) for pid in old]
    assert (h / "ff" / "code" / "DEPLOYED_COMMIT").read_text() == "a416b3ae\n"


@pytest.mark.parametrize("fault", [{"FAKE_SSH_FAIL": "-- stops "},     # ssh fails
                                   {"FAKE_SSH_HEAD": "-- stops :0"},   # no answer
                                   {"FAKE_SSH_HEAD": "-- stops :1"}])  # 1 stop of 8
def test_a_failed_query_of_the_stops_stops_no_follower(box, fault):
    h, _ = box
    (old,) = followers(h)
    r = swap(box, **fault)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "Nothing is stopped" in r.stdout
    assert followers(h) == [old] and state(old) != "T"
    assert (h / "ff" / "code" / "DEPLOYED_COMMIT").read_text() == "a416b3ae\n"


def test_a_box_where_each_stop_has_its_score_keeps_its_follower(box):
    h, _ = box
    (old,) = followers(h)
    for k in (50, 75, 100, 125, 150, 166):
        write(h / "ff" / "results" / f"score_{RUN}_bb{k}k_h30k_control.txt", "0.95\n")
    r = swap(box)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "each stop of STOPS has its score" in r.stdout
    assert followers(h) == [old] and not (h / "ff" / FOLDER).exists()


def test_a_swap_stops_the_main_and_the_recovery_follower_and_starts_one(box):
    h, _ = box
    (h / "ff" / "results" / f"score_{RUN}_bb25k_h30k_control.txt").unlink()
    start_follower(box, "25", f"follow_{SHORT}_25k_box.log")      # the recovery follower
    wait_until(lambda: len(followers(h)) == 2)
    old = set(followers(h))
    r = swap(box)
    assert r.returncode == 0, r.stdout + r.stderr
    (new,) = followers(h)
    assert new not in old
    assert box_log(h).splitlines()[-1].startswith(
        f"[stub] start: stops 25 50 75 100 125 150 166, code {NEW_COMMIT}")


def test_no_follower_and_a_log_that_grew_in_the_last_10_min_stop_the_swap(box):
    h, _ = box
    stop_all(h)
    r = swap(box)
    assert r.returncode == 4, r.stdout + r.stderr
    assert not followers(h)
    assert (h / "ff" / "code" / "DEPLOYED_COMMIT").read_text() == "a416b3ae\n"


def test_no_follower_and_logs_of_11_min_ago_let_the_swap_go_on(box):
    h, _ = box
    stop_all(h)
    then = time.time() - 11 * 60
    for log in (h / "ff" / "results").glob("follow_*.log"):
        os.utime(log, (then, then))
    r = swap(box)
    assert r.returncode == 0, r.stdout + r.stderr
    (new,) = followers(h)
    assert (h / "ff" / "code" / "DEPLOYED_COMMIT").read_text() == NEW_COMMIT + "\n"


def test_a_follower_that_ends_after_its_start_fails_the_swap(box):
    h, env = box
    write(Path(env["CF"]) / FOLDER / FOLLOW_DIR / "follow_multi.sh", "#!/bin/bash\nexit 2\n")
    r = swap(box)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "no follower runs" in r.stdout


def test_a_copy_with_another_md5_changes_nothing(box):
    h, _ = box
    (old,) = followers(h)
    r = swap(box, FAKE_RSYNC_CORRUPT=FOLDER)
    assert r.returncode == 2 and "MISMATCH" in r.stdout
    assert followers(h) == [old]
    assert (h / "ff" / "code" / "DEPLOYED_COMMIT").read_text() == "a416b3ae\n"


def test_a_second_swap_starts_the_follower_again_on_the_same_code(box):
    h, _ = box
    assert swap(box).returncode == 0
    (first,) = followers(h)
    r = swap(box)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "has the score code of ff_code_1234abcd already" in r.stdout
    (second,) = followers(h)
    assert second != first and len(list(h.joinpath("ff").glob("code_*"))) == 1


def test_the_check_names_the_token_of_each_head_trainer_and_changes_nothing(box):
    h, _ = box
    (old,) = followers(h)
    trainers = {name: head_trainer(box, token) for name, token in
                (("card", "hf_card"), ("other", "hf_other"), ("none", ""))}
    try:
        pids = {p.pid for p in trainers.values()}
        wait_until(lambda: pids <= {pid for pid, _ in box_procs(h)})
        r = swap(box, MODE="check")
    finally:
        for p in trainers.values():
            p.kill()
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"head trainer {trainers['card'].pid}: the card token" in r.stdout
    assert f"head trainer {trainers['other'].pid}: another token" in r.stdout
    assert f"head trainer {trainers['none'].pid}: no token" in r.stdout
    assert "stops with no score: 50 75 100 125 150 166" in r.stdout
    assert "hf_card" not in r.stdout + r.stderr
    assert followers(h) == [old] and not (h / "ff" / FOLDER).exists()


@pytest.mark.parametrize("folder", ["", "code", "ff_code_12", "../ff_code_1234abcd"])
def test_a_wrong_folder_name_is_refused_before_any_ssh(box, folder):
    r = swap(box, FOLDER=folder)
    assert r.returncode != 0
    assert not Path(box[1]["FAKE_LOG"]).exists()
