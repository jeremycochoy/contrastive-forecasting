"""Tests for `scripts/hub_gate.sh`, the Hub-outage gate every lane shares.

On 2026-08-23 at 18:48 elisa lost DNS. The training data streams from the
Hugging Face Hub, so every leg died in about 3 seconds with a connection
error. A lane read those 3-second deaths as failed arms, spent its whole
retry ladder in two minutes, and moved on. Three arms went that way in seven
minutes and the card then sat idle for 27 hours.

The gate tells a network failure apart from a crash, and it waits over hours
instead of minutes. It holds four public functions.

  * `hub_outage_in_text` reads a trainer tail and says whether the Hub was
    unreachable. Every pattern comes from the log of that outage.
  * `hub_is_up` probes the Hub once.
  * `hub_backoff_delay` grows the delay between probes and caps it.
  * `hub_wait_up` blocks until the Hub answers, or until a deadline.

`HUB_GATE_PROBE` replaces the probe, so these tests never touch the network.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HUB_GATE = REPO_ROOT / "scripts" / "hub_gate.sh"

# Tails of `results/leg_arm6_v2_combab_alignT.log`, 2026-08-23 18:46 to 18:55.
OUTAGE_TAILS = (
    "requests.exceptions.ConnectionError: (MaxRetryError("
    "'HTTPSConnectionPool(host=\\'huggingface.co\\', port=443): Max retries "
    "exceeded with url: /datasets/jeremycochoy/gift-pretrain-full-4096/"
    "resolve/main/small_v1/shard_00032.parquet (Caused by "
    "NameResolutionError(\"<urllib3.connection.HTTPSConnection object at "
    "0x731095336020>: Failed to resolve 'huggingface.co' ([Errno -3] "
    "Temporary failure in name resolution)\"))')",
    "ConnectionError: Couldn't reach 'jeremycochoy/gift-pretrain-full-4096' "
    "on the Hub (ConnectionError)",
)

# Failures a lane MUST count against the ladder. A re-fire trains the same
# fault, so reading one of these as an outage would loop for hours.
CRASH_TAILS = (
    "[  22800] loss=0.7683  ema_loss=0.7515  gap=-0.2024  1.4 sps  AUC=0.9540",
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "RuntimeError: CUDA error: device-side assert triggered",
    "AssertionError: loss is nan at step 1200",
    "FileNotFoundError: [Errno 2] No such file or directory: 'model.pth'",
)


def gate(snippet: str, env=None) -> subprocess.CompletedProcess:
    """Run one snippet against the sourced library."""
    full = dict(os.environ)
    full.update(env or {})
    return subprocess.run(
        ["bash", "-c", f'. "{HUB_GATE}" && {snippet}'],
        capture_output=True, text=True, timeout=120, env=full)


def gate_out(snippet: str, env=None) -> str:
    out = gate(snippet, env)
    assert out.returncode == 0, f"{snippet}: {out.stderr}"
    return out.stdout.strip()


def probe_stub(tmp_path: Path, down_calls: int) -> dict:
    """A probe that reports the Hub down for the first `down_calls` reads.

    The counter is a file, because each probe runs in its own subshell.
    """
    counter = tmp_path / "probes"
    script = tmp_path / "probe.sh"
    script.write_text(
        "#!/bin/bash\n"
        f'n=$(cat "{counter}" 2>/dev/null || echo 0)\n'
        "n=$(( n + 1 ))\n"
        f'printf "%s" "$n" >"{counter}"\n'
        f'[ "$n" -gt {down_calls} ]\n')
    script.chmod(0o755)
    return {"HUB_GATE_PROBE": f"bash {script}", "_counter": str(counter)}


def probes_run(stub: dict) -> int:
    return int(Path(stub["_counter"]).read_text() or 0)


def env_of(stub: dict) -> dict:
    return {k: v for k, v in stub.items() if not k.startswith("_")}


class TestTheLibraryIsSourceable:

    def test_the_file_exists(self):
        assert HUB_GATE.exists()

    def test_it_is_a_library_and_runs_nothing(self):
        out = subprocess.run(["bash", str(HUB_GATE)], capture_output=True,
                             text=True, timeout=60)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == ""


class TestANetworkFailureIsNotACrash:
    """Rule 1 of the card: a leg that dies because the Hub is unreachable is
    not a failed arm."""

    @pytest.mark.parametrize("tail", OUTAGE_TAILS)
    def test_the_outage_tails_of_08_23_are_read_as_outages(self, tail):
        assert gate('hub_outage_in_text "$TAIL"', env={"TAIL": tail}).returncode == 0

    @pytest.mark.parametrize("tail", CRASH_TAILS)
    def test_a_real_crash_is_not_read_as_an_outage(self, tail):
        assert gate('hub_outage_in_text "$TAIL"', env={"TAIL": tail}).returncode == 1

    def test_every_pattern_the_library_lists_matches_itself(self):
        patterns = gate_out("hub_outage_patterns").splitlines()
        assert len(patterns) >= 8
        for p in patterns:
            out = gate('hub_outage_in_text "$TAIL"', env={"TAIL": f"boom: {p} here"})
            assert out.returncode == 0, p

    def test_it_reads_the_tail_of_a_log_not_the_head(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text(OUTAGE_TAILS[0] + "\n" + "[ 100] loss=1.0\n" * 400)
        assert gate(f'hub_outage_in_log "{log}" 40').returncode == 1
        assert gate(f'hub_outage_in_log "{log}" 500').returncode == 0

    def test_a_log_that_is_not_there_is_not_an_outage(self, tmp_path):
        assert gate(f'hub_outage_in_log "{tmp_path}/nope.log"').returncode == 1


class TestTheDelayGrowsAndIsCapped:
    """Rule 2 of the card: wait over hours, not minutes."""

    def test_the_first_delay_is_the_base(self):
        assert gate_out("hub_backoff_delay 1 60 1800") == "60"

    def test_each_try_doubles_the_one_before(self):
        got = [int(gate_out(f"hub_backoff_delay {t} 60 1800")) for t in
               range(1, 6)]
        assert got == [60, 120, 240, 480, 960]

    def test_the_delay_stops_at_the_cap(self):
        for t in (7, 12, 40):
            assert gate_out(f"hub_backoff_delay {t} 60 1800") == "1800"

    def test_the_defaults_ride_out_a_thirty_minute_outage(self):
        """The card's own case: 30 minutes of DNS loss must not end a study."""
        base = int(gate_out("printf %s $HUB_GATE_BASE_WAIT"))
        cap = int(gate_out("printf %s $HUB_GATE_MAX_WAIT"))
        total, t = 0, 1
        while total < 1800:
            total += int(gate_out(f"hub_backoff_delay {t} {base} {cap}"))
            t += 1
        assert t <= 8, "30 minutes of outage costs more than 8 probes"
        assert int(gate_out("printf %s $HUB_GATE_DEADLINE")) >= 3 * 3600


class TestTheGateWaitsForTheHub:
    """Rule 4 of the card: check that the Hub answers before an arm starts."""

    def test_a_hub_that_answers_returns_at_once(self, tmp_path):
        stub = probe_stub(tmp_path, down_calls=0)
        assert gate("hub_wait_up 600", env=env_of(stub)).returncode == 0
        assert probes_run(stub) == 1

    def test_it_probes_again_while_the_hub_is_down(self, tmp_path):
        stub = probe_stub(tmp_path, down_calls=3)
        env = env_of(stub) | {"HUB_GATE_BASE_WAIT": "1", "HUB_GATE_MAX_WAIT": "1"}
        assert gate("hub_wait_up 600", env=env).returncode == 0
        assert probes_run(stub) == 4

    def test_it_gives_up_at_the_deadline(self, tmp_path):
        stub = probe_stub(tmp_path, down_calls=10_000)
        env = env_of(stub) | {"HUB_GATE_BASE_WAIT": "1", "HUB_GATE_MAX_WAIT": "1"}
        out = gate("hub_wait_up 3", env=env)
        assert out.returncode == 1
        assert probes_run(stub) >= 2

    def test_it_names_the_wait_on_stdout(self, tmp_path):
        stub = probe_stub(tmp_path, down_calls=1)
        env = env_of(stub) | {"HUB_GATE_BASE_WAIT": "1", "HUB_GATE_MAX_WAIT": "1"}
        out = gate("hub_wait_up 600", env=env)
        assert "huggingface.co" in out.stdout
        assert "up" in out.stdout.lower()

    def test_the_default_probe_reaches_the_hub_host(self):
        assert gate_out("printf %s $HUB_GATE_HOST") == "huggingface.co"
        body = HUB_GATE.read_text()
        assert "curl" in body
