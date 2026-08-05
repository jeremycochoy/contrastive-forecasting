"""#393 — the config split that lets one GIFT-Eval use several cores.

GIFT-Eval is 97 configs run one after another, and on 2026-08-05 it moved
off the rented GPUs onto elisa's CPUs, split across shards. Two things can
go wrong quietly there, and both put a number in the report rather than an
error:

  * a shard regex that matches nothing. `--config-filter` is applied to the
    RAW identifier (`SZ_TAXI/15T/short`), not the pretty one the script
    prints (`sz_taxi/15T/short`). A filter written against the pretty name
    keeps 0 of 97 configs, and the eval then "succeeds" in two seconds with
    an empty summary.

  * a split that drops or doubles a config. Either produces an aggregate
    GM-Relative MASE over the wrong set, which looks exactly like a real
    score.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(REPO, "experiments", "2026-08-04_ema_sched_ladder")
SHARDER = os.path.join(EXP, "scripts", "shard_configs.py")
COSTS = os.path.join(EXP, "results", "config_costs.csv")
GEVAL = os.path.join(REPO, "experiments", "2026-04-13_gift-eval", "scripts",
                     "eval_gift_eval_official.py")

N_CONFIGS = 97


def load_geval():
    """The official eval module, without running its main()."""
    spec = importlib.util.spec_from_file_location("geval_for_test", GEVAL)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def all_raw_configs():
    mod = load_geval()
    return [f"{d}/{t}" for d, t in mod.get_all_dataset_configs()]


def shard_regex(shards: int, shard: int) -> str:
    return subprocess.check_output(
        [sys.executable, SHARDER, "--shards", str(shards),
         "--shard", str(shard)], text=True).strip()


class TestTheCostTableCoversEveryConfig:
    """The weights are only weights, but a short table is a broken split."""

    def test_one_row_per_config_and_the_names_are_the_raw_ones(self):
        with open(COSTS) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == N_CONFIGS
        assert {r["raw_config"] for r in rows} == set(all_raw_configs())

    def test_the_weights_are_the_measured_ones(self):
        """Non-negative, and summing to the eval they were read from.

        `us_births/M/short` is 0.0 — it really does take under 0.05 s, and
        rounding it to zero costs the split nothing. What would matter is a
        total that has drifted from the 13,303 s reference, because that
        would mean the table is describing a different protocol.
        """
        with open(COSTS) as fh:
            seconds = [float(r["seconds"]) for r in csv.DictReader(fh)]
        assert min(seconds) >= 0
        assert sum(seconds) == pytest.approx(13303.2, abs=1.0)


@pytest.mark.parametrize("shards", [1, 2, 3, 4, 6, 8])
class TestTheSplitIsAPartition:

    def test_every_config_lands_in_exactly_one_shard(self, shards):
        configs = all_raw_configs()
        seen = []
        for i in range(shards):
            pat = re.compile(shard_regex(shards, i))
            seen += [c for c in configs if pat.search(c)]
        assert sorted(seen) == sorted(configs), "split is not a partition"
        assert len(seen) == len(set(seen)), "a config matched two shards"

    def test_no_shard_is_empty(self, shards):
        configs = all_raw_configs()
        for i in range(shards):
            pat = re.compile(shard_regex(shards, i))
            kept = [c for c in configs if pat.search(c)]
            assert kept, f"shard {i} of {shards} matches nothing"


class TestTheSplitIsBalanced:
    """Balance is the reason for the cost table; an unweighted split would
    put `electricity` and `loop_seattle` in one bin and the eval would take
    as long as it did before sharding."""

    def test_the_heaviest_shard_is_within_a_tenth_of_the_ideal(self):
        with open(COSTS) as fh:
            weight = {r["raw_config"]: float(r["seconds"])
                      for r in csv.DictReader(fh)}
        configs = all_raw_configs()
        for shards in (2, 4, 6):
            loads = []
            for i in range(shards):
                pat = re.compile(shard_regex(shards, i))
                loads.append(sum(weight[c] for c in configs if pat.search(c)))
            ideal = sum(weight.values()) / shards
            assert max(loads) <= 1.10 * ideal, (
                f"{shards} shards: heaviest {max(loads):.0f}s "
                f"against ideal {ideal:.0f}s")


class TestTheRegexIsAnchored:
    """An unanchored alternation would let one shard's name match another
    config that merely contains it, and the merge would count it twice."""

    def test_a_shard_regex_rejects_a_suffixed_name(self):
        rx = re.compile(shard_regex(4, 0))
        configs = all_raw_configs()
        one = next(c for c in configs if rx.search(c))
        assert not rx.search(one + "_extra")
        assert not rx.search("prefix_" + one)


class TestTheFilterMatchesTheIdentifierTheEvalUses:
    """The bug this whole file exists for: filtering on the pretty name."""

    def test_pretty_names_are_not_what_config_filter_sees(self):
        mod = load_geval()
        raw_upper = [f"{d}/{t}" for d, t in mod.get_all_dataset_configs()
                     if d != d.lower()]
        assert raw_upper, "expected some configs whose raw name is uppercased"
        rx = re.compile(shard_regex(4, 0))
        matched_raw = [c for c in all_raw_configs() if rx.search(c)]
        pretty = {mod.get_ds_config_name(*c.rsplit("/", 1))[0]
                  for c in matched_raw}
        # Some raw names differ from their pretty form; a filter written
        # against the pretty set would miss exactly those.
        assert pretty != set(matched_raw)
