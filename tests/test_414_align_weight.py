"""Tests for #414: the `align_w` column, and the orphan-stop detector.

`run_leg_k.sh` hardcodes `--align-loss-weight 1.0`, and no study of this
lineage has moved it. At 400,000 steps the best arm of #412 reads a total loss
of 0.7110 with `l_align` at 0.1744, so L_align carries about a quarter of the
objective once `L_rep` has decayed away.

Three properties:

1. `-` in the column keeps the runner's own weight, and the arm passes no
   flag. Every arm of #412 reads `-`, so each one builds the command line it
   built before the column existed.
2. A value in the column appends `--align-loss-weight <value>` to the END of
   the trainer command line, where a repeated flag wins.
3. `orphan_stops.sh` reports a stop that holds a checkpoint and no score, and
   reports nothing when every stop is covered.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDY = REPO_ROOT / "reports" / "2026-09-06_moirai_small_size"
ARMS_TSV = STUDY / "scripts" / "arms.tsv"


def arms_rows():
    return [l.split("\t") for l in ARMS_TSV.read_text().splitlines()
            if l.strip() and not l.startswith("#") and not l.startswith("arm\t")]


def dry_run(arm, stop="40000"):
    env = {"CF412_DRY_RUN": "1", "BB_GPU": "0", "PATH": "/usr/bin:/bin",
           "HOME": str(Path.home())}
    return subprocess.run(["bash", "scripts/run_arm.sh", arm, stop],
                          cwd=STUDY, capture_output=True, text=True,
                          env=env).stdout


class TestTheColumn:

    def test_every_row_carries_the_column(self):
        rows = arms_rows()
        assert rows, "arms.tsv holds no row"
        assert all(len(r) == 10 for r in rows), \
            [r[0] for r in rows if len(r) != 10]

    def test_the_header_names_it_last(self):
        header = [l for l in ARMS_TSV.read_text().splitlines()
                  if l.startswith("# arm\t")]
        assert len(header) == 1
        assert header[0].lstrip("# ").split("\t")[-1] == "align_w"

    def test_a_dash_passes_no_flag(self):
        held = [r[0] for r in arms_rows() if r[9] == "-"]
        assert held, "no arm holds the runner's weight"
        assert "--align-loss-weight" not in dry_run(held[0])

    def test_a_value_appends_the_flag(self):
        out = dry_run("k3_r100_09_lr56_fix09_dec10k_aw3")
        assert "--align-loss-weight 3.0" in out

    def test_the_new_arm_moves_the_weight_alone(self):
        out = subprocess.run(
            [sys.executable, str(STUDY / "scripts" / "arm_diff.py"),
             "k3_r100_09_lr56_fix09_dec10k",
             "k3_r100_09_lr56_fix09_dec10k_aw3"],
            capture_output=True, text=True, check=True).stdout
        assert "SINGLE AXIS" in out


class TestOrphanStops:

    def _run(self, root):
        env = dict(os.environ, CF412_CKPT_ROOT=str(root))
        return subprocess.run(["bash", "scripts/orphan_stops.sh"], cwd=STUDY,
                              capture_output=True, text=True, env=env).stdout

    def test_a_checkpoint_with_no_score_is_an_orphan(self, tmp_path):
        leg = (tmp_path / "some_arm" / "arm6_v2_combab_alignT" / "leg_400k")
        leg.mkdir(parents=True)
        (leg / "cf393_some_arm_400k.pth").write_bytes(b"x")
        out = self._run(tmp_path)
        assert "ORPHAN some_arm 400k" in out
        assert "1 orphan stop(s)" in out

    def test_a_directory_with_no_checkpoint_is_not_an_orphan(self, tmp_path):
        leg = (tmp_path / "some_arm" / "arm6_v2_combab_alignT" / "leg_400k")
        leg.mkdir(parents=True)
        (leg / "cf393_some_arm_losses.csv").write_text("step\n1\n")
        assert "0 orphan stop(s)" in self._run(tmp_path)
