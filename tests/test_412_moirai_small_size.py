"""Tests for #412: the four teacher-align cells, at the size of Moirai-2-Small.

The card moves ONE axis of the published cell, the width `d_model`, from 64 to
a value that gives about 11.4 million trainable parameters. Three groups of
guards follow from that.

1. The count itself. `src/model_size.py` gives three different numbers for one
    model, and the card quotes the wrong one. A parameter count and a sum over
    a checkpoint file are not the same number, because the patch encoder has
    two names in that file.
2. The shape that reaches the target. The chosen shape must sit near
    11.4 million, and it must differ from the published cell in the width
    alone.
3. The launcher. Every arm writes the same file names as its parent cell, so
    a study that writes where #373, #393, #401, #404 or #409 wrote destroys a
    published number. And a width that does not reach the trainer, the head
    trainer or the GIFT-Eval gives a run of the wrong model under this card's
    name.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP = REPO_ROOT / "reports" / "2026-09-06_moirai_small_size"
PARENT = REPO_ROOT / "reports" / "2026-08-08_rollout_depth"

STUDY_SH = EXP / "scripts" / "study.sh"
ARMS_TSV = EXP / "scripts" / "arms.tsv"
RUN_ARM = EXP / "scripts" / "run_arm.sh"
HEAD_EVAL = EXP / "scripts" / "head_eval.sh"
SIZE_TABLE = EXP / "scripts" / "size_table.sh"
SMOKE = EXP / "scripts" / "smoke.sh"
TRIAL = EXP / "scripts" / "trial.sh"
PHASE1 = EXP / "scripts" / "phase1.sh"
COLLECT = EXP / "scripts" / "collect.sh"
RUN_SH = EXP / "run.sh"

RUN_LEG = PARENT / "scripts" / "run_leg_k.sh"
BB_SHAPE = PARENT / "scripts" / "bb_shape.sh"
HEAD_EVAL_BB = PARENT / "scripts" / "head_eval_bb.sh"
EVAL_LOCAL = PARENT / "scripts" / "eval_local.sh"

MODEL_SIZE_CLI = REPO_ROOT / "scripts" / "model_size.py"

# Moirai-2-Small, from arXiv:2511.11698. The card's target.
MOIRAI_2_SMALL = 11_400_000

# The shape this card picked. `scripts/size_table.sh` measures the candidates.
CHOSEN_D_MODEL = 384
CHOSEN_LAYERS = 3
CHOSEN_ENCODER_LAYERS = 3


def bash(script: str, cwd: Path = REPO_ROOT, env: dict | None = None):
    """Run a shell snippet and give back the completed process."""
    full = dict(os.environ)
    full.update(env or {})
    return subprocess.run(["bash", "-c", script], cwd=str(cwd), env=full,
                          capture_output=True, text=True)


def study(script: str, env: dict | None = None):
    """Run a shell snippet with `study.sh` sourced, and give back its output."""
    proc = bash(f'. "{STUDY_SH}"\n{script}', env=env)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


# ---- 1. The three counts of one model ---------------------------------------


def test_parameter_counts_splits_the_trainable_from_the_frozen():
    """The teacher is frozen, so it is not part of the trainable count."""
    from src.model_size import build_backbone, parameter_counts

    counts = parameter_counts(build_backbone(H=64, num_layers=3,
                                             num_encoder_layers=3))
    assert counts["frozen"] > 0
    assert counts["total"] == counts["trainable"] + counts["frozen"]


def test_the_state_dict_sum_is_larger_than_the_parameter_count():
    """The patch encoder has two names in the file, so the sum counts it two times."""
    from src.model_size import build_backbone, parameter_counts

    model = build_backbone(H=64, num_layers=3, num_encoder_layers=3)
    counts = parameter_counts(model)
    encoder = sum(p.numel() for p in model.encoder.parameters())
    assert encoder > 0
    # The two buffers of the channel-mixing module are in the file as well.
    buffers = sum(b.numel() for b in model.buffers())
    assert counts["state_dict"] == counts["total"] + encoder + buffers


def test_state_dict_counts_reproduce_the_number_the_card_quotes():
    """The card's 1,135,774 is the file sum without the teacher, not a parameter count."""
    from src.model_size import build_backbone, parameter_counts, state_dict_counts

    model = build_backbone(H=64, num_layers=3, num_encoder_layers=3)
    file_counts = state_dict_counts(model.state_dict())
    counts = parameter_counts(model)
    assert file_counts["teacher"] == counts["frozen"]
    assert file_counts["student"] + file_counts["teacher"] == file_counts["total"]
    assert file_counts["total"] == counts["state_dict"]
    # The published cell trains 720,668 parameters. The card reads 1,135,774
    # off the file, and the difference is the patch encoder, counted two times.
    assert counts["trainable"] == 720_668
    assert file_counts["student"] == 1_135_774


def test_nearest_to_target_picks_the_closest_row():
    from src.model_size import nearest_to_target

    rows = [{"trainable": 8_000_000}, {"trainable": 11_431_548},
            {"trainable": 13_336_172}]
    assert nearest_to_target(rows, MOIRAI_2_SMALL) is rows[1]


# ---- 2. The shape that reaches Moirai-2-Small --------------------------------


def test_the_chosen_shape_reaches_the_size_of_moirai_2_small():
    from src.model_size import backbone_counts

    counts = backbone_counts(H=CHOSEN_D_MODEL, num_layers=CHOSEN_LAYERS,
                             num_encoder_layers=CHOSEN_ENCODER_LAYERS)
    error = abs(counts["trainable"] - MOIRAI_2_SMALL) / MOIRAI_2_SMALL
    assert error < 0.01, counts


def test_the_chosen_shape_is_the_nearest_of_the_candidates():
    """No other candidate width at this depth is nearer to the target."""
    from src.model_size import backbone_counts, nearest_to_target

    rows = [backbone_counts(H=h, num_layers=CHOSEN_LAYERS,
                            num_encoder_layers=CHOSEN_ENCODER_LAYERS)
            for h in (256, 320, 352, 384, 416)]
    assert nearest_to_target(rows, MOIRAI_2_SMALL)["H"] == CHOSEN_D_MODEL


def test_the_backbone_defaults_match_the_published_runner():
    """`CELL_BACKBONE` must stay the shape `run_leg_k.sh` trains."""
    from src.model_size import CELL_BACKBONE

    text = RUN_LEG.read_text()
    assert "--n-channels 1" in text
    assert "--d-model 64" in text
    assert "--n-heads 8" in text
    assert "--num-encoder-layers 3 --num-layers 3" in text
    assert "--rev-norm-kind ewma --rev-norm-span 128" in text
    assert "--freq-emb-dim 3 --seasonality-emb-dim 3" in text
    assert "--encoder-type gru" in text
    assert "--qk-norm --attn-out-norm" in text
    assert CELL_BACKBONE["C"] == 1
    assert CELL_BACKBONE["nhead"] == 8
    assert CELL_BACKBONE["rev_norm_kind"] == "ewma"
    assert CELL_BACKBONE["rev_norm_span"] == 128
    assert CELL_BACKBONE["freq_emb_dim"] == 3
    assert CELL_BACKBONE["seasonality_emb_dim"] == 3
    assert CELL_BACKBONE["encoder_type"] == "gru"
    assert CELL_BACKBONE["qk_norm"] is True
    assert CELL_BACKBONE["attn_out_norm"] is True
    assert CELL_BACKBONE["ema_embedding"] is True
    assert CELL_BACKBONE["ema_encoder"] is True


def test_the_size_command_prints_a_table_and_marks_the_target():
    proc = bash(f'python3 "{MODEL_SIZE_CLI}" --d-model-list 320,384 '
                f'--num-layers 3 --num-encoder-layers 3 --target {MOIRAI_2_SMALL}')
    assert proc.returncode == 0, proc.stderr
    assert "11431548" in proc.stdout.replace(",", "")
    # The nearest row carries the mark, and no other row does.
    marked = [ln for ln in proc.stdout.splitlines() if ln.strip().endswith("<-")]
    assert len(marked) == 1 and "384" in marked[0], proc.stdout


# ---- 3. The arms of the card -------------------------------------------------


def test_the_card_has_its_four_configurations():
    arms = study("printf '%s\\n' $CF412_ARMS").split()
    assert len(arms) >= 4
    shape = {a: (study(f'cf412_depth {a}'), study(f'cf412_reduce {a}'),
                 study(f'cf412_ema_sig {a}')) for a in arms[:4]}
    assert shape[arms[0]] == ("3", "sum", "0.9 1.0 100000")
    assert shape[arms[1]] == ("32", "mean", "0.9 1.0 100000")
    assert shape[arms[2]] == ("32", "mean", "0.8 1.0 200000")
    assert shape[arms[3]] == ("8", "mean", "0.9 1.0 100000")


def test_every_arm_targets_the_teacher():
    """`L_align` targets the EMA teacher on every arm of this card."""
    assert study("printf '%s' $CF412_CELL") == "arm6_v2_combab_alignT"
    assert "arm6_v2_combab_alignT)" in RUN_LEG.read_text()


def test_the_repeat_seed_moves_the_seed_column_alone():
    """The fifth arm measures this size's own seed band, so it changes one column."""
    arms = study("printf '%s\\n' $CF412_ARMS").split()
    assert len(arms) == 5
    first, repeat = arms[0], arms[4]
    assert study(f'cf412_depth {repeat}') == study(f'cf412_depth {first}')
    assert study(f'cf412_reduce {repeat}') == study(f'cf412_reduce {first}')
    assert study(f'cf412_ema_sig {repeat}') == study(f'cf412_ema_sig {first}')
    assert study(f'cf412_seed {repeat}') != study(f'cf412_seed {first}')


def test_the_stops_are_the_stops_the_parent_reports_use():
    assert study("echo $CF412_STOPS").split() == ["40000", "100000", "200000"]
    assert study("printf '%s' $CF412_HEAD_STEPS") == "30000"


# ---- 4. Nothing of this card lands where a published number lives -------------


def test_the_root_the_run_name_and_the_results_are_this_card_alone():
    root = study("printf '%s' $CF412_ROOT")
    assert root.rstrip("/").endswith("cf-412")
    for other in ("cf-373", "cf-393", "cf-401", "cf-404", "cf-409"):
        assert other not in root
    results = study("printf '%s' $CF412_RESULTS")
    assert results.startswith(str(EXP))
    arms = study("printf '%s\\n' $CF412_ARMS").split()
    names = {study(f'cf412_run_name {a}') for a in arms}
    assert len(names) == len(arms)
    for name in names:
        assert "cf412" in name
    roots = {study(f'cf412_arm_root {a}') for a in arms}
    assert len(roots) == len(arms)


def test_a_dry_run_names_the_width_the_depth_and_the_momentum():
    arms = study("printf '%s\\n' $CF412_ARMS").split()
    proc = bash(f'CF412_DRY_RUN=1 bash "{RUN_ARM}" {arms[1]} 40000')
    assert proc.returncode == 0, proc.stderr
    assert f"--d-model {CHOSEN_D_MODEL}" in proc.stdout
    assert "k=32" in proc.stdout
    assert "--ema-tau 0.9 --ema-tau-end 1.0 --ema-tau-ramp-steps 100000" in proc.stdout


def test_the_injected_shape_replaces_the_one_the_arm_carries():
    """`CF412_FORCE_ARCH` is what a wiring defect does, for the guard's own test."""
    arms = study("printf '%s\\n' $CF412_ARMS").split()
    proc = bash(f'CF412_DRY_RUN=1 CF412_FORCE_ARCH="--d-model 64" '
                f'bash "{RUN_ARM}" {arms[0]} 40000')
    assert proc.returncode == 0, proc.stderr
    gap = [ln for ln in proc.stdout.splitlines() if "gap=" in ln]
    assert len(gap) == 1 and "--d-model 64" in gap[0], proc.stdout
    assert f"--d-model {CHOSEN_D_MODEL}" not in gap[0]


# ---- 5. The width reaches the trainer, the head and the evaluation ------------


def test_the_arch_flags_are_read_back_off_a_trainer_command_line():
    """The guard reads the LAST value, because `run_leg_k.sh` states 64 first."""
    line = ("Command line: train.py --d-model 64 --num-layers 3 "
            "--num-encoder-layers 3 --seed 20260520 --d-model 384")
    got = study(f'printf %s {line!r} | cf412_arch_of_cmdline')
    assert got == "384 3 3"


def test_a_wrong_width_on_the_command_line_does_not_match_the_arm():
    arms = study("printf '%s\\n' $CF412_ARMS").split()
    want = study('cf412_arch_sig')
    line = "Command line: train.py --d-model 64 --num-layers 3 --num-encoder-layers 3"
    got = study(f'printf %s {line!r} | cf412_arch_of_cmdline')
    assert want == f"{CHOSEN_D_MODEL} {CHOSEN_LAYERS} {CHOSEN_ENCODER_LAYERS}"
    assert got != want


def test_the_head_and_the_evaluation_take_the_shape_from_one_file():
    """`bb_shape.sh` holds the shape both scripts build the backbone with."""
    assert BB_SHAPE.exists()
    default = bash(f'. "{BB_SHAPE}"; printf "%s " "${{BB_SHAPE[@]}}"')
    assert default.returncode == 0, default.stderr
    assert default.stdout.split() == ["--d-model", "64", "--n-heads", "8",
                                      "--num-layers", "3"]
    wide = bash(f'CF_BB_SHAPE="--d-model 384 --n-heads 8 --num-layers 3"; '
                f'export CF_BB_SHAPE; . "{BB_SHAPE}"; printf "%s " "${{BB_SHAPE[@]}}"')
    assert wide.stdout.split()[1] == str(CHOSEN_D_MODEL)
    for script in (HEAD_EVAL_BB, EVAL_LOCAL):
        text = script.read_text()
        assert "bb_shape.sh" in text, script
        assert '"${BB_SHAPE[@]}"' in text, script
        assert "--d-model 64" not in text, script


def test_the_head_wrapper_hands_the_new_width_to_both_scripts():
    arms = study("printf '%s\\n' $CF412_ARMS").split()
    proc = bash(f'CF412_DRY_RUN=1 bash "{HEAD_EVAL}" {arms[0]} 40000')
    assert proc.returncode == 0, proc.stderr
    assert f"--d-model {CHOSEN_D_MODEL}" in proc.stdout
    # `eval_local.sh` is a child of `head_eval_bb.sh`, so one exported value
    # reaches both.
    assert "CF_BB_SHAPE" in HEAD_EVAL.read_text()


# ---- 6. The scripts the card ships -------------------------------------------


@pytest.mark.parametrize("path", [STUDY_SH, ARMS_TSV, RUN_ARM, HEAD_EVAL,
                                  SIZE_TABLE, SMOKE, TRIAL, PHASE1, COLLECT,
                                  RUN_SH])
def test_the_script_is_present(path):
    assert path.exists(), path


@pytest.mark.parametrize("path", [RUN_ARM, HEAD_EVAL, SIZE_TABLE, SMOKE, TRIAL,
                                  PHASE1, COLLECT, RUN_SH])
def test_the_script_parses(path):
    proc = bash(f'bash -n "{path}"')
    assert proc.returncode == 0, proc.stderr


def test_a_trial_writes_nowhere_the_study_writes():
    plain = study("printf '%s %s' $CF412_ROOT $CF412_RESULTS")
    trial = study("printf '%s %s' $CF412_ROOT $CF412_RESULTS",
                  env={"CF412_TRIAL": "60"})
    assert plain != trial
    assert trial.split()[0].endswith("-trial")
    assert trial.split()[1].endswith("/trial")


# ---- 7. The note the card asks to correct ------------------------------------


def test_the_domain_note_names_the_backbone_of_each_row():
    """`~21M` is the April backbone, and the note must say so and give today's."""
    note = (REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "notes"
            / "DOMAIN_COMPARISON.md").read_text()
    assert "19,952,384" in note
    assert "720,668" in note
    assert re.search(r"Ours \(Contrastive Tiny, v2[^)]*\)", note), note
