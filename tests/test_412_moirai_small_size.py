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


def test_every_arm_targets_the_teacher():
    """`L_align` targets the EMA teacher on every arm of this card."""
    assert study("printf '%s' $CF412_CELL") == "arm6_v2_combab_alignT"
    assert "arm6_v2_combab_alignT)" in RUN_LEG.read_text()


def test_the_repeat_seed_moves_the_seed_column_alone():
    """The repeat measures this size's own seed band, so it changes one column."""
    first, repeat = "k3_r100_09", "k3_r100_09b"
    for reader in ("cf412_depth", "cf412_reduce", "cf412_ema_sig",
                   "cf412_decay_ramp"):
        assert study(f'{reader} {repeat}') == study(f'{reader} {first}'), reader
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


# ---- 8. The two decay arms ---------------------------------------------------
#
# The card lists SIX configurations. Configurations 4 and 5 decay the `L_rep`
# weight from 1.0 to 0.0 by step 2,000. Configuration 5 minus configuration 2
# measures the decay alone at 11.4M parameters, so the pair must differ in the
# decay column and in nothing else.

CONFIGS = {
    1: "k3_r100_09",
    2: "k32_r100_09",
    3: "k32_r200_08",
    4: "k3_r100_09_dec",
    5: "k32_r100_09_dec",
    6: "k8_r100_09",
}
REPEAT_SEED_ARM = "k3_r100_09b"
DECAY_RAMP = "2000"


def test_the_card_has_its_six_configurations_and_the_repeat_seed():
    arms = study("printf '%s\\n' $CF412_ARMS").split()
    assert arms == [CONFIGS[n] for n in (1, 2, 3, 4, 5, 6)] + [REPEAT_SEED_ARM]


@pytest.mark.parametrize("config,k,reduce,ema,decay", [
    (1, "3", "sum", "0.9 1.0 100000", "-"),
    (2, "32", "mean", "0.9 1.0 100000", "-"),
    (3, "32", "mean", "0.8 1.0 200000", "-"),
    (4, "3", "sum", "0.9 1.0 100000", DECAY_RAMP),
    (5, "32", "mean", "0.9 1.0 100000", DECAY_RAMP),
    (6, "8", "mean", "0.9 1.0 100000", "-"),
])
def test_each_configuration_of_the_card_is_a_row(config, k, reduce, ema, decay):
    arm = CONFIGS[config]
    assert study(f'cf412_depth {arm}') == k
    assert study(f'cf412_reduce {arm}') == reduce
    assert study(f'cf412_ema_sig {arm}') == ema
    assert study(f'cf412_decay_ramp {arm}') == decay


@pytest.mark.parametrize("decay_arm,plain_arm", [(4, 1), (5, 2)])
def test_a_decay_twin_moves_the_decay_column_alone(decay_arm, plain_arm):
    """Configuration 5 minus configuration 2 measures the decay, and nothing else."""
    twin, plain = CONFIGS[decay_arm], CONFIGS[plain_arm]
    for reader in ("cf412_depth", "cf412_reduce", "cf412_ema_sig", "cf412_seed"):
        assert study(f'{reader} {twin}') == study(f'{reader} {plain}'), reader
    assert study(f'cf412_decay_ramp {twin}') != study(f'cf412_decay_ramp {plain}')


def test_the_decay_flags_are_one_unit():
    """A decay arm carries three flags. An arm without a decay carries none."""
    assert study(f'cf412_decay_args {CONFIGS[5]}') == (
        "--rep-loss-weight 1.0 --rep-loss-weight-end 0.0 "
        f"--rep-loss-weight-ramp-steps {DECAY_RAMP}")
    assert study(f'cf412_decay_args {CONFIGS[2]}') == ""


def test_the_trainer_takes_the_decay_flags():
    """#409 added them. Without that merge no decay arm can run."""
    train = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding" / "scripts"
             / "train.py").read_text()
    assert "--rep-loss-weight-end" in train
    assert "--rep-loss-weight-ramp-steps" in train


def test_a_dry_run_of_a_decay_arm_names_the_decay():
    proc = bash(f'CF412_DRY_RUN=1 bash "{RUN_ARM}" {CONFIGS[5]} 40000')
    assert proc.returncode == 0, proc.stderr
    gap = [ln for ln in proc.stdout.splitlines() if "gap=" in ln]
    assert len(gap) == 1
    assert f"--rep-loss-weight-end 0.0 --rep-loss-weight-ramp-steps {DECAY_RAMP}" \
        in gap[0], proc.stdout


def test_the_decay_is_read_back_off_a_trainer_command_line():
    line = ("python3 train.py --rep-loss-weight 1.0 --rep-loss-weight-end 0.0 "
            f"--rep-loss-weight-ramp-steps {DECAY_RAMP}")
    assert study(f"printf '%s' '{line}' | cf412_decay_of_cmdline") == \
        f"1.0 0.0 {DECAY_RAMP}"
    assert study(f'cf412_decay_sig {CONFIGS[5]}') == f"1.0 0.0 {DECAY_RAMP}"
    # An arm with no decay must read back the trainer's own default, so a stray
    # decay flag on its command line is a mismatch and not a match.
    assert study("printf '%s' 'python3 train.py --lr 1e-3' "
                 "| cf412_decay_of_cmdline") == "1.0 - -"
    assert study(f'cf412_decay_sig {CONFIGS[2]}') == "1.0 - -"


def test_a_trial_scales_the_decay_ramp_into_its_budget():
    """A 400-step smoke must still cross the whole decay, or it proves nothing."""
    full = study(f'cf412_ramp {CONFIGS[5]}')
    trial = study(f'cf412_ramp {CONFIGS[5]}', env={"CF412_TRIAL": "400"})
    assert full == DECAY_RAMP
    assert 1 <= int(trial) < int(full)


# ---- 9. The contrastive AUC is watched, and collected -------------------------
#
# The card asks for the AUC of every run and names a threshold of 0.55 on a
# rolling median over 500 rows, after a 1,000-step warm-up. A lost arm must not
# burn 32 GPU-hours in silence.

AUC_WATCH = REPO_ROOT / "scripts" / "auc_watch.py"
AUC_GUARD = EXP / "scripts" / "auc_guard.sh"

LOSS_HEADER = ("step,loss,auc,rep_w,l_pred,l_rep,l_align,ema_tau,"
               "cos_err_d0,cos_err_d1\n")


def loss_rows(rows) -> str:
    """The data rows of a `<run>_losses.csv`, one for each `(step, auc)`."""
    return "".join(f"{step},0.5,{auc},0.0,0.1,0.2,0.3,0.95,0.4,0.5\n"
                   for step, auc in rows)


def losses_csv(path: Path, rows, header: str = LOSS_HEADER):
    """Write a `<run>_losses.csv` whose `auc` column holds `rows`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + loss_rows(rows))
    return path


def gate_run(tmp_path, arm, auc_of_step, hold: int) -> subprocess.CompletedProcess:
    """Start a leg that fills its losses CSV, and put the gate on it.

    The trainer opens the CSV, then flushes rows into it as it trains. The gate
    counts the rows on disk BEFORE the leg writes one, so a test that lays the
    whole file down first would have it skip every row. This grows the file the
    way a leg does.
    """
    stop = "40000"
    env = {"CF412_ROOT": str(tmp_path / "root"),
           "CF412_RESULTS": str(tmp_path / "results"), "CF412_AUC_POLL": "1"}
    leg = Path(study(f'cf412_leg_dir {arm} {stop}', env=env))
    csv = leg / f"{study(f'cf412_run_name {arm}', env=env)}_losses.csv"
    losses_csv(csv, [])
    rows = tmp_path / "rows.csv"
    rows.write_text(loss_rows((s, auc_of_step(s)) for s in range(1, 3001)))
    proc = bash(f'( sleep 1; cat "{rows}" >>"{csv}"; sleep {hold} ) & pid=$!\n'
                f'bash "{AUC_GUARD}" {arm} {stop} "$pid"; rc=$?\n'
                f'kill "$pid" 2>/dev/null; exit $rc', env=env)
    return proc


def test_the_auc_watch_is_a_shared_script_of_the_main_codebase():
    """Two cards read it now, so it is not one card's file."""
    assert AUC_WATCH.is_file()
    assert not (EXP / "scripts" / "auc_watch.py").exists()


def test_the_watch_calls_a_healthy_run_held_and_a_collapsed_run_lost(tmp_path):
    held = losses_csv(tmp_path / "held_losses.csv",
                      [(s, 0.96) for s in range(1, 3001)])
    lost = losses_csv(tmp_path / "lost_losses.csv",
                      [(s, 0.96 if s < 1500 else 0.50) for s in range(1, 3001)])
    ok = bash(f'python3 "{AUC_WATCH}" "{held}" --warmup 1000')
    assert ok.returncode == 0, ok.stdout + ok.stderr
    bad = bash(f'python3 "{AUC_WATCH}" "{lost}" --warmup 1000')
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "lost" in bad.stdout


def test_the_card_reads_the_threshold_the_window_and_the_warmup_the_card_names():
    assert study("printf '%s' $CF412_AUC_WINDOW") == "500"
    assert study("printf '%s' $CF412_AUC_THRESHOLD") == "0.55"
    assert study("printf '%s' $CF412_AUC_WARMUP") == "1000"


def test_the_gate_stops_a_leg_that_lost_the_task(tmp_path):
    """The gate reads the live CSV, kills the leg and writes the collapse note."""
    arm = CONFIGS[5]
    proc = gate_run(tmp_path, arm, lambda s: 0.96 if s < 1500 else 0.50, hold=120)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    note = tmp_path / "results" / f"collapsed_{arm}.txt"
    assert note.is_file(), proc.stdout + proc.stderr
    assert "lost" in note.read_text()


def test_the_gate_lets_a_healthy_leg_run(tmp_path):
    """A run that holds 0.96 keeps its card until it finishes on its own."""
    arm = CONFIGS[2]
    proc = gate_run(tmp_path, arm, lambda s: 0.96, hold=3)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (tmp_path / "results" / f"collapsed_{arm}.txt").exists()


def test_every_leg_starts_the_gate():
    text = RUN_ARM.read_text()
    assert "auc_guard.sh" in text
    assert "CF412_RC_COLLAPSED" in text


def test_collect_writes_the_auc_and_the_loss_by_term(tmp_path):
    """Both are card deliverables, beside the score."""
    arm, stop = CONFIGS[2], "40000"
    root, results = tmp_path / "root", tmp_path / "results"
    env = {"CF412_ROOT": str(root), "CF412_RESULTS": str(results)}
    leg = Path(study(f'cf412_leg_dir {arm} {stop}', env=env))
    name = study(f'cf412_run_name {arm}', env=env)
    losses_csv(leg / f"{name}_losses.csv", [(s, 0.96) for s in range(1, 3001)])
    results.mkdir(parents=True, exist_ok=True)
    (results / f"score_{study(f'cf412_tag {arm} {stop} 30000', env=env)}.txt"
     ).write_text("1.1491\n")
    proc = bash(f'bash "{COLLECT}"', env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    verdicts = (results / "auc_verdicts.tsv").read_text()
    assert name in verdicts and "held" in verdicts
    terms = (results / "loss_terms.csv").read_text()
    assert terms.splitlines()[0].split(",")[:2] == ["arm", "stop"]
    for column in ("l_pred", "l_rep", "l_align", "rep_w", "auc"):
        assert column in terms.splitlines()[0], terms
    assert arm in terms
    assert "1.1491" in (results / "scores.csv").read_text()


# ---- 10. The shape guard cannot pass without a check --------------------------
#
# An empty command line skipped the whole comparison, and a timeout wrote a
# WARNING and let the leg climb. Both are the align-target trap the card names.


def stub_runner(path: Path, body: str) -> Path:
    """A `run_leg_k.sh` that does what one line of the test says, and exits."""
    path.write_text("#!/bin/bash\n" + body + "\n")
    return path


def guard_env(tmp_path, runner: Path, extra: dict | None = None) -> dict:
    env = {"CF412_ROOT": str(tmp_path / "root"),
           "CF412_RESULTS": str(tmp_path / "results"),
           "CF412_RUNNER": str(runner), "CF412_CHECK_TIMEOUT": "5",
           "CF412_AUC_WATCH": "0"}
    env.update(extra or {})
    return env


def test_a_leg_that_never_names_its_shape_stops(tmp_path):
    """A timeout is not a pass. The leg is stopped, not left to climb."""
    runner = stub_runner(tmp_path / "runner.sh", "sleep 120")
    proc = bash(f'bash "{RUN_ARM}" {CONFIGS[1]} 40000',
                env=guard_env(tmp_path, runner))
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "unchecked" in (proc.stdout + proc.stderr)


def test_a_leg_that_trained_without_naming_its_shape_stops(tmp_path):
    """A clean exit with no command line and no checkpoint is not a pass."""
    runner = stub_runner(tmp_path / "runner.sh", "exit 0")
    proc = bash(f'bash "{RUN_ARM}" {CONFIGS[1]} 40000',
                env=guard_env(tmp_path, runner))
    assert proc.returncode == 3, proc.stdout + proc.stderr


def test_a_stop_already_on_disk_is_not_a_failure(tmp_path):
    """`run_leg_k.sh` exits 0 without a trainer, and that is the idempotent path."""
    env = guard_env(tmp_path, stub_runner(tmp_path / "runner.sh", "exit 0"))
    leg = Path(study(f'cf412_leg_dir {CONFIGS[1]} 40000', env=env))
    leg.mkdir(parents=True, exist_ok=True)
    (leg / f"{study(f'cf412_run_name {CONFIGS[1]}', env=env)}_40k.pth").touch()
    proc = bash(f'bash "{RUN_ARM}" {CONFIGS[1]} 40000', env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_a_refusal_of_the_runner_keeps_its_own_exit_code(tmp_path):
    """9 is a session hold and 10 is another machine's cell. Neither is a defect."""
    runner = stub_runner(tmp_path / "runner.sh", "exit 10")
    proc = bash(f'bash "{RUN_ARM}" {CONFIGS[1]} 40000',
                env=guard_env(tmp_path, runner))
    assert proc.returncode == 10, proc.stdout + proc.stderr


def test_the_leg_stops_when_the_command_line_names_another_objective(tmp_path):
    """The align-target trap: the guard reads the trainer's own line, and aborts."""
    env = guard_env(tmp_path, tmp_path / "runner.sh")
    log = Path(study(f'cf412_leg_log {CONFIGS[1]}', env=env))
    stub_runner(tmp_path / "runner.sh",
                f'mkdir -p "{log.parent}"\n'
                f'echo "Command line: python3 train.py --d-model 64 '
                f'--num-layers 3 --num-encoder-layers 3 --align-target student" '
                f'>>"{log}"\nsleep 120')
    proc = bash(f'bash "{RUN_ARM}" {CONFIGS[1]} 40000', env=env)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    # The message wraps, so read the values and not the sentence around them.
    assert "'student'" in proc.stderr and "'64 3 3'" in proc.stderr, proc.stderr


# ---- 11. The cost numbers ----------------------------------------------------


def test_the_smoke_samples_the_card_faster_than_an_arm_runs():
    """A 5-second sampler over a 25-second arm can miss the peak."""
    assert study("printf '%s' $CF412_SMOKE_POLL") == "1"
    assert int(study("printf '%s' $CF412_SMOKE_STEPS")) >= 150


def test_the_committed_smoke_agrees_with_the_claim():
    """The evidence a reader sees must be the evidence the PR body describes."""
    log = (EXP / "results" / "trial" / "smoke.log").read_text()
    assert "smoke done — 0 failure(s)" in log, log
    assert " rc=" not in log, log
    rows = (EXP / "results" / "trial" / "smoke.csv").read_text().splitlines()
    arms = study("printf '%s\\n' $CF412_ARMS").split()
    assert [r.split(",")[0] for r in rows[1:]] == arms


def test_a_smoke_measures_and_never_inherits(tmp_path):
    """A second smoke must retrain, or its cost table is the table before it.

    The runner is idempotent, so a re-run of a smoke that kept its checkpoints
    would train nothing and report a memory of 0 beside the step time of the
    run before it.
    """
    arm = CONFIGS[1]
    env = {"CF412_ROOT": str(tmp_path / "root"),
           "CF412_RESULTS": str(tmp_path / "results"),
           "CF412_RUNNER": str(stub_runner(tmp_path / "runner.sh", "exit 0")),
           "CF412_CHECK_TIMEOUT": "5", "CF412_AUC_WATCH": "0"}
    leg = Path(study(f'cf412_leg_dir {arm} 150', env={**env, "CF412_TRIAL": "150"}))
    leg.mkdir(parents=True, exist_ok=True)
    stale = leg / "stale.pth"
    stale.touch()
    bash(f'ARMS={arm} bash "{SMOKE}" 150', env=env)
    assert not stale.exists(), "the smoke kept the leg of the run before it"


def test_a_smoke_writes_only_under_a_trial_root():
    """The removal above is `rm -rf`, so its path must never be a study root."""
    assert study("printf '%s' $CF412_ROOT", env={"CF412_TRIAL": "150"}) \
        .endswith("-trial")
    text = SMOKE.read_text()
    fence = text.index("*-trial) rm -rf")
    assert "ABORT: a smoke must write under a trial root" in text[fence:]
