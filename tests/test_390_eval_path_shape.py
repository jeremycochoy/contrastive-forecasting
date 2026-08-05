"""The eval path of #390, checked against #379's the way the launcher is.

`tests/test_390_launcher_shape.py` proves the *backbone* command line is
#379's plus one flag. Nothing proved the same for the *measurement* half,
and everything between a backbone checkpoint and a reported GM-Relative
MASE sits there: the head's seed, lr, batch size, forecast length, the
`--head-*` architecture, `--grad-clip`, and the GIFT-Eval invocation. One
silent difference moves all 30 #390 measurements together and none of the
20 copied from #379 — a bias no per-cell check can see.

`eval_arm.sh` only *claims* to be #379's `eval_2L_gm_mase.sh` "verbatim
except for three things". These tests check it, and they check the argv the
two scripts actually build rather than their source text: both are run with
a stub `python3` on PATH that records `"$@"`. Source-level comparison would
miss a difference introduced by a variable's default.

The three differences the claim allows, and nothing else:

  1. the checkout (`WT`),
  2. the experiment directory,
  3. arm → run-name resolution (`arm_names.sh` instead of an awk over
     #379's `run_arm.sh` case block).

All three are paths or the run name derived from one, so the tests
normalise exactly those and require token-for-token equality on the rest.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_390 = REPO_ROOT / "experiments" / "2026-08-01_lalign_teacher"
EXP_379 = REPO_ROOT / "experiments" / "2026-07-21_split_pred_rep_small"
EVAL_390 = EXP_390 / "scripts" / "eval_arm.sh"
EVAL_379 = EXP_379 / "scripts" / "eval_2L_gm_mase.sh"
RUN_ARM_379 = EXP_379 / "scripts" / "run_arm.sh"

ARM = "arm5"
BB_STEP_K = "40"
HEAD_STEPS = "15000"
N_CONFIGS = 97

# Tokens whose values are a path or the run name derived from one. These are
# the three allowed differences; everything else must match exactly.
PATH_VALUED_FLAGS = {
    "--backbone-path", "--save-dir", "--run-name", "--head-path",
    "--output-dir",
}

# A stub `python3`. Records argv, then fabricates whatever the calling
# script checks for so it proceeds to its next stage.
PY_STUB = r"""#!/bin/bash
n=$(ls "$ARGV_DIR" | wc -l)
printf '%s\n' "$@" > "$ARGV_DIR/argv.$n"

save_dir=""; run_name=""; out_dir=""; is_head=0; is_eval=0
prev=""
for a in "$@"; do
  case "$prev" in
    --save-dir)   save_dir="$a" ;;
    --run-name)   run_name="$a" ;;
    --output-dir) out_dir="$a" ;;
  esac
  case "$a" in
    --quantile-head) is_head=1 ;;
    --strategy)      is_eval=1 ;;
  esac
  prev="$a"
done

if [ "$is_head" = 1 ] && [ -n "$save_dir" ] && [ -n "$run_name" ]; then
  mkdir -p "$save_dir"; : > "$save_dir/${run_name}_final.pth"
fi
if [ "$is_eval" = 1 ] && [ -n "$out_dir" ]; then
  mkdir -p "$out_dir"
  { echo "dataset,config,MASE,seasonal_naive_MASE"
    for i in $(seq 1 __N_CONFIGS__); do echo "ds$i,cfg$i,1.0,1.0"; done
  } > "$out_dir/all_results.csv"
  echo "Aggregate GM-Relative MASE (__N_CONFIGS__ configs): 1.2345" \
    > "$out_dir/summary.txt"
fi
exit 0
"""


def read_argv(argv_dir: Path) -> list[list[str]]:
    """Recorded argv lists, in call order."""
    files = sorted(argv_dir.glob("argv.*"), key=lambda p: int(p.suffix[1:]))
    return [f.read_text().splitlines() for f in files]


def normalise(argv: list[str]) -> list[str]:
    """Replace the three allowed differences with placeholders.

    A path-valued flag's value becomes `<PATH>`, any remaining absolute path
    becomes `<PATH>` (that covers the trainer/eval script paths, which are
    passed positionally). Everything else — every flag, and every non-path
    value including `--seed 20260722` — is left alone.
    """
    out: list[str] = []
    prev = ""
    for tok in argv:
        if prev in PATH_VALUED_FLAGS or tok.startswith("/"):
            out.append("<PATH>")
        else:
            out.append(tok)
        prev = tok
    return out


def make_stub_dir(root: Path) -> Path:
    stub_dir = root / "bin"
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub = stub_dir / "python3"
    stub.write_text(PY_STUB.replace("__N_CONFIGS__", str(N_CONFIGS)))
    stub.chmod(0o755)
    return stub_dir


def gift_eval_dirs(wt: Path) -> None:
    """The two scripts the eval invokes. #390's eval_arm.sh stats them."""
    d = wt / "experiments" / "2026-04-13_gift-eval" / "scripts"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train_forecasting_head.py").write_text("")
    (d / "eval_gift_eval_official.py").write_text("")
    (wt / "experiments").mkdir(parents=True, exist_ok=True)
    (wt / "experiments" / "hf_token.txt").write_text("hf_stub_token\n")


@pytest.fixture(scope="module")
def scratch():
    """A scratch root outside /tmp — eval_arm.sh refuses a WT under it."""
    root = tempfile.mkdtemp(prefix="cf390-evalshape-",
                            dir=os.path.expanduser("~/.cache"))
    yield Path(root)
    shutil.rmtree(root, ignore_errors=True)


def bb_name_390(arm: str) -> str:
    return subprocess.run(
        ["bash", "-c",
         f'source "{EXP_390 / "scripts" / "arm_names.sh"}"; bb_name {arm}'],
        capture_output=True, text=True, check=True).stdout.strip()


def bb_name_379(arm: str) -> str:
    """#379 resolves the name by awk over its own run_arm.sh case block."""
    code = RUN_ARM_379.read_text()
    m = re.search(rf'(?m)^\s*{re.escape(arm)}\)\s*\n.*?NAME="([^"]+)"',
                  code, re.DOTALL)
    assert m is not None, f"no NAME for {arm} in #379's run_arm.sh"
    return m.group(1)


def run_390(scratch: Path) -> list[list[str]]:
    wt = scratch / "wt390"
    argv_dir = scratch / "argv390"
    if argv_dir.exists():
        return read_argv(argv_dir)
    argv_dir.mkdir(parents=True)
    gift_eval_dirs(wt)
    runs = wt / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{bb_name_390(ARM)}_{BB_STEP_K}k.pth").write_bytes(b"x")

    env = {**os.environ,
           "PATH": f"{make_stub_dir(scratch / 'stub390')}:{os.environ['PATH']}",
           "ARGV_DIR": str(argv_dir),
           "WT": str(wt), "ARM": ARM, "BB_GPU": "0",
           "BB_STEP_K": BB_STEP_K, "HEAD_STEPS": HEAD_STEPS}
    res = subprocess.run(["bash", str(EVAL_390)], env=env,
                         capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, (
        f"#390 eval_arm.sh failed under the stub:\n{res.stdout}\n{res.stderr}")
    return read_argv(argv_dir)


def run_379(scratch: Path) -> list[list[str]]:
    """#379's script, with its one hardcoded `WT=` line pointed at scratch.

    That line is the first of the three differences the claim allows, so
    rewriting it is exactly what makes the two comparable. Nothing else in
    the file is touched — asserted below.
    """
    wt = scratch / "wt379"
    argv_dir = scratch / "argv379"
    if argv_dir.exists():
        return read_argv(argv_dir)
    argv_dir.mkdir(parents=True)
    gift_eval_dirs(wt)
    exp = wt / "experiments" / "2026-07-21_split_pred_rep_small"
    (exp / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(RUN_ARM_379, exp / "scripts" / "run_arm.sh")
    (exp / "runs").mkdir(parents=True, exist_ok=True)
    (exp / "runs" / f"{bb_name_379(ARM)}_{BB_STEP_K}k.pth").write_bytes(b"x")

    src = EVAL_379.read_text().splitlines(keepends=True)
    patched, n_hits = [], 0
    for line in src:
        if line.startswith("WT="):
            patched.append(f"WT={wt}\n")
            n_hits += 1
        else:
            patched.append(line)
    assert n_hits == 1, (
        f"expected exactly one hardcoded WT= line in {EVAL_379.name}, "
        f"found {n_hits}")
    # The script loads `scripts/resolve_eval_checkpoint.sh` and
    # `scripts/eval_cell_identity.sh` three levels above itself, so the
    # scratch copy has to sit at the same depth under a root that carries
    # both. Dropping it anywhere else makes the run abort on a missing
    # dependency rather than compare the two command lines.
    repo = scratch / "repo379"
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    for dep in ("resolve_eval_checkpoint.sh", "eval_cell_identity.sh"):
        shutil.copy(REPO_ROOT / "scripts" / dep, repo / "scripts" / dep)
    script_dir = repo / "experiments" / "2026-07-21_split_pred_rep_small" / \
        "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    script = script_dir / "eval_2L_gm_mase_scratch.sh"
    script.write_text("".join(patched))

    env = {**os.environ,
           "PATH": f"{make_stub_dir(scratch / 'stub379')}:{os.environ['PATH']}",
           "ARGV_DIR": str(argv_dir),
           "ARM": ARM, "BB_GPU": "0",
           "BB_STEP_K": BB_STEP_K, "HEAD_STEPS": HEAD_STEPS,
           "GIFT_EVAL": str(scratch / "gift-eval-data")}
    res = subprocess.run(["bash", str(script)], env=env,
                         capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, (
        f"#379 eval script failed under the stub:\n{res.stdout}\n{res.stderr}")
    return read_argv(argv_dir)


@pytest.fixture(scope="module")
def argv_390(scratch):
    return run_390(scratch)


@pytest.fixture(scope="module")
def argv_379(scratch):
    return run_379(scratch)


def test_both_scripts_make_the_same_two_calls(argv_390, argv_379):
    """Head trainer then GIFT-Eval, once each. A third call, or a missing
    one, is a difference in the measurement path on its own."""
    assert len(argv_379) == 2, f"#379 made {len(argv_379)} python3 calls"
    assert len(argv_390) == 2, f"#390 made {len(argv_390)} python3 calls"
    for got in (argv_379, argv_390):
        assert "--quantile-head" in got[0]
        assert "--strategy" in got[1]


def test_head_trainer_argv_is_identical(argv_390, argv_379):
    """The head is what turns a backbone into a forecast. Seed, lr, batch
    size, forecast length, `--head-*` and `--grad-clip` all live here."""
    assert normalise(argv_390[0]) == normalise(argv_379[0]), (
        "#390's head-training command line is not #379's.\n"
        f"  #379: {' '.join(normalise(argv_379[0]))}\n"
        f"  #390: {' '.join(normalise(argv_390[0]))}")


def test_gift_eval_argv_is_identical(argv_390, argv_379):
    assert normalise(argv_390[1]) == normalise(argv_379[1]), (
        "#390's GIFT-Eval command line is not #379's.\n"
        f"  #379: {' '.join(normalise(argv_379[1]))}\n"
        f"  #390: {' '.join(normalise(argv_390[1]))}")


def flag_value(argv: list[str], flag: str) -> str:
    i = argv.index(flag)
    return argv[i + 1]


@pytest.mark.parametrize("flag,expected", [
    ("--seed", "20260722"),
    ("--lr", "1e-3"),
    ("--batch-size", "256"),
    ("--forecast-len", "16"),
    ("--grad-clip", "1.0"),
    ("--head-arch", "transformer"),
    ("--head-num-layers", "2"),
    ("--head-nhead", "8"),
    ("--head-ffn-mult", "4.0"),
    ("--head-causal", "true"),
    ("--head-train-input", "e_then_f"),
    ("--head-dropout", "0.1"),
    ("--total-steps", HEAD_STEPS),
])
def test_head_flag_value_matches_379(argv_390, argv_379, flag, expected):
    """Named one by one so a failure says which knob moved. The path
    normalisation above cannot reach these — none of them is a path."""
    assert flag_value(argv_379[0], flag) == expected, (
        f"#379's own {flag} is not {expected}; update this test, not the "
        "script.")
    assert flag_value(argv_390[0], flag) == expected


@pytest.mark.parametrize("flag,expected", [
    ("--strategy", "B4"),
    ("--forecast-len", "16"),
    ("--head-nhead", "8"),
    ("--head-causal", "true"),
])
def test_gift_eval_flag_value_matches_379(argv_390, argv_379, flag, expected):
    assert flag_value(argv_379[1], flag) == expected
    assert flag_value(argv_390[1], flag) == expected


def test_head_seed_default_is_379s_literal():
    """#390 parameterised the seed for review item 4's replicates. The
    default has to stay #379's literal, or all 30 measurements shift."""
    src = EVAL_390.read_text()
    assert 'HEAD_SEED="${HEAD_SEED:-20260722}"' in src
    assert "--seed 20260722" in EVAL_379.read_text()


def test_cell_tag_defaults_to_empty():
    """CELL_TAG gives a replicate its own directory. Non-empty by default
    would rename every wave cell and orphan the committed results."""
    assert 'CELL_TAG="${CELL_TAG:-}"' in EVAL_390.read_text()


def test_backbone_and_arch_flags_match(argv_390, argv_379):
    """The architecture the head is built against. A mismatch here loads a
    different model from the same checkpoint."""
    for call in (0, 1):
        for flag in ("--t-raw", "--n-channels", "--d-model", "--n-heads",
                     "--num-layers", "--encoder-type", "--rev-norm-kind",
                     "--rev-norm-span"):
            assert flag_value(argv_390[call], flag) == \
                   flag_value(argv_379[call], flag), \
                   f"call {call}: {flag} differs"


def test_390_requires_all_97_configs(argv_390):
    """#379's script writes whatever aggregate it finds; #390's refuses
    anything but 97 configs. That is a tightening, not a difference in the
    measurement, and the report depends on it."""
    src = EVAL_390.read_text()
    assert "N_CONFIGS_EXPECTED=97" in src
    assert "No partial reported" in src
