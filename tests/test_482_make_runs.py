"""#482: the run scripts of the paper ablations
(reports/2026-11-27_paper_ablations/scripts/make_runs.py).

Each run script must be the reference run script (scripts/reference/, the
box_train script of the reference run of #425) with only the flags of its
row, its run name, its paths and its header. The tests write the 15 run
folders in a temp folder, compare each script with the reference line by
line, and run the dry run of each.
"""
import importlib.util
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "reports" / "2026-11-27_paper_ablations" / "scripts"
CODE = "code_0123abcd"


def load():
    spec = importlib.util.spec_from_file_location("make_runs_482",
                                                  SCRIPTS / "make_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mr = load()
ROWS = {row[0]: row for row in mr.ROWS}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("box_train")
    mr.main(["--code", CODE, "--out", str(out)])
    return out


def args_of(text):
    """The flags of the ARGS=( ... ) block, as (flag, value or None) pairs."""
    lines = text.split("\n")
    start = lines.index("ARGS=(") + 1
    pairs = []
    for line in lines[start:lines.index(")", start)]:
        words = line.split()
        pairs.append((words[0], words[1] if len(words) > 1 else None))
    return pairs


def with_edits(pairs, edits):
    """The reference flags with the edits of a row, computed here again."""
    out = list(pairs)
    for edit in edits:
        flags = [f for f, _ in out]
        if edit[0] == "set":
            out[flags.index(edit[1])] = (edit[1], edit[2])
        elif edit[0] == "drop":
            del out[flags.index(edit[1])]
        else:
            out.append((edit[1], edit[2]))
    return out


def code_lines(text):
    """The lines that run: no comment, and the ARGS block left out."""
    lines = [line for line in text.split("\n") if not line.startswith("#")]
    start = lines.index("ARGS=(")
    return lines[:start] + lines[lines.index(")", start):]


def run_script(runs, row):
    name = mr.run_name(row[0], row[1])
    return runs / name / f"run_{name}.sh"


def dry_run(script, tmp_path, code):
    env = dict(os.environ, ABC_GIFT_DRY_RUN="1", BB_GPU="0", R=str(tmp_path),
               WT=code)
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                         env=env, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    words = shlex.split(out.stdout.strip().splitlines()[-1])
    cut = words.index("python3")
    return words[:cut], words[cut + 3:]


def test_there_is_one_run_for_each_row_of_the_table():
    # Rows 1 to 15 of the paper table, and row 16: an extra row of the user.
    assert sorted(ROWS) == list(range(1, 17))
    names = [mr.run_name(r[0], r[1]) for r in mr.ROWS]
    assert len(set(names)) == 16
    assert all(n.startswith("g166b_w320e6f6_ab") for n in names)


def test_the_second_reference_changes_the_seed_only(tmp_path):
    folder = mr.write_row(mr.SEED2, CODE, tmp_path)
    text = (folder / f"run_{folder.name}.sh").read_text()
    ref = mr.REF_RUN.read_text()
    assert folder.name == "g166b_w320e6f6_ab00_ref_seed2"
    assert args_of(text) == with_edits(args_of(ref), [("set", "--seed", "20260521")])
    assert args_of(ref) != args_of(text)


@pytest.mark.parametrize("row", sorted(ROWS))
def test_the_flags_are_the_reference_flags_with_the_row_edits(runs, row):
    ref = mr.REF_RUN.read_text()
    text = run_script(runs, ROWS[row]).read_text()
    assert args_of(text) == with_edits(args_of(ref), ROWS[row][5])
    assert args_of(text) != args_of(ref)


@pytest.mark.parametrize("row", sorted(ROWS))
def test_only_the_name_and_the_paths_change_out_of_the_flags(runs, row):
    name = mr.run_name(row, ROWS[row][1])
    ref = code_lines(mr.REF_RUN.read_text())
    new = code_lines(run_script(runs, ROWS[row]).read_text())
    changed = [(a, b) for a, b in zip(ref, new) if a != b]
    assert len(ref) == len(new)
    assert changed == [
        (f'R="${{R:-/workspace/hp64/{mr.REF}}}"',
         f'R="${{R:-/workspace/hp64/{name}}}"'),
        ('WT="${WT:-/workspace/hp64/code_038ea26c}"',
         f'WT="${{WT:-/workspace/hp64/{CODE}}}"'),
        (f'NAME="{mr.REF}"', f'NAME="{name}"')]


@pytest.mark.parametrize("row", sorted(ROWS))
def test_the_dry_run_gives_the_reference_command_with_the_row_edits(
        runs, row, tmp_path):
    ref_env, ref_argv = dry_run(mr.REF_RUN, tmp_path / "ref", "/code")
    env, argv = dry_run(run_script(runs, ROWS[row]), tmp_path / "row", "/code")
    assert env == ref_env
    name = mr.run_name(row, ROWS[row][1])
    ref_argv = [w.replace(mr.REF, name).replace(str(tmp_path / "ref"),
                                                 str(tmp_path / "row"))
                for w in ref_argv]
    pairs = lambda a: args_of("ARGS=(\n" + "\n".join(
        " ".join(shlex.quote(w) for w in group) for group in groups(a)) + "\n)")
    want = with_edits(pairs(ref_argv), [
        (e[0], e[1], e[2].replace('"$TOTAL"', "166000") if len(e) > 2
         and e[2] else (e[2] if len(e) > 2 else None)) for e in ROWS[row][5]])
    assert pairs(argv) == want


def groups(argv):
    """The command split into one group per flag."""
    out = []
    for word in argv:
        if word.startswith("--"):
            out.append([word])
        else:
            out[-1].append(word)
    return out


@pytest.mark.parametrize("row", sorted(ROWS))
def test_the_supervisor_is_the_reference_supervisor(runs, row):
    name = mr.run_name(row, ROWS[row][1])
    text = (runs / name / f"supervise_{name}.sh").read_text()
    assert text == mr.REF_SUPERVISE.read_text().replace(mr.REF, name)
    assert (runs / name / "ckpt").is_dir() and (runs / name / "logs").is_dir()


def test_the_generator_does_not_write_over_a_run_folder(runs):
    with pytest.raises(FileExistsError):
        mr.main(["--code", CODE, "--out", str(runs), "--rows", "1"])
