"""tests/conftest.py deletes the tmp_path folder of a passed test at its end,
and keeps the folder of a failed test for the debug.

Runs pytest on two small tests in a subprocess, with a copy of the conftest.
"""
import shutil
import subprocess
import sys
from pathlib import Path

CONFTEST = Path(__file__).with_name("conftest.py")

TWO_TESTS = '''
def test_pass(tmp_path):
    (tmp_path / "save.bin").write_bytes(b"x" * 1000)


def test_fail(tmp_path):
    (tmp_path / "save.bin").write_bytes(b"x" * 1000)
    assert False
'''


def run_two_tests(tmp_path, conftest=True):
    suite = tmp_path / "suite"
    suite.mkdir()
    if conftest:
        shutil.copy(CONFTEST, suite / "conftest.py")
    (suite / "test_two.py").write_text(TWO_TESTS)
    base = tmp_path / "base"
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"--basetemp={base}", str(suite)],
        capture_output=True, text=True, cwd=suite, timeout=300)
    assert "1 failed, 1 passed" in run.stdout, run.stdout[-2000:]
    # pytest 6.2 also writes a symlink <test>current to the folder of each
    # test: only the real folders count.
    return {p.name for p in base.iterdir() if p.is_dir() and not p.is_symlink()}


def test_a_passed_test_loses_its_folder(tmp_path):
    names = run_two_tests(tmp_path)
    assert not any(name.startswith("test_pass") for name in names), names


def test_a_failed_test_keeps_its_folder(tmp_path):
    names = run_two_tests(tmp_path)
    assert any(name.startswith("test_fail") for name in names), names


def test_without_the_conftest_both_folders_stay(tmp_path):
    names = run_two_tests(tmp_path, conftest=False)
    assert any(name.startswith("test_pass") for name in names), names
