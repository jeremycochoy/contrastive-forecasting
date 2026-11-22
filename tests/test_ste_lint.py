"""Tests for the Semicolon rule of the ASD-STE100 checker in .vale/, and for the files that it checks."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

VALE_CONFIG = Path(__file__).resolve().parents[1] / ".vale" / ".vale.ini"
LINT_SCRIPT = VALE_CONFIG.with_name("ste-lint.sh")
SEMICOLON_CHECK = "ASD-STE100.Semicolon"

# Each text holds one semicolon that rule 8.1 does not permit.
FORBIDDEN = [
    "A word; the next word.",
    "A word;the next word.",
    "A note (see note 12); the next note.",
    "A shape [B, T]; the next shape.",
    "A set {a}; the next set.",
    "A value ``x``; the next value.",
    "A letter τ; the next letter.",
    "A **bold**; the next word.",
    "A line that stops after the mark (see note 12);",
]

# Each text names the mark itself, so it gives no alert.
NAMES_OF_THE_MARK = [
    "The semicolon (;) is not permitted.",
    'The mark ";" is not permitted.',
    "The mark ';' is not permitted.",
]


def semicolon_alerts(text, suffix):
    """Return the (line, column) of each Semicolon alert that Vale gives."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"fixture{suffix}"
        path.write_text(text, encoding="utf-8")
        result = subprocess.run(
            ["vale", "--config", str(VALE_CONFIG), "--output", "JSON", str(path)],
            capture_output=True, text=True)
    alerts = [a for file_alerts in json.loads(result.stdout).values() for a in file_alerts]
    return sorted((a["Line"], a["Span"][0]) for a in alerts if a["Check"] == SEMICOLON_CHECK)


@unittest.skipUnless(shutil.which("vale"), "vale is not installed")
class TestSemicolonRule(unittest.TestCase):

    # One alert for each forbidden semicolon, at the column of the mark.
    def test_markdown_alerts_each_forbidden_semicolon(self):
        # One paragraph for each text: text i is on line 2 * i + 1.
        text = "\n\n".join(FORBIDDEN + NAMES_OF_THE_MARK) + "\n"
        expected = [(2 * i + 1, t.index(";") + 1) for i, t in enumerate(FORBIDDEN)]
        self.assertEqual(semicolon_alerts(text, ".md"), expected)

    def test_python_comment_alerts_each_forbidden_semicolon(self):
        # One comment line for each text: text i is on line i + 1, after "# ".
        text = "".join(f"# {line}\n" for line in FORBIDDEN + NAMES_OF_THE_MARK)
        expected = [(i + 1, t.index(";") + 3) for i, t in enumerate(FORBIDDEN)]
        self.assertEqual(semicolon_alerts(text, ".py"), expected)


def changed_list(repo, names):
    """The files that `ste-lint.sh --changed HEAD~1 --list` gives in a new git repository `repo` with one commit that
    adds the Python files `names`."""
    def git(*args):
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
                       check=True, capture_output=True)
    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "base")
    for name in names:
        (repo / name).write_text("# A comment.\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "files")
    result = subprocess.run([str(LINT_SCRIPT), "--changed", "HEAD~1", "--list"], cwd=repo, capture_output=True,
                            text=True, check=True)
    return result.stdout.split()


@unittest.skipUnless(shutil.which("vale"), "vale is not installed")
class TestChangedFiles(unittest.TestCase):

    # The list of the changed files is larger than a pipe buffer (64 KiB): the check of each file must not stop the
    # write of that list.
    def test_each_changed_file_is_in_the_list(self):
        names = [f"module_{index:04d}_{'x' * 60}.py" for index in range(1500)]
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(sorted(changed_list(Path(tmp), names)), names)


if __name__ == "__main__":
    unittest.main()
