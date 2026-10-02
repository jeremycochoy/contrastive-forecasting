"""Tests for the Semicolon rule of the ASD-STE100 checker in .vale/."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

VALE_CONFIG = Path(__file__).resolve().parents[1] / ".vale" / ".vale.ini"
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


if __name__ == "__main__":
    unittest.main()
