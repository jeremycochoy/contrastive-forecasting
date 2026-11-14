"""Writes the 1-sqrt cooldown of one finished width-64 run for elisa (the user, 11-12 about 09:55 UTC: "You can run
the square root cooldown according to the reference research paper [...] redo it for each individual cooldown so
that you can compare the exponential and the square root schedule").

The cooldown resumes from the run's save at START with its optimizer state (the same file as its exponential
cooldown of make_cooldown.py), and anneals the rate with the 1-sqrt shape of Hägele and others 2024 ("Scaling Laws and
Compute-Optimal Training Beyond Fixed Training Durations"): lr = 1e-3 x (1 - sqrt((step - START) / (166000 - START))),
0 at 166k. Every other flag is the run's own flag. Its
folder hp64/<run>_cdsqrt<start> gets the run script and the supervisor of the run (elisa paths, the new name), the
line of the cooldown flags, and init/ with the save and its optimizer file.

Usage: python3 make_cooldown_sqrt.py <run name> <start k> <elisa code folder>
"""
import re
import shutil
import sys
from pathlib import Path

from make_cooldown import HP, init_save

TOTAL = 166000


def cooldown_flags(start):
    return ["  --lr-final 0", "  --lr-decay-shape sqrt", f"  --lr-decay-start {start * 1000}"]


def run_script(text, run, new, code, init_name, start):
    """The run's script with the new name and folder, the new code, the cooldown flags and the init save."""
    text = re.sub(r'^R="\$\{R:-.*\}"$', f'R="${{R:-{HP}/{new}}}"', text, count=1, flags=re.M)
    text = re.sub(r'^WT="\$\{WT:-.*\}"$', f'WT="${{WT:-{code}}}"', text, count=1, flags=re.M)
    text = text.replace(f'NAME="{run}"', f'NAME="{new}"', 1)
    text = re.sub(r"Code: \S+/code_\w+\.", f"Code: {code}. Cooldown: make_cooldown_sqrt.py.",
                  text)
    text = text.replace('TOTAL="${TOTAL:-166000}"\n',
                        'TOTAL="${TOTAL:-166000}"\n'
                        f'INIT="$R/init/{init_name}"  # the save the cooldown starts from\n', 1)
    head, sep, tail = text.partition("ARGS=(\n")
    body, close, rest = tail.partition("\n)\n")
    text = head + sep + body + "\n" + "\n".join(cooldown_flags(start)) + close + rest
    text = text.replace('RESUME="$(last_checkpoint)"\n',
                        'RESUME="$(last_checkpoint)"\n[ -n "$RESUME" ] || RESUME="$INIT"\n', 1)
    why = (f"# {new}: the 1-sqrt cooldown (Hägele et al. 2024) of {run} from its {start}k save: "
           f"lr = 1e-3 x (1 - sqrt((step - {start * 1000}) / {TOTAL - start * 1000})), 0 at 166k.\n")
    return text.replace("#!/bin/bash\n", "#!/bin/bash\n" + why, 1)


def check(text, new, code, start):
    for needle in (f'NAME="{new}"', f"{HP}/{new}", f'WT="${{WT:-{code}}}"', "--lr-decay-shape sqrt",
                   f"--lr-decay-start {start * 1000}", '[ -n "$RESUME" ] || RESUME="$INIT"', "INIT=",
                   'TOTAL="${TOTAL:-166000}"'):
        assert needle in text, needle
    assert "--lr-half-life" not in text and "/workspace" not in text, "a flag or a path of another cooldown"


def main(run, start, code):
    start, new = int(start), f"{run}_cdsqrt{start}"
    assert Path(code, "experiments/2026-04-27_freq-embedding/scripts/train.py").is_file(), code
    folder = HP / new
    (folder / "init").mkdir(parents=True)
    (folder / "logs").mkdir()
    (folder / "ckpt").mkdir()
    save, optimizer = init_save(run, start)
    for path in (save, optimizer):
        shutil.copy2(path, folder / "init" / path.name)
    text = run_script((HP / run / f"run_{run}.sh").read_text(), run, new, code, save.name, start)
    check(text, new, code, start)
    (folder / f"run_{new}.sh").write_text(text)
    (folder / f"supervise_{new}.sh").write_text((HP / run / f"supervise_{run}.sh").read_text().replace(run, new))
    print("wrote", folder, "init", save.name)


if __name__ == "__main__":
    main(*sys.argv[1:])
