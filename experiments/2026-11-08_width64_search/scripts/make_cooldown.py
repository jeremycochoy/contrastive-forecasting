"""Writes the cooldown of one finished width-64 run for the rented box (the user, 11-11: "take a checkpoint around
20/100 of 166 total step [...] and do an exponential decay training to zero going toward the end [...] at 166 [...]
for all the colors [...] exponential decay exactly like in the Mini CPM paper").

The cooldown resumes from the run's save at START with its optimizer state, and anneals the rate with the MiniCPM
decay (arXiv 2404.06395, section 6.2): lr = 1e-3 x 0.5^((step - START) / 5000), so 1.4e-5 at 166k from 135k.
Every other flag is the run's own flag. Its folder gets the run script and the supervisor of the run (paths of the
box, the new name), the line of the cooldown flags, and init/ with the save and its optimizer file.

Usage: python3 make_cooldown.py <run name> <start k> <box code folder> <output folder>
"""
import os
import re
import shutil
import sys
from pathlib import Path

HP = Path(os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64"))
BOX = "/workspace/hp64"
HALF_LIFE = 5000


def cooldown_flags(start):
    return [f"  --lr-final 0", f"  --lr-decay-shape exponential", f"  --lr-decay-start {start * 1000}",
            f"  --lr-half-life {HALF_LIFE}"]


def init_save(run, start):
    """The save of the run at START (the file of the last start) and its optimizer file."""
    saves = sorted((HP / run / "ckpt").glob(f"{run}*_{start}k.pth"), key=lambda p: p.name)
    saves = [p for p in saves if not p.name.endswith("_optimizer.pth")]
    assert saves, f"no {start}k save of {run}"
    save = saves[-1]
    optimizer = save.with_name(save.stem + "_optimizer.pth")
    assert optimizer.is_file(), optimizer
    return save, optimizer


def run_script(text, run, new, code, init_name, start):
    """The run's script with the paths of the box, the new name, the cooldown flags and the init save."""
    text = re.sub(r'^R="\$\{R:-.*\}"$', f'R="${{R:-{BOX}/{new}}}"', text, count=1, flags=re.M)
    text = re.sub(r'^WT="\$\{WT:-.*\}"$', f'WT="${{WT:-{code}}}"', text, count=1, flags=re.M)
    text = text.replace(f'NAME="{run}"', f'NAME="{new}"', 1)
    text = re.sub(r"Code: \S+/code_\w+\.", f"Code: {code}. Cooldown: make_cooldown.py.", text)
    text = text.replace('TOTAL="${TOTAL:-166000}"\n',
                        'TOTAL="${TOTAL:-166000}"\n'
                        f'INIT="$R/init/{init_name}"  # the save the cooldown starts from\n', 1)
    head, sep, tail = text.partition("ARGS=(\n")
    body, close, rest = tail.partition("\n)\n")
    text = head + sep + body + "\n" + "\n".join(cooldown_flags(start)) + close + rest
    text = text.replace('RESUME="$(last_checkpoint)"\n',
                        'RESUME="$(last_checkpoint)"\n[ -n "$RESUME" ] || RESUME="$INIT"\n', 1)
    why = (f"# {new}: the cooldown of {run} from its {start}k save: the MiniCPM decay, "
           f"lr = 1e-3 x 0.5^((step - {start * 1000}) / {HALF_LIFE}), to 166k.\n")
    return text.replace("#!/bin/bash\n", "#!/bin/bash\n" + why, 1)


def check(text, new, code, start):
    for needle in (f'NAME="{new}"', f"{BOX}/{new}", code, f"--lr-decay-start {start * 1000}",
                   f"--lr-half-life {HALF_LIFE}", '[ -n "$RESUME" ] || RESUME="$INIT"', "INIT="):
        assert needle in text, needle
    assert "/home/jupyter" not in text.split("ARGS=(")[1], "an elisa path in the box script"


def main(run, start, code, out):
    start, new = int(start), f"{run}_cd{start}"
    folder = Path(out) / new
    (folder / "init").mkdir(parents=True)
    (folder / "logs").mkdir()
    (folder / "ckpt").mkdir()
    save, optimizer = init_save(run, start)
    for path in (save, optimizer):
        shutil.copy2(path, folder / "init" / path.name)
    text = run_script((HP / run / f"run_{run}.sh").read_text(), run, new, code, save.name, start)
    check(text, new, code, start)
    (folder / f"run_{new}.sh").write_text(text)
    sup = (HP / run / f"supervise_{run}.sh").read_text().replace(run, new).replace(str(HP), BOX)
    (folder / f"supervise_{new}.sh").write_text(sup)
    print("wrote", folder, "init", save.name)


if __name__ == "__main__":
    main(*sys.argv[1:])
