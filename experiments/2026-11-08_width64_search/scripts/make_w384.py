"""Writes a width-384 copy of one finished width-64 run (the user, 11-18 at about 07:20 UTC: "We would like to scale
the grey to 1.5 and 380 something in latent and the green the same and train for the full 1.66").

The copy has --d-model 384, 1 encoder layer and 5 forecaster layers (the user's "1.5". MPE has 3 and 3). It keeps
the muP base (64, 3, 3), so the rate 1e-3 transfers. The attention, the FFN and the conv use bf16 in place of fp16
(the brown NaN of 11-17 was an fp16 gradient overflow through the 31-step rollout). Every other flag of the run
stays. The copy trains 166k steps from step 0.

Usage: python3 make_w384.py <run name> <new name> <run folder of the new run> <code folder of the new run> <out dir>
  writes <out dir>/<new name>/ with run_<new>.sh, supervise_<new>.sh, ckpt/ and logs/
"""
import os
import re
import sys
from pathlib import Path

HP = Path(os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64"))
CHANGES = [("--d-model", "64", "384"), ("--num-layers", "3", "5"), ("--attn-dtype", "fp16", "bf16"),
           ("--ffn-dtype", "fp16", "bf16"), ("--conv-dtype", "fp16", "bf16")]


def changed(text, run, new, rdir, code):
    """The run's script with the new name, folder, code and the width-384 flags."""
    text = re.sub(r'^R="\$\{R:-.*\}"$', f'R="${{R:-{rdir}}}"', text, count=1, flags=re.M)
    text = re.sub(r'^WT="\$\{WT:-.*\}"$', f'WT="${{WT:-{code}}}"', text, count=1, flags=re.M)
    text = text.replace(f'NAME="{run}"', f'NAME="{new}"', 1)
    for flag, old, value in CHANGES:
        line = f"  {flag} {old}\n"
        assert text.count(line) == 1, line
        text = text.replace(line, f"  {flag} {value}\n")
    why = (f"# {new}: {run} at width 384 with 1 encoder and 5 forecaster layers (muP base 64, 3, 3), bf16 in place "
           f"of fp16, 166k steps from step 0 (make_w384.py).\n")
    return text.replace("#!/bin/bash\n", "#!/bin/bash\n" + why, 1)


def check(text, new, rdir, code):
    """The new script has the new name and paths, the width-384 flags and no fp16 flag."""
    for needle in (f'NAME="{new}"', rdir, code, "  --d-model 384\n", "  --num-layers 5\n",
                   "  --num-encoder-layers 1\n", "  --n-heads 8\n", "  --attn-dtype bf16\n", "  --ffn-dtype bf16\n",
                   "  --conv-dtype bf16\n", "  --mup-base-d-model 64\n", 'TOTAL="${TOTAL:-166000}"'):
        assert needle in text, needle
    assert "fp16" not in text.split("ARGS=(")[1].split(")")[0], "an fp16 flag left"


def main(run, new, rdir, code, out):
    folder = Path(out) / new
    (folder / "ckpt").mkdir(parents=True)
    (folder / "logs").mkdir()
    text = changed((HP / run / f"run_{run}.sh").read_text(), run, new, rdir, code)
    check(text, new, rdir, code)
    (folder / f"run_{new}.sh").write_text(text)
    sup = (HP / run / f"supervise_{run}.sh").read_text().replace(run, new).replace(str(HP / new), rdir)
    sup = sup.replace(str(HP), str(Path(rdir).parent))
    (folder / f"supervise_{new}.sh").write_text(sup)
    print("wrote", folder)


if __name__ == "__main__":
    main(*sys.argv[1:])
