"""Writes the joint-teacher copy of green (the user, 11-18: the forecast and the representation terms in one InfoNCE
fraction per rollout depth, both in the numerator and the denominator, every h key from the teacher, one temperature
"to 1.0". See PR #470).

The copy is green's run (10-L-ALIGN, width 64) with --loss-shape cosine_similarity_batch_joint_teacher in place of
L_align and L_rep: --align-loss-weight 0.0, --tau <tau>, and no --moco-rep-keys, --tau-rep, --rep-loss-weight or
--align-target. Every other flag of green stays.

Usage: python3 make_joint.py <run folder of the new run> <code folder of the new run> <out dir> [tau]
  writes <out dir>/<NEW>/ with run_<NEW>.sh, supervise_<NEW>.sh, ckpt/ and logs/. The tau (default 1.0) names the
  run: tau 1.0 gives ..._joint_tau1_rollfix (the first run, 11-18), tau 0.3 gives ..._joint_tau03_rollfix (the user,
  11-20: "the same but with maybe 0.3 in temperature").
"""
import os
import sys
from pathlib import Path

HP = Path(os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64"))
GREEN = "g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix"
TAU = sys.argv[4] if len(sys.argv) > 4 else "1.0"
NEW = "g166b_lr1em3_tau99995_bs256_sigfix_ss_sphere_joint_tau" + TAU.replace("1.0", "1").replace(".", "") + "_rollfix"
CHANGES = [("  --loss-shape cosine_similarity_batch_rep_only\n", "  --loss-shape cosine_similarity_batch_joint_teacher\n"),
           ("  --align-loss-weight 10.0\n", "  --align-loss-weight 0.0\n"),
           ("  --moco-rep-keys\n", ""), ("  --tau-rep 1.0\n", ""), ("  --tau 0.10\n", f"  --tau {TAU}\n"),
           ("  --rep-loss-weight 0.1\n", ""),
           ("  --align-target teacher\n", "")]  # train.py refuses --align-target with no L_align (smoke test 11-18)


def changed(text, rdir, code):
    """Green's run script with the new name, folder, code and loss flags."""
    for old, new in [(f'R="${{R:-{HP / GREEN}}}"', f'R="${{R:-{rdir}}}"'),
                     (f'WT="${{WT:-{HP}/code_038ea26c}}"', f'WT="${{WT:-{code}}}"'),
                     (f'NAME="{GREEN}"', f'NAME="{NEW}"'), *CHANGES]:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    why = (f"# {NEW}: green ({GREEN}) with the joint teacher fraction in place of L_align and L_rep, tau {TAU} "
           f"(make_joint.py; cf PR #470).\n")
    return text.replace("#!/bin/bash\n", "#!/bin/bash\n" + why, 1)


def check(text, green_text):
    """Only the lines of CHANGES, R, WT, NAME and the header differ from green."""
    for needle in ("  --loss-shape cosine_similarity_batch_joint_teacher\n", f"  --tau {TAU}\n",
                   "  --align-loss-weight 0.0\n", "  --train-rollout-depth 3\n", 'TOTAL="${TOTAL:-166000}"'):
        assert needle in text, needle
    for gone in ("--moco-rep-keys", "--tau-rep", "--rep-loss-weight", "  --tau 0.10\n", "--align-target"):
        assert gone not in text, gone
    new_lines, old_lines = set(text.splitlines()), set(green_text.splitlines())
    assert len(new_lines - old_lines) == 7, sorted(new_lines - old_lines)  # header, R, WT, NAME, 3 flag lines


def main(rdir, code, out):
    folder = Path(out) / NEW
    (folder / "ckpt").mkdir(parents=True)
    (folder / "logs").mkdir()
    green_text = (HP / GREEN / f"run_{GREEN}.sh").read_text()
    text = changed(green_text, rdir, code)
    check(text, green_text)
    (folder / f"run_{NEW}.sh").write_text(text)
    sup = (HP / GREEN / f"supervise_{GREEN}.sh").read_text().replace(GREEN, NEW).replace(str(HP / NEW), rdir)
    (folder / f"supervise_{NEW}.sh").write_text(sup.replace(str(HP), str(Path(rdir).parent)))
    print("wrote", folder)


if __name__ == "__main__":
    main(*sys.argv[1:4])
