"""Write the run folders of the paper ablations (#482, paper/sections/ablation.tex).

Each run is the reference run (dark green: width 320, 6 encoder and 6
forecaster layers, warmup 10k then a constant 1e-3, 166k steps) with the one
change of its row. The run script of a row is the reference run script
(scripts/reference/, a copy of the box_train script of #425) with these
changes only:
  - the flags of the row (ROWS)
  - the run name (NAME) and the run folder (R)
  - the code folder (WT): the run commit, which is 038ea26c (the code of the
    reference) plus the new flags of #482
  - the header comment.
The supervisor is the reference supervisor with the new run name.

Each run folder gets run_<name>.sh, supervise_<name>.sh, logs/ and ckpt/, in
<out>/<name>/. The box runs it at /workspace/hp64/<name>, as the reference.

Usage:
    python3 make_runs.py --code code_<sha> [--out <folder>] [--rows 1,4]
"""
import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = "g166b_w320e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k"
REF_RUN = HERE / "reference" / f"run_{REF}.sh"
REF_SUPERVISE = HERE / "reference" / f"supervise_{REF}.sh"
BOX = "/workspace/hp64"
OUT = "/home/jupyter/cf_runs/cf-482/box_train"
PREFIX = "g166b_w320e6f6_ab"

# (row, short name, part, our recipe, replaced by, edits). An edit is one of:
#   ("set", flag, value): a new value for a flag of the reference.
#   ("drop", flag): the flag leaves the command.
#   ("add", flag, value or None): a new flag at the end of the command.
ROWS = [
    (1, "no_synth", "Data", "Forked ARMA series", "No synthetic series",
     [("set", "--mix-ratio", "0")]),
    (2, "no_crossfade", "Data", "Crossfade series", "No crossfade series",
     [("set", "--crossfade-triplets", "0")]),
    (3, "no_mixup", "Data", "Mixup, p = 0.3", "No mixup",
     [("set", "--mixup-p", "0")]),
    (4, "instnorm", "Input", "EWMA normalisation",
     "Instance normalisation with no target leak",
     [("set", "--rev-norm-kind", "meanstd"), ("add", "--meanstd-no-leak", None),
      ("add", "--meanstd-z-max", "0")]),
    (5, "no_timescale", "Input", "Time-scale embedding", "No time-scale embedding",
     [("add", "--zero-time-scale-embedding", None)]),
    (6, "timesfm_patch", "Input", "GRU patch embedding", "Residual block of TimesFM",
     [("set", "--encoder-type", "timesfm")]),
    (7, "no_sphere", "Model", "Latent sphere after each stage", "No sphere",
     [("drop", "--latent-sphere")]),
    (8, "no_conv", "Model", "Depthwise causal convolution", "No convolution",
     [("set", "--depthwise-conv", "0")]),
    (9, "no_dropkey", "Model", "DropKey in the encoder, p = 0.7", "No DropKey",
     [("set", "--encoder-dropkey", "0")]),
    (10, "no_dropout", "Model", "Dropout of 0.1 in the blocks", "No dropout",
     [("add", "--dropout", "0")]),
    (11, "no_ema", "Loss", "EMA teacher", "Student targets with stop-gradient",
     [("set", "--ema-tau", "0"), ("drop", "--ema-tau-end"),
      ("drop", "--ema-tau-ramp-steps")]),
    (12, "no_sigreg_e", "Loss", "SIGReg on e", "No SIGReg on e",
     [("drop", "--sigreg-embedding")]),
    (13, "no_sigreg_h", "Loss", "SIGReg on h", "No SIGReg on h",
     [("drop", "--sigreg-encoding")]),
    (14, "no_lrep", "Loss", "L_rep", "No L_rep",
     [("set", "--rep-loss-weight", "0")]),
    (15, "align_d1", "Loss", "L_align at distances 1 to 4", "Distance 1 only",
     [("set", "--train-rollout-depth", "0")]),
    # An extra row of the user (11-28): the changes of rows 11 and 14 together.
    (16, "no_ema_no_lrep", "Loss", "EMA teacher and L_rep", "No EMA teacher and no L_rep",
     [("set", "--ema-tau", "0"), ("drop", "--ema-tau-end"), ("drop", "--ema-tau-ramp-steps"),
      ("set", "--rep-loss-weight", "0")]),
    # An extra row of the user (11-28): no SIGReg at all (rows 12 and 13 together).
    (17, "no_sigreg", "Loss", "SIGReg on e and on h", "No SIGReg",
     [("drop", "--sigreg-embedding"), ("drop", "--sigreg-encoding")]),
]
# A second training of the reference (11-28, the user): only the seed changes.
# The two references give the noise of one training. It is row 0, out of ROWS.
SEED2 = (0, "ref_seed2", "Reference", "Reference recipe, seed 20260520",
         "Second training, seed 20260521", [("set", "--seed", "20260521")])


def run_name(row, short):
    return f"{PREFIX}{row:02d}_{short}"


def args_block(text):
    """(start, end) of the lines of the ARGS=( ... ) block of a run script."""
    lines = text.split("\n")
    start = lines.index("ARGS=(") + 1
    return lines, start, lines.index(")", start)


def apply_edits(arg_lines, edits):
    """The ARGS lines with the edits of a row. Each edit must find its flag."""
    out = list(arg_lines)
    for edit in edits:
        kind, flag = edit[0], edit[1]
        found = [i for i, line in enumerate(out) if line.split()[0] == flag]
        if kind == "add":
            assert not found, f"{flag} is already in the reference"
            out.append(f"  {flag}" + (f" {edit[2]}" if edit[2] is not None else ""))
            continue
        assert len(found) == 1, f"{flag} is not once in the reference"
        if kind == "drop":
            del out[found[0]]
        else:
            assert len(out[found[0]].split()) == 2, f"{flag} takes no value"
            out[found[0]] = f"  {flag} {edit[2]}"
    return out


def replace_once(text, pattern, new):
    """The text with the one line that matches the pattern replaced."""
    found = re.findall(pattern, text, flags=re.M)
    assert len(found) == 1, pattern
    return re.sub(pattern, lambda _: new, text, count=1, flags=re.M)


def header(name, row):
    _, _, part, ours, replaced, edits = row
    changes = ", ".join(" ".join(str(x) for x in e if x is not None) for e in edits)
    return (f"# {name}: the ablation row {row[0]} of #482 ({part}): {ours}, replaced by "
            f"{replaced}.\n"
            f"# The reference run {REF} with the flags of the row only: {changes}.\n"
            "# Written by reports/2026-11-27_paper_ablations/scripts/make_runs.py.")


def run_script(ref_text, row, code):
    """The run script of a row, from the text of the reference run script."""
    name = run_name(row[0], row[1])
    lines, start, end = args_block(ref_text)
    lines[start:end] = apply_edits(lines[start:end], row[5])
    # The history of the reference (its header) ends where the comment on the
    # resume starts. The rest of the script stays.
    first = next(i for i, line in enumerate(lines) if line.startswith("# A start with"))
    text = "\n".join([lines[0], header(name, row)] + lines[first:])
    text = replace_once(text, r'^R="\$\{R:-.*\}"$', f'R="${{R:-{BOX}/{name}}}"')
    text = replace_once(text, r'^WT="\$\{WT:-.*\}"$', f'WT="${{WT:-{BOX}/{code}}}"')
    return replace_once(text, rf'^NAME="{REF}"$', f'NAME="{name}"')


def supervise_script(ref_text, row):
    return ref_text.replace(REF, run_name(row[0], row[1]))


def write_row(row, code, out):
    """Write the run folder of a row. Refuses to write over a folder."""
    name = run_name(row[0], row[1])
    folder = Path(out) / name
    folder.mkdir(parents=True)
    for sub in ("logs", "ckpt"):
        (folder / sub).mkdir()
    (folder / f"run_{name}.sh").write_text(run_script(REF_RUN.read_text(), row, code))
    (folder / f"supervise_{name}.sh").write_text(
        supervise_script(REF_SUPERVISE.read_text(), row))
    return folder


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--code", required=True, help="the code folder, code_<sha>")
    p.add_argument("--out", default=OUT)
    p.add_argument("--rows", default=None,
                   help="comma-separated rows, 0 for SEED2; default all")
    a = p.parse_args(argv)
    keep = None if a.rows is None else {int(r) for r in a.rows.split(",")}
    for row in [SEED2] + ROWS:
        if keep is None or row[0] in keep:
            print("wrote", write_row(row, a.code, a.out))


if __name__ == "__main__":
    main()
