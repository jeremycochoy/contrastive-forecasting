"""CPU checks of the ablation runs of #482, on a local corpus (no Hub read).

parity: the reference run script on the code of the reference (code_038ea26c)
    and on the run code, a few steps each, with the same seed. The two losses
    CSVs must be equal at each step and in each column.
rows: each row script on the run code, a few steps each. Each must end with
    code 0 and write its losses CSV and its last checkpoint.

A run takes the trainer command of the dry run of its script (ABC_GIFT_DRY_RUN=1,
the flags and the environment of the box), with only these changes: the CPU, a
small batch, a few steps, a log line at each step, a save folder in --out, the
anneal length of the box run (--lr-cosine-steps 166000, so the learning rate of
each step is the one of the box run), and the local corpus
(--gift-pretrain-root, --gift-pretrain-index). The corpus is
the corpus of tests/test_419_gift_pretrain.py (GiftEvalPretrain layout).

Usage:
    python3 cpu_check.py parity --ref-code <dir> --run-code <dir> --script <run script> --out <dir>
    python3 cpu_check.py rows --run-code <dir> --scripts <run script> ... --out <dir>
"""
import argparse
import csv
import importlib.util
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# The anneal of the box run spans --total-steps (166k). A short run keeps it.
CPU = {"--device": "cpu", "--log-every": "1", "--save-every": "1000000",
       "--lr-cosine-steps": "166000"}
THREADS = "8"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def local_corpus(folder):
    """(root, index file) of the corpus of the #419 tests, written once."""
    root, index_file = folder / "corpus", folder / "index.json.gz"
    if not index_file.exists():
        sys.path.insert(0, str(REPO))
        tests = load_module(REPO / "tests" / "test_419_gift_pretrain.py", "t419")
        root.mkdir(parents=True)
        _, index, _ = tests.build_corpus(root)
        tests.write_index(folder, index)
    return root, index_file


def dry_run(script, code, folder):
    """(environment, trainer path, flags) of the dry run of a run script."""
    env = dict(os.environ, ABC_GIFT_DRY_RUN="1", BB_GPU="none", WT=str(code),
               R=str(folder))
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                         env=env, check=True).stdout
    words = shlex.split(out.strip().splitlines()[-1])
    cut = words.index("python3")
    envs = dict(w.split("=", 1) for w in words[:cut])
    return envs, words[cut + 2], words[cut + 3:]


def with_flags(argv, values):
    """The flags with new values. A flag that is not there goes at the end."""
    out, seen = list(argv), set()
    for i, word in enumerate(out):
        if word in values and i + 1 < len(out) and not out[i + 1].startswith("--"):
            out[i + 1] = values[word]
            seen.add(word)
    for flag in values:
        if flag not in seen:
            out += [flag, values[flag]]
    return out


def run_on_cpu(script, code, out, name, steps, batch, corpus):
    """Run the trainer command of a script on the CPU. Returns (rc, save dir, log)."""
    save = out / name
    save.mkdir(parents=True, exist_ok=True)
    envs, trainer, argv = dry_run(script, code, save)
    root, index = corpus
    argv = with_flags(argv, dict(CPU, **{
        "--batch-size": str(batch), "--total-steps": str(steps),
        "--save-dir": str(save), "--gift-pretrain-root": str(root),
        "--gift-pretrain-index": str(index)}))
    envs.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=THREADS,
                MKL_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS)
    log = save / "train.log"
    start = time.time()
    with open(log, "w") as f:
        f.write(f"{shlex.join(['python3', '-u', trainer] + argv)}\n")
        f.flush()
        rc = subprocess.run(["python3", "-u", trainer] + argv, stdout=f,
                            stderr=subprocess.STDOUT, env=dict(os.environ, **envs),
                            timeout=7200).returncode
    print(f"{name}: rc {rc}, {time.time() - start:.0f} s, log {log}", flush=True)
    drop_extra_checkpoints(save, keep_final=(name != "ref_code" and name != "run_code"))
    return rc, save, log


def drop_extra_checkpoints(save, keep_final):
    """Delete the checkpoints of a CPU run but its last backbone (the head
    check reads it): each one has 93 MB, its optimizer file 122 MB."""
    for path in save.glob("*.pth"):
        if not (keep_final and path.name.endswith("_final.pth")):
            path.unlink()


def losses(save):
    """The rows of the losses CSV of a run folder."""
    path = next(save.glob("*_losses.csv"))
    with open(path) as f:
        return list(csv.DictReader(f))


def compare(rows_a, rows_b):
    """The columns that differ, as {column: (step, a, b)} at their first step."""
    assert len(rows_a) == len(rows_b) and rows_a, (len(rows_a), len(rows_b))
    assert list(rows_a[0]) == list(rows_b[0]), "the columns differ"
    diff = {}
    for a, b in zip(rows_a, rows_b):
        for col in a:
            if a[col] != b[col] and col not in diff:
                diff[col] = (a["step"], a[col], b[col])
    return diff


def parity(a):
    corpus = local_corpus(a.out)
    runs = {}
    for name, code in (("ref_code", a.ref_code), ("run_code", a.run_code)):
        rc, save, log = run_on_cpu(a.script, code, a.out, name, a.steps, a.batch_size, corpus)
        assert rc == 0, f"{name} failed, see {log}"
        runs[name] = losses(save)
    diff = compare(runs["ref_code"], runs["run_code"])
    rows = runs["ref_code"]
    print(f"parity: {len(rows)} steps x {len(rows[0])} columns, "
          f"{'EQUAL' if not diff else 'DIFFERENT: ' + str(diff)}")
    for row in rows:
        print(f"  step {row['step']}: loss {row['loss']}, l_rep {row['l_rep']}, "
              f"l_align {row['l_align']}, sigreg_e {row['sigreg_e']}, sigreg_h {row['sigreg_h']}")
    return 0 if not diff else 1


def rows(a):
    corpus = local_corpus(a.out)
    failed = []
    for script in a.scripts:
        name = Path(script).stem.removeprefix("run_")
        rc, save, log = run_on_cpu(Path(script), a.run_code, a.out, name, a.steps,
                                   a.batch_size, corpus)
        ok = rc == 0 and list(save.glob("*_losses.csv")) and list(save.glob("*_final.pth"))
        if not ok:
            failed.append(name)
            continue
        last = losses(save)[-1]
        print(f"  {name}: step {last['step']}, loss {last['loss']}, l_rep {last['l_rep']}, "
              f"l_align {last['l_align']}, sigreg_e {last['sigreg_e']}, sigreg_h {last['sigreg_h']}")
    print(f"rows: {len(a.scripts) - len(failed)} of {len(a.scripts)} ran; failed: {failed}")
    return 1 if failed else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("parity", "rows"):
        s = sub.add_parser(name)
        s.add_argument("--run-code", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--steps", type=int, default=3)
        s.add_argument("--batch-size", type=int, default=128)
    sub.choices["parity"].add_argument("--ref-code", type=Path, required=True)
    sub.choices["parity"].add_argument("--script", type=Path, required=True)
    sub.choices["rows"].add_argument("--scripts", nargs="+", required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    return parity(a) if a.cmd == "parity" else rows(a)


if __name__ == "__main__":
    sys.exit(main())
