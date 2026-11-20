"""The copy of each loss term at each rollout depth, in the losses CSV.

The loss holds one copy of each f-bearing term for each rollout depth j
(#373): L_align, L_pred of the split shape, and the joint teacher fraction.
Before this change the losses CSV logged the depth-0 copy only (``l_align``,
``l_pred``, ``l_joint``). Now ``term_out`` receives the copy of each depth j
as ``l_align_d<j>``, ``l_pred_d<j>`` and ``l_joint_d<j>``, and a run with
--train-rollout-depth writes them as the last columns of the CSV.

1. The loss functions: the copies, weighted as --train-rollout-reduce
   weighs them, give the total of the term. A depth that
   --train-rollout-distances leaves out gets no key.
2. train.py: the columns of the terms that a run builds, L_align on two
   patch sizes, and the logger on a new CSV and on a resume.
3. Short runs of the real trainer on the CPU: a new run writes the columns,
   and a run of the commit before them resumes under its own header.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.loss import (JOINT_TEACHER_SHAPE, align_loss,
                      contrastive_latent_loss, joint_teacher_term, pred_term,
                      reduce_depth_terms)
from tests.test_412_ours_moirai import git_tree, run

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
# The last commit before the per-depth term columns.
COMMIT_BEFORE = "59720ed4"

SPLIT = "cosine_similarity_batch_split_pred_rep"
REP_ONLY = "cosine_similarity_batch_rep_only"
REDUCES = ("sum", "mean", "halving", "harmonic")
DEPTH = 3
# --train-rollout-distances 1,2,4 keeps the depths 0, 1 and 3.
KEPT = (0, 1, 3)


def names(term, depths):
    return [f"{term}_d{j}" for j in depths]


def spec(shape, **extra):
    config = {"contrastive_divergence_temperature": 0.1,
              "contrastive_latent_noise": None, "loss_shape": shape,
              "contrastive_latent_delay": 0}
    config.update(extra)
    return SimpleNamespace(train_configuration=config)


def latents(seed, T=8, B=3, C=2, H=6):
    """``(f, h, teacher, rollout)`` in float64, with DEPTH rollout copies."""
    g = torch.Generator().manual_seed(seed)

    def draw():
        return torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    return draw(), draw(), draw(), [draw() for _ in range(DEPTH)]


def padding(f):
    """Two padded patches at the start of row 0."""
    pad = torch.zeros(f.shape[:3], dtype=torch.bool)
    pad[0, :2] = True
    return pad


def from_copies(terms, term, reduce, weight=1.0, depths=range(DEPTH + 1)):
    """The total of a term from its logged copies, as the loss reduces
    them."""
    return reduce_depth_terms([weight * terms[name]
                               for name in names(term, depths)], reduce)


# ---------------------------------------------------------------------------
# 1. The loss functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("teacher_target", [False, True])
def test_the_align_copies_give_the_total(reduce, padded, teacher_target):
    f, h, teacher, rollout = latents(seed=1)
    terms = {}
    total = align_loss(f, h, 2.0,
                       target_latent=teacher if teacher_target else None,
                       rollout_latents=rollout, depth_reduce=reduce,
                       pad_patches=padding(f) if padded else None,
                       term_out=terms)
    assert sorted(terms) == ["l_align", *names("l_align", range(DEPTH + 1))]
    assert terms["l_align"] == terms["l_align_d0"]
    assert total.item() == pytest.approx(
        from_copies(terms, "l_align", reduce, 2.0), rel=1e-12)


@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("shape", [REP_ONLY, SPLIT])
def test_the_copies_in_the_main_loss_give_the_total(reduce, shape):
    """L_rep enters once. The copy of each depth holds L_align, and L_pred
    on the split shape."""
    f, h, _, rollout = latents(seed=2)
    s = spec(shape, align_loss_weight=1.5, pred_loss_weight=0.7,
             train_rollout_depth=DEPTH, train_rollout_reduce=reduce)
    terms = {}
    total = contrastive_latent_loss((f, h), False, s, rollout_latents=rollout,
                                    rep_loss_weight=0.5, term_out=terms)
    f_terms = ["l_align"] + (["l_pred"] if shape == SPLIT else [])
    assert sorted(terms) == sorted(
        ["l_rep", *f_terms,
         *(name for t in f_terms for name in names(t, range(DEPTH + 1)))])
    for t in f_terms:
        assert terms[t] == terms[f"{t}_d0"]
    copies = [1.5 * terms[f"l_align_d{j}"]
              + 0.7 * terms.get(f"l_pred_d{j}", 0.0) for j in range(DEPTH + 1)]
    want = 0.5 * terms["l_rep"] + reduce_depth_terms(copies, reduce)
    assert total.item() == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("padded", [False, True])
def test_the_joint_copies_give_the_total(reduce, padded):
    f, h, teacher, rollout = latents(seed=3)
    terms = {}
    total = joint_teacher_term(
        f, h, spec(JOINT_TEACHER_SHAPE), padding(f) if padded else None,
        teacher_original_latent=teacher, rollout_latents=rollout,
        depth_reduce=reduce, term_out=terms)
    assert sorted(terms) == ["l_joint", *names("l_joint", range(DEPTH + 1))]
    assert terms["l_joint"] == terms["l_joint_d0"]
    assert total.item() == pytest.approx(
        from_copies(terms, "l_joint", reduce), rel=1e-12)


@pytest.mark.parametrize("reduce", REDUCES)
def test_the_pred_copies_on_a_padded_batch_give_the_total(reduce):
    f, h, teacher, rollout = latents(seed=4)
    terms = {}
    total = pred_term(f, h, spec(SPLIT, pred_loss_weight=0.7), padding(f),
                      teacher_original_latent=teacher,
                      rollout_latents=rollout, depth_reduce=reduce,
                      term_out=terms)
    assert sorted(terms) == ["l_pred", *names("l_pred", range(DEPTH + 1))]
    assert terms["l_pred"] == terms["l_pred_d0"]
    assert total.item() == pytest.approx(
        from_copies(terms, "l_pred", reduce, 0.7), rel=1e-12)


@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_a_left_out_depth_gets_no_key(reduce):
    f, h, teacher, rollout = latents(seed=5)
    rollout = [None if j not in KEPT else r
               for j, r in enumerate(rollout, start=1)]
    main = spec(REP_ONLY, align_loss_weight=1.0, rep_loss_weight=0.0,
                train_rollout_depth=DEPTH, train_rollout_reduce=reduce)
    calls = (
        ("l_align", lambda t: align_loss(
            f, h, 1.0, rollout_latents=rollout, depth_reduce=reduce,
            term_out=t)),
        ("l_align", lambda t: contrastive_latent_loss(
            (f, h), False, main, rollout_latents=rollout, term_out=t)),
        ("l_joint", lambda t: joint_teacher_term(
            f, h, spec(JOINT_TEACHER_SHAPE), None,
            teacher_original_latent=teacher, rollout_latents=rollout,
            depth_reduce=reduce, term_out=t)),
        ("l_pred", lambda t: pred_term(
            f, h, spec(SPLIT), None, rollout_latents=rollout,
            depth_reduce=reduce, term_out=t)))
    for term, call in calls:
        terms = {}
        total = call(terms)
        assert sorted(terms) == [term, *names(term, KEPT)]
        assert total.item() == pytest.approx(
            from_copies(terms, term, reduce, depths=KEPT), rel=1e-12)


def test_the_keys_change_no_loss_and_no_gradient():
    """The copies reach ``term_out`` as floats, off the graph: the loss and
    its gradients are the same bits with and without the dict."""
    f, h, teacher, rollout = latents(seed=6)
    leaves = [f, *rollout]
    for leaf in leaves:
        leaf.requires_grad_(True)
    main = spec(SPLIT, align_loss_weight=1.5, train_rollout_depth=DEPTH)
    calls = (
        lambda t: contrastive_latent_loss((f, h), False, main,
                                          rollout_latents=rollout, term_out=t),
        lambda t: align_loss(f, h, 1.5, rollout_latents=rollout,
                             pad_patches=padding(f), term_out=t),
        lambda t: joint_teacher_term(
            f, h, spec(JOINT_TEACHER_SHAPE), padding(f),
            teacher_original_latent=teacher, rollout_latents=rollout,
            term_out=t))
    for call in calls:
        bare, logged = call(None), call({})
        assert torch.equal(bare, logged)
        for a, b in zip(torch.autograd.grad(bare, leaves),
                        torch.autograd.grad(logged, leaves)):
            assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 2. train.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train():
    spec_ = importlib.util.spec_from_file_location("train_depth_terms",
                                                   TRAIN_PY)
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    return module


def parse(train, *flags):
    """The flags of a run, with the one flag that train.py requires."""
    return train.parse_args(["--weight-decay", "0.1", *flags])


@pytest.mark.parametrize("flags,want", [
    (["--align-loss-weight", "1.0"], names("l_align", range(4))),
    (["--loss-shape", SPLIT, "--align-loss-weight", "1.0"],
     names("l_pred", range(4)) + names("l_align", range(4))),
    (["--loss-shape", SPLIT], names("l_pred", range(4))),
    (["--loss-shape", SPLIT, "--pred-loss-weight", "0",
      "--align-loss-weight", "1.0"], names("l_align", range(4))),
    (["--loss-shape", SPLIT, "--no-main-contrastive-loss",
      "--align-loss-weight", "1.0"], names("l_align", range(4))),
    (["--loss-shape", JOINT_TEACHER_SHAPE, "--align-loss-weight", "1.0"],
     names("l_align", range(4)) + names("l_joint", range(4))),
])
def test_the_columns_of_the_terms_a_run_builds(train, flags, want):
    args = parse(train, "--train-rollout-depth", "3", *flags)
    assert train.term_depth_columns(args) == want


def test_a_depth_0_run_writes_no_term_depth_column(train):
    args = parse(train, "--loss-shape", SPLIT, "--align-loss-weight", "1.0")
    assert train.term_depth_columns(args) == []


def test_the_listed_distances_name_the_columns(train):
    args = parse(train, "--train-rollout-distances", "1,2,4",
                 "--align-loss-weight", "1.0")
    train.resolve_rollout_distances(args)
    assert train.term_depth_columns(args) == names("l_align", KEPT)


@pytest.mark.parametrize("reduce", REDUCES)
def test_l_align_on_two_patch_sizes_gives_the_total(train, reduce):
    """Each group adds its copies by its share of the rows (#412). Each
    reduction is linear, so the added copies give the total."""
    lats, pads = {}, {}
    for size, steps, seed in ((8, 8, 7), (16, 4, 8)):
        f, h, _, rollout = latents(seed=seed, T=steps)
        lats[size] = SimpleNamespace(f=f, o=h, teacher=None,
                                     rollout=rollout[:2])
        pads[size] = padding(f) if size == 8 else None
    args = SimpleNamespace(align_loss_weight=2.0, align_target="student",
                           train_rollout_reduce=reduce,
                           align_loss_kind="cosine")
    values = train.no_contrastive_values()
    total = train.group_align_terms(lats, pads, {8: 3, 16: 2}, args, values)
    terms = values["terms"]
    assert sorted(terms) == ["l_align", *names("l_align", range(3))]
    assert terms["l_align"] == terms["l_align_d0"]
    assert total.item() == pytest.approx(
        from_copies(terms, "l_align", reduce, 2.0, range(3)), rel=1e-12)


TERM_COLUMNS = names("l_align", range(3))


def log_row(logger, terms):
    logger.log(1, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 7, 8, False,
               0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, terms=terms)
    logger.close()


def csv_rows(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


def test_a_new_csv_ends_with_the_term_depth_columns(train, tmp_path):
    path = str(tmp_path / "r_losses.csv")
    log_row(train.CSVLogger(path, rollout_depth=2,
                            term_depth_columns=TERM_COLUMNS),
            {"l_align_d0": 0.25, "l_align_d1": 0.5})
    header, row = csv_rows(path)
    assert header[-4:] == ["cos_err_d2", *TERM_COLUMNS]
    assert row[-3:] == ["0.25", "0.5", ""]


def test_a_resume_keeps_a_header_without_the_columns(train, tmp_path):
    """A CSV from before the columns: the resume writes none of them."""
    path = str(tmp_path / "r_losses.csv")
    log_row(train.CSVLogger(path, rollout_depth=2), {})
    log_row(train.CSVLogger(path, rollout_depth=2,
                            term_depth_columns=TERM_COLUMNS),
            {"l_align_d0": 0.25})
    header, first, second = csv_rows(path)
    assert header[-1] == "cos_err_d2"
    assert len(first) == len(second) == len(header)


def test_a_resume_keeps_a_header_with_the_columns(train, tmp_path):
    path = str(tmp_path / "r_losses.csv")
    for value in (0.25, 0.75):
        log_row(train.CSVLogger(path, rollout_depth=2,
                                term_depth_columns=TERM_COLUMNS),
                {"l_align_d0": value})
    header, first, second = csv_rows(path)
    assert header[-3:] == TERM_COLUMNS
    assert [first[-3], second[-3]] == ["0.25", "0.75"]


@pytest.mark.parametrize("columns", [[], names("l_pred", range(3))
                                     + TERM_COLUMNS])
def test_a_resume_with_other_term_columns_is_refused(train, tmp_path,
                                                     columns):
    """Rows of other columns would shift every value of the CSV."""
    path = str(tmp_path / "r_losses.csv")
    log_row(train.CSVLogger(path, rollout_depth=2,
                            term_depth_columns=TERM_COLUMNS), {})
    with pytest.raises(SystemExit, match="schema mismatch"):
        train.CSVLogger(path, rollout_depth=2, term_depth_columns=columns)


# ---------------------------------------------------------------------------
# 3. The real trainer on the CPU
# ---------------------------------------------------------------------------

TINY = ("--total-steps", "3", "--batch-size", "2", "--lr", "1e-3",
        "--weight-decay", "0.1", "--mix-ratio", "1.0",
        "--synth-kind", "periodic", "--t-raw", "128", "--n-channels", "1",
        "--d-model", "16", "--n-heads", "2", "--num-layers", "1",
        "--num-encoder-layers", "1", "--log-every", "1",
        "--save-every", "1000", "--seed", "42", "--tau", "0.10",
        "--hf-repo", "none", "--hf-path", "none",
        "--train-rollout-depth", "2")
TEACHER = ("--ema-embedding", "--ema-encoder", "--ema-tau", "0.9",
           "--align-target", "teacher")
# The best cell of the project (#409): L_rep, and L_align on the teacher.
REP_ONLY_CELL = ("--loss-shape", REP_ONLY, "--align-loss-weight", "1.0",
                 "--moco-rep-keys", "--tau-rep", "1.0", *TEACHER)
SPLIT_CELL = ("--loss-shape", SPLIT, "--align-loss-weight", "1.0")
JOINT_CELL = ("--loss-shape", JOINT_TEACHER_SHAPE, "--align-loss-weight",
              "1.0", *TEACHER)


def losses_csv(save_dir):
    with open(save_dir / "r_losses.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def loss_from_columns(row, terms):
    """The loss of the cells below from the columns: L_rep once at its
    weight, and every copy at weight 1 (--train-rollout-reduce sum)."""
    rep = float(row["rep_w"]) * float(row["l_rep"]) if row["l_rep"] else 0.0
    return rep + sum(float(row[f"{t}_d{j}"]) for t in terms for j in range(3))


@pytest.mark.parametrize("cell,terms", [
    (REP_ONLY_CELL, ("l_align",)),
    (SPLIT_CELL, ("l_pred", "l_align")),
    (JOINT_CELL, ("l_align", "l_joint")),
])
def test_a_new_run_writes_a_column_for_each_copy(tmp_path, cell, terms):
    r = run(REPO_ROOT, tmp_path / "run", *TINY, *cell)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    header, rows = losses_csv(tmp_path / "run")
    want = [name for t in terms for name in names(t, range(3))]
    assert header[-len(want):] == want
    assert len(rows) == 3
    for row in rows:
        assert all(math.isfinite(float(row[name])) for name in want)
        assert all(row[t] == row[f"{t}_d0"] for t in terms)
        assert float(row["loss"]) == pytest.approx(
            loss_from_columns(row, terms), rel=1e-5)


def test_a_csv_from_before_the_columns_resumes_under_its_header(tmp_path):
    """The commit before the columns trains two steps. Its CSV then takes
    two more steps from its checkpoint: once on that commit and once on this
    one. This commit adds no column, and it writes the same bytes."""
    before = git_tree(COMMIT_BEFORE, tmp_path)
    flags = (*TINY, *REP_ONLY_CELL, "--save-every", "2")
    first = run(before, tmp_path / "first", *flags, "--total-steps", "2")
    assert first.returncode == 0, first.stderr[-2000:]
    resume = ("--total-steps", "4", "--resume",
              str(tmp_path / "first" / "r_0k.pth"))
    for leg, root in (("old", before), ("new", REPO_ROOT)):
        (tmp_path / leg).mkdir()
        shutil.copy(tmp_path / "first" / "r_losses.csv", tmp_path / leg)
        r = run(root, tmp_path / leg, *flags, *resume)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    header, rows = losses_csv(tmp_path / "new")
    assert header == losses_csv(tmp_path / "first")[0]
    assert [row["step"] for row in rows] == ["1", "2", "3", "4"]
    assert ((tmp_path / "new" / "r_losses.csv").read_bytes()
            == (tmp_path / "old" / "r_losses.csv").read_bytes())
