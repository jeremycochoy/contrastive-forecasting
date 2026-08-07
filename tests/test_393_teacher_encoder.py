"""Tests for #393: running the EMA teacher as the downstream encoder.

Head training and the official GIFT-Eval both strip ``teacher_*`` before
they load a backbone, so every downstream number the project has ever
produced came from the student encoder. #393 needs a second head per
checkpoint, trained and evaluated on the teacher.

The mechanism is weight promotion, not a second forward path: the teacher
is a same-shape EMA copy of ``input_to_latent`` and ``encoder_layers``, so
copying it into the student's slots makes the ordinary pipeline — head
training, latent rollout, every forecast strategy — run the teacher with
no other change. Two heads then differ only by the weights they read,
never by the kernel that ran them.

Covered here:

1. :func:`src.checkpoint.prepare_backbone_state_dict` promotes and strips.
2. A promoted checkpoint reproduces ``extract_teacher_encoder_latents``.
3. The forecaster stays the student's — the teacher has no forecaster.
4. The head's encoder source is recorded next to its checkpoint, and a
   mismatch at eval time is an error, not a silently wrong number.
5. Both downstream scripts expose ``--encoder-source`` and default to
   ``student``, so every pre-#393 command line is unchanged.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from src.checkpoint import (encoder_source_marker_path, has_teacher_weights,
                            load_backbone_from_checkpoint,
                            load_encoder_source, prepare_backbone_state_dict,
                            save_encoder_source)
from src.forecasting_head import (extract_encoder_latents,
                                  extract_teacher_encoder_latents,
                                  rollout_latent)
from src.models import ConfigurableModel

REPO_ROOT = Path(__file__).resolve().parent.parent
GIFT = REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
HEAD_TRAIN_PY = GIFT / "train_forecasting_head.py"
EVAL_PY = GIFT / "eval_gift_eval_official.py"

BASE_CFG = dict(C=1, H=16, W=8, nhead=2, num_layers=1, num_encoder_layers=2,
                encoder_type="gru", dropout=0.0, rev_norm_kind="ewma",
                rev_norm_span=8)


def teacher_model(**over):
    cfg = dict(BASE_CFG, ema_embedding=True, ema_encoder=True)
    cfg.update(over)
    return ConfigurableModel(**cfg).eval()


def move_the_student(model, delta=0.25):
    """Push the student away from its teacher so the two are separable."""
    with torch.no_grad():
        for p in model.transformer.input_to_latent.parameters():
            p.add_(delta)
        for layer in model.transformer.encoder_layers:
            for p in layer.parameters():
                p.add_(delta)


def probe(seed=7, B=2, T_raw=64):
    return torch.randn(B, T_raw, 1, generator=torch.Generator().manual_seed(seed))


# --- 1. prepare_backbone_state_dict ---------------------------------------


class TestPrepareBackboneStateDict:

    def test_student_source_is_the_pre_393_behaviour(self):
        """Default path: drop the pretraining-only keys, touch nothing else."""
        sd = teacher_model().state_dict()
        out = prepare_backbone_state_dict(sd)
        assert not any(k.startswith("teacher_") for k in out)
        assert not any(k.startswith("cpc_w1") for k in out)
        for k, v in out.items():
            assert torch.equal(v, sd[k]), k

    def test_teacher_source_promotes_the_patch_embedding(self):
        model = teacher_model()
        move_the_student(model)
        model.update_teacher(0.9)
        sd = model.state_dict()
        out = prepare_backbone_state_dict(sd, encoder_source="teacher")
        # `encoder.*` and `transformer.input_to_latent.*` are the same module
        # registered twice. Both must carry the teacher, or which one wins
        # depends on load_state_dict's key order.
        for k, v in sd.items():
            if not k.startswith("teacher_input_to_latent."):
                continue
            tail = k[len("teacher_input_to_latent."):]
            assert torch.equal(out[f"encoder.{tail}"], v), tail
            assert torch.equal(out[f"transformer.input_to_latent.{tail}"], v), tail

    def test_teacher_source_promotes_the_encoder_stack(self):
        model = teacher_model()
        move_the_student(model)
        model.update_teacher(0.9)
        sd = model.state_dict()
        out = prepare_backbone_state_dict(sd, encoder_source="teacher")
        for k, v in sd.items():
            if not k.startswith("teacher_encoder_layers."):
                continue
            tail = k[len("teacher_encoder_layers."):]
            assert torch.equal(out[f"transformer.encoder_layers.{tail}"], v), tail

    def test_promotion_actually_changes_the_weights(self):
        """Guard against a promotion that silently no-ops."""
        model = teacher_model()
        move_the_student(model)
        model.update_teacher(0.9)
        sd = model.state_dict()
        student = prepare_backbone_state_dict(sd)
        teacher = prepare_backbone_state_dict(sd, encoder_source="teacher")
        assert set(student) == set(teacher)
        assert any(not torch.equal(student[k], teacher[k]) for k in student)

    def test_teacher_source_strips_the_teacher_keys_too(self):
        """The model is rebuilt without ema flags, so the load stays strict."""
        sd = teacher_model().state_dict()
        out = prepare_backbone_state_dict(sd, encoder_source="teacher")
        assert not any(k.startswith("teacher_") for k in out)
        assert set(out) == set(prepare_backbone_state_dict(sd))

    def test_embedding_only_teacher_leaves_the_stack_alone(self):
        """--ema-embedding without --ema-encoder: only the patch embed moves,
        matching teacher_forward()'s fallback to the student's stack."""
        model = teacher_model(ema_encoder=False)
        move_the_student(model)
        model.update_teacher(0.9)
        sd = model.state_dict()
        out = prepare_backbone_state_dict(sd, encoder_source="teacher")
        for k, v in sd.items():
            if k.startswith("transformer.encoder_layers."):
                assert torch.equal(out[k], v), k

    def test_encoder_only_teacher_leaves_the_embedding_alone(self):
        model = teacher_model(ema_embedding=False)
        move_the_student(model)
        model.update_teacher(0.9)
        sd = model.state_dict()
        out = prepare_backbone_state_dict(sd, encoder_source="teacher")
        for k, v in sd.items():
            if k.startswith("encoder."):
                assert torch.equal(out[k], v), k

    def test_teacher_source_without_a_teacher_raises(self):
        """A backbone trained with no EMA cannot answer for a teacher head."""
        sd = ConfigurableModel(**BASE_CFG).eval().state_dict()
        assert not has_teacher_weights(sd)
        with pytest.raises(ValueError, match="teacher"):
            prepare_backbone_state_dict(sd, encoder_source="teacher")

    def test_unknown_source_raises(self):
        sd = teacher_model().state_dict()
        with pytest.raises(ValueError):
            prepare_backbone_state_dict(sd, encoder_source="ema")

    def test_input_state_dict_is_not_mutated(self):
        model = teacher_model()
        move_the_student(model)
        model.update_teacher(0.9)
        sd = model.state_dict()
        before = {k: v.clone() for k, v in sd.items()}
        prepare_backbone_state_dict(sd, encoder_source="teacher")
        for k, v in sd.items():
            assert torch.equal(v, before[k]), k


# --- 2. the promoted backbone IS the teacher encoder ----------------------


class TestPromotedBackboneMatchesTheTeacher:

    def loaded(self, sd, source):
        backbone = ConfigurableModel(**BASE_CFG).eval()
        backbone.load_state_dict(prepare_backbone_state_dict(sd, source))
        return backbone

    def test_promoted_latents_equal_the_teacher_path(self):
        """The whole point: e_t off a promoted checkpoint is the teacher's h_t
        the drift probe measures (PR #387), to floating-point noise."""
        model = teacher_model()
        move_the_student(model)
        model.update_teacher(0.9)
        x = probe()

        want, _ = extract_teacher_encoder_latents(model, x)
        got, _ = extract_encoder_latents(self.loaded(model.state_dict(), "teacher"), x)
        assert got.shape == want.shape
        # Two module instances holding one set of fp32 weights agree to a few
        # 1e-6; the student and the teacher are 6.0 apart here (see
        # test_the_two_encoders_disagree), so 1e-4 separates the two cases.
        assert torch.allclose(got, want, atol=1e-4)

    def test_student_load_still_gives_the_student_latents(self):
        model = teacher_model()
        move_the_student(model)
        model.update_teacher(0.9)
        x = probe()

        want, _ = extract_encoder_latents(model, x)
        got, _ = extract_encoder_latents(self.loaded(model.state_dict(), "student"), x)
        assert torch.allclose(got, want, atol=1e-4)

    def test_the_two_encoders_disagree(self):
        """If they matched, the two heads would be measuring one thing."""
        model = teacher_model()
        move_the_student(model)
        model.update_teacher(0.9)
        x = probe()
        sd = model.state_dict()
        s, _ = extract_encoder_latents(self.loaded(sd, "student"), x)
        t, _ = extract_encoder_latents(self.loaded(sd, "teacher"), x)
        assert not torch.allclose(s, t, atol=1e-3)

    def test_the_forecaster_is_untouched(self):
        """There is no teacher forecaster. Rollout must stay the student's, so
        the teacher head differs from the student head in the encoder only."""
        model = teacher_model()
        move_the_student(model)
        model.update_teacher(0.9)
        sd = model.state_dict()
        student = self.loaded(sd, "student")
        teacher = self.loaded(sd, "teacher")
        for a, b in zip(student.transformer.layers.parameters(),
                        teacher.transformer.layers.parameters()):
            assert torch.equal(a, b)

        h = torch.randn(2, 8, BASE_CFG["H"],
                        generator=torch.Generator().manual_seed(11))
        assert torch.allclose(rollout_latent(student, h, 2),
                              rollout_latent(teacher, h, 2), atol=1e-6)


# --- 3. load_backbone_from_checkpoint -------------------------------------


class TestLoadBackboneFromCheckpoint:

    def saved(self, tmp_path):
        model = teacher_model()
        move_the_student(model)
        model.update_teacher(0.9)
        path = tmp_path / "bb.pth"
        torch.save(model.state_dict(), path)
        return model, str(path)

    def kwargs(self):
        """The fields a state_dict cannot disambiguate, matched to BASE_CFG."""
        return dict(C=1, H=16, W=8, nhead=2, num_layers=1,
                    encoder_type="gru", rev_norm_kind="ewma", rev_norm_span=8,
                    ffn_mult=2, dropout=0.0)

    def test_teacher_source_loads_the_teacher_encoder(self, tmp_path):
        model, path = self.saved(tmp_path)
        backbone, _ = load_backbone_from_checkpoint(
            path, device="cpu", encoder_source="teacher", **self.kwargs())
        x = probe()
        want, _ = extract_teacher_encoder_latents(model, x)
        got, _ = extract_encoder_latents(backbone, x)
        assert torch.allclose(got, want, atol=1e-4)

    def test_default_is_still_the_student(self, tmp_path):
        model, path = self.saved(tmp_path)
        backbone, _ = load_backbone_from_checkpoint(
            path, device="cpu", **self.kwargs())
        x = probe()
        want, _ = extract_encoder_latents(model, x)
        got, _ = extract_encoder_latents(backbone, x)
        assert torch.allclose(got, want, atol=1e-4)


# --- 4. the head remembers which encoder trained it -----------------------


class TestEncoderSourceMarker:

    def test_round_trip(self, tmp_path):
        head = tmp_path / "qhead_final.pth"
        head.write_bytes(b"")
        save_encoder_source(str(head), "teacher")
        assert load_encoder_source(str(head)) == "teacher"

    def test_marker_sits_next_to_the_checkpoint(self, tmp_path):
        head = tmp_path / "qhead_final.pth"
        marker = Path(encoder_source_marker_path(str(head)))
        assert marker.parent == head.parent
        assert marker.name.startswith("qhead_final")
        assert marker.suffix != ".pth"

    def test_absent_marker_reads_as_unknown(self, tmp_path):
        """Every head trained before #393 has no marker. Those stay loadable."""
        assert load_encoder_source(str(tmp_path / "old_final.pth")) is None

    def test_rejects_an_unknown_source(self, tmp_path):
        head = tmp_path / "qhead_final.pth"
        with pytest.raises(ValueError):
            save_encoder_source(str(head), "ema")


# --- 5. the two downstream scripts ----------------------------------------


def argparse_defaults(path: Path) -> dict[str, object]:
    """default= of every add_argument call in a script."""
    defaults: dict[str, object] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value.startswith("--")):
            continue
        value = None
        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    value = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    value = "<non-literal>"
                break
        defaults[first.value] = value
    return defaults


@pytest.mark.parametrize("script", [HEAD_TRAIN_PY, EVAL_PY],
                         ids=["head_train", "gift_eval"])
class TestDownstreamScriptWiring:

    def test_encoder_source_flag_defaults_to_student(self, script):
        assert argparse_defaults(script).get("--encoder-source") == "student"

    def test_encoder_type_flag_still_exists(self, script):
        """--encoder-source must not have displaced --encoder-type."""
        assert "--encoder-type" in argparse_defaults(script)

    def test_the_inline_strip_is_gone(self, script):
        """Both scripts had their own copy of the strip. One helper now owns
        promotion + strip; a leftover copy would drop the promoted weights."""
        src = script.read_text()
        assert "prepare_backbone_state_dict(" in src
        assert 'k.startswith("teacher_")' not in src


class TestEvalGuardsTheHeadEncoder:

    def test_eval_reads_the_marker(self):
        """A teacher head run through the student encoder is a wrong number
        that looks fine. The eval must refuse it."""
        src = EVAL_PY.read_text()
        assert "load_encoder_source(" in src

    def test_head_train_writes_the_marker(self):
        assert "save_encoder_source(" in HEAD_TRAIN_PY.read_text()


# --- 6. end-to-end on CPU -------------------------------------------------


TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")


def run_script(script, tmp_path, argv):
    import os
    import subprocess
    import sys
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-u", str(script)] + argv
    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          cwd=str(tmp_path), timeout=1800)


def pretrain_a_backbone(tmp_path, with_teacher=True):
    """4 steps of real contrastive training, with or without an EMA teacher.

    With one, the student moves every step and the teacher trails it at
    α = 0.9, so the saved checkpoint holds two genuinely different encoders.
    """
    name = "e2e393" if with_teacher else "e2e393noema"
    save_dir = tmp_path / f"bb_{name}"
    save_dir.mkdir(exist_ok=True)
    ema = (["--ema-embedding", "--ema-encoder", "--ema-tau", "0.9"]
           if with_teacher else [])
    res = run_script(TRAIN_PY, tmp_path, [
        "--device", "cpu", "--total-steps", "4", "--save-every", "2",
        "--batch-size", "2", "--lr", "1e-3", "--weight-decay", "0.1",
        "--save-dir", str(save_dir), "--run-name", name,
        "--mix-ratio", "1.0", "--synth-kind", "periodic",
        "--t-raw", "64", "--n-channels", "1", "--d-model", "32",
        "--n-heads", "2", "--num-layers", "1", "--num-encoder-layers", "1",
        "--log-every", "1", "--seed", "42", "--tau", "0.10",
        "--hf-repo", "none", "--hf-path", "none",
        "--loss-shape", "cosine_similarity_batch_split_pred_rep",
    ] + ema)
    if res.returncode != 0:
        pytest.fail(f"backbone rc={res.returncode}\n{res.stderr[-3000:]}")
    ckpt = save_dir / f"{name}_0k.pth"
    assert ckpt.exists(), sorted(p.name for p in save_dir.iterdir())
    return str(ckpt)


def train_a_head(tmp_path, backbone, source):
    save_dir = tmp_path / f"head_{source}"
    save_dir.mkdir(exist_ok=True)
    res = run_script(HEAD_TRAIN_PY, tmp_path, [
        "--backbone-path", backbone, "--encoder-source", source,
        "--device", "cpu", "--quantile-head", "--grad-clip", "1.0",
        "--forecast-len", "16", "--batch-size", "2", "--lr", "1e-3",
        "--total-steps", "3", "--save-every", "100", "--log-every", "1",
        "--save-dir", str(save_dir), "--run-name", f"h393_{source}",
        "--seed", "20260722", "--mix-ratio", "1.0",
        "--hf-repo", "none", "--hf-path", "none",
        "--t-raw", "64", "--n-channels", "1", "--d-model", "32",
        "--n-heads", "2", "--num-layers", "1",
        "--encoder-type", "gru", "--rev-norm-kind", "ewma",
        "--rev-norm-span", "8",
    ])
    if res.returncode != 0:
        pytest.fail(f"head[{source}] rc={res.returncode}\n"
                    f"{res.stdout[-2000:]}\n{res.stderr[-3000:]}")
    return save_dir, res.stdout


def head_losses(save_dir, source):
    import csv as _csv
    path = save_dir / f"h393_{source}_losses.csv"
    with open(path) as fh:
        return [float(r["loss"]) for r in _csv.DictReader(fh)]


class TestEndToEnd:
    """One real backbone, two real heads. Static wiring tests can't tell a
    flag that parses from a flag that reaches the forward pass."""

    def test_the_two_encoders_train_different_heads(self, tmp_path):
        backbone = pretrain_a_backbone(tmp_path)
        student_dir, student_log = train_a_head(tmp_path, backbone, "student")
        teacher_dir, teacher_log = train_a_head(tmp_path, backbone, "teacher")

        assert "encoder=student" in student_log
        assert "encoder=teacher" in teacher_log

        student = head_losses(student_dir, "student")
        teacher = head_losses(teacher_dir, "teacher")
        assert len(student) == len(teacher) == 3
        assert student != teacher, (
            "--encoder-source teacher trained on the student's latents: "
            f"{student} vs {teacher}")

    def test_the_marker_lands_next_to_every_head_checkpoint(self, tmp_path):
        backbone = pretrain_a_backbone(tmp_path)
        save_dir, _ = train_a_head(tmp_path, backbone, "teacher")
        final = save_dir / "h393_teacher_final.pth"
        assert final.exists()
        assert load_encoder_source(str(final)) == "teacher"

    def test_a_backbone_without_a_teacher_is_refused(self, tmp_path):
        """Silently falling back to the student is the failure mode this
        experiment cannot detect after the fact."""
        backbone = pretrain_a_backbone(tmp_path, with_teacher=False)
        res = run_script(HEAD_TRAIN_PY, tmp_path, [
            "--backbone-path", backbone, "--encoder-source", "teacher",
            "--device", "cpu", "--quantile-head", "--forecast-len", "16",
            "--batch-size", "2", "--total-steps", "1", "--log-every", "1",
            "--save-dir", str(tmp_path / "out"), "--run-name", "nope",
            "--seed", "20260722", "--mix-ratio", "1.0",
            "--hf-repo", "none", "--hf-path", "none",
            "--t-raw", "64", "--n-channels", "1", "--d-model", "32",
            "--n-heads", "2", "--num-layers", "1", "--encoder-type", "gru",
            "--rev-norm-kind", "ewma", "--rev-norm-span", "8",
        ])
        assert res.returncode != 0
        assert "teacher_* weights" in (res.stdout + res.stderr)
