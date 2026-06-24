"""Pin the CLI surface of the SIGReg λ-sweep launchers (#363).

#363 sweeps three (λ_embedding, λ_encoding) combinations on the
``λ_e=1.0, λ_h=0.1`` SIGReg recipe from #359 (which itself extends #355's
``λ_e=λ_h=0.1``). The arms are:

    emb100_enc01   λ_e=10.0, λ_h=0.1
    emb100_enc10   λ_e=10.0, λ_h=1.0
    emb100_enc100  λ_e=10.0, λ_h=10.0
    emb10_enc10    λ_e=1.0,  λ_h=1.0   (optional 4th, run only if compute is
                                         left after the first three)

These tests pin:
 * The parameterised backbone launcher passes through the two λ flags and
   builds a canonical $NAME tag from the suffix.
 * The recipe-flag set on the python command is the #359 backbone launcher's
   verbatim, so the only cross-arm degree of freedom is (λ_e, λ_h).
 * The sweep driver iterates exactly the three required (λ_e, λ_h, suffix)
   tuples (plus the optional fourth).
 * The downstream launcher and the parallel-GPU dispatcher accept a suffix
   and inherit the #359 head-matched protocol verbatim.
"""

import os
import re

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP_DIR = os.path.join(
    REPO_ROOT, "experiments", "2026-06-24_sigreg_lambda_sweep", "scripts")

BB_PATH = os.path.join(EXP_DIR, "train_backbone_sigreg.sh")
DL_PATH = os.path.join(EXP_DIR, "downstream_sigreg.sh")
LDS_PATH = os.path.join(EXP_DIR, "launch_downstream.sh")
SWEEP_PATH = os.path.join(EXP_DIR, "launch_arms.sh")


def _read(p):
    with open(p) as fh:
        return fh.read()


REQUIRED_BB_FLAGS = [
    "--batch-size 512",
    "--lr 1e-3",
    "--weight-decay 0.1",
    "--adam-beta1 0.9",
    "--adam-beta2 0.98",
    "--seed",
    "--hf-repo jeremycochoy/gift-pretrain-full-4096",
    "--hf-path small_v1",
    "--t-raw 4096",
    "--n-channels 1",
    "--d-model 384",
    "--n-heads 6",
    "--encoder-dropkey 0.70",
    "--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt",
    "--pos-in-denominator",
    "--subtract-contrastive-floor",
    "--ema-embedding",
    "--ema-encoder",
    "--ema-tau 0.99",
    "--cpc-infonce-weight 1.0",
    "--sigreg-embedding",
    "--sigreg-encoding",
    "--sigreg-n-chunk 2048",
    "--tau 0.10",
    "--rev-norm-kind ewma",
    "--rev-norm-span 128",
    "--encoder-type gru",
    "--synth-kind forked-arma",
    "--mix-ratio 0.0078125",
    "--crossfade-triplets 1",
    "--mixup-p 0.3",
    "--freq-emb-dim 3",
    "--seasonality-emb-dim 3",
    "--residual-dtype fp32",
    "--attn-dtype fp16",
    "--ffn-dtype fp16",
    "--conv-dtype fp16",
    "--patch-emb-dtype fp32",
]


def test_backbone_launcher_exists_and_is_executable():
    assert os.path.isfile(BB_PATH), f"missing launcher: {BB_PATH}"
    assert os.access(BB_PATH, os.X_OK), f"launcher not executable: {BB_PATH}"


def test_backbone_launcher_consumes_positional_lambdas_and_suffix():
    """train_backbone_sigreg.sh <gpu> <lambda_e> <lambda_h> <suffix> [steps] [save_every]."""
    text = _read(BB_PATH)
    assert 'GPU="${1' in text and "?" in text.split('GPU="${1', 1)[1].split('}')[0]
    assert 'LAMBDA_E="${2' in text
    assert 'LAMBDA_H="${3' in text
    assert 'SUFFIX="${4' in text


def test_backbone_launcher_threads_lambdas_through_to_train_py():
    text = _read(BB_PATH)
    assert "--sigreg-embedding-weight" in text
    assert "--sigreg-encoding-weight" in text
    # The λ values must come from the shell vars, not hardcoded — otherwise the
    # script would only train one of the four arms.
    m = re.search(r'--sigreg-embedding-weight\s+"\$\{?LAMBDA_E\}?"', text)
    assert m, "expected --sigreg-embedding-weight \"$LAMBDA_E\" in launcher"
    m = re.search(r'--sigreg-encoding-weight\s+"\$\{?LAMBDA_H\}?"', text)
    assert m, "expected --sigreg-encoding-weight \"$LAMBDA_H\" in launcher"


def test_backbone_launcher_tag_encodes_suffix():
    """$NAME must end in `_${SUFFIX}` so per-arm artefacts don't collide."""
    text = _read(BB_PATH)
    assert re.search(
        r'NAME="bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_\$\{?SUFFIX\}?"',
        text), "expected $NAME to embed ${SUFFIX}"


@pytest.mark.parametrize("flag", REQUIRED_BB_FLAGS)
def test_backbone_launcher_carries_359_recipe_flag(flag):
    """Every #359 recipe flag must still be on the python command verbatim."""
    text = _read(BB_PATH)
    assert flag in text, f"missing recipe flag: {flag}"


def test_backbone_launcher_keeps_359_memory_knobs():
    text = _read(BB_PATH)
    for env in ("FCST_GRAD_CKPT=1", "XSHH_ALLT_CHUNK=1", "PATCH_ENC_CKPT=1",
                "PATCH_ENC_CHUNK=4", "TEACHER_EMBED_CHUNK", "CPC_CB_CHUNK"):
        assert env in text, f"missing memory knob: {env}"


def test_backbone_launcher_seed_matches_359():
    text = _read(BB_PATH)
    assert "SEED=20260520" in text, "seed must match #355/#359 for cross-arm comparison"


SWEEP_ARMS = [
    ("emb100_enc01",  "10.0", "0.1"),
    ("emb100_enc10",  "10.0", "1.0"),
    ("emb100_enc100", "10.0", "10.0"),
]
OPTIONAL_ARM = ("emb10_enc10", "1.0", "1.0")


def test_sweep_driver_exists_and_is_executable():
    assert os.path.isfile(SWEEP_PATH), f"missing sweep driver: {SWEEP_PATH}"
    assert os.access(SWEEP_PATH, os.X_OK)


@pytest.mark.parametrize("suffix,le,lh", SWEEP_ARMS)
def test_sweep_driver_lists_required_arm(suffix, le, lh):
    """Each of the three required arms must appear in the driver, in (λ_e, λ_h, suffix) form."""
    text = _read(SWEEP_PATH)
    # The driver must reference the suffix and both λ values somewhere together.
    pattern = re.compile(rf'{re.escape(le)}\s+{re.escape(lh)}\s+{re.escape(suffix)}')
    assert pattern.search(text), f"sweep driver missing arm {suffix} ({le}, {lh})"


def test_sweep_driver_keeps_optional_fourth_arm_referenced():
    """The fourth `λ_e=1.0, λ_h=1.0` arm is optional but must still be listed
    (commented or guarded) so the experiment phase can opt in."""
    suffix, le, lh = OPTIONAL_ARM
    text = _read(SWEEP_PATH)
    pattern = re.compile(rf'{re.escape(le)}\s+{re.escape(lh)}\s+{re.escape(suffix)}')
    assert pattern.search(text), f"sweep driver missing optional arm {suffix}"


def test_sweep_driver_runs_arms_in_issue_order():
    """#363 §Objective fixes the order: emb100_enc01 → emb100_enc10 → emb100_enc100."""
    text = _read(SWEEP_PATH)
    idxs = [text.find(s) for s, _, _ in SWEEP_ARMS]
    assert all(i >= 0 for i in idxs), f"missing arm(s) in driver: {idxs}"
    assert idxs == sorted(idxs), f"arms must appear in issue order, got {idxs}"


def test_downstream_launcher_exists_and_takes_suffix():
    """downstream_sigreg.sh <head_layers: 2|6> <gpu> <suffix>."""
    assert os.path.isfile(DL_PATH)
    assert os.access(DL_PATH, os.X_OK)
    text = _read(DL_PATH)
    assert 'HL="${1' in text
    assert 'GPU="${2' in text
    assert 'SUFFIX="${3' in text
    # TAG built from suffix so it parallels the backbone $NAME.
    assert re.search(
        r'TAG="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_\$\{?SUFFIX\}?"',
        text), "expected $TAG to embed ${SUFFIX}"


def test_downstream_protocol_matches_359():
    text = _read(DL_PATH)
    # Head-matched: 30k steps best-loss, then 10k resume from best-loss head on last bb.
    for flag in [
        "--head-arch transformer", "--head-causal true",
        "--head-num-layers", '"$HL"',
        "--head-nhead 6", "--head-ffn-mult 4.0",
        "--head-dropout 0.1", "--head-train-input e_then_f",
        "--total-steps", "--batch-size 256", "--lr 1e-3",
        "--beta1 0.9", "--beta2 0.98", "--weight-decay 0.1",
        "--schedule cosine", "--final-lr-ratio 0.1",
        "--reconstruction forecaster",
        "--forecast-len 16",
        "--quantile-head",
        "--strategy B4",
    ]:
        assert flag in text, f"downstream launcher missing #359 protocol flag: {flag}"


def test_launch_downstream_drives_two_heads_on_two_gpus():
    """launch_downstream.sh <suffix> — 2L on GPU 0 + 6L on GPU 1 (parallel)."""
    assert os.path.isfile(LDS_PATH)
    assert os.access(LDS_PATH, os.X_OK)
    text = _read(LDS_PATH)
    assert 'SUFFIX="${1' in text
    # Must dispatch (head=2, gpu=0, $SUFFIX) and (head=6, gpu=1, $SUFFIX) in parallel.
    assert re.search(r'\s2\s+0\s+"\$\{?SUFFIX\}?"', text), \
        "launcher must invoke <something> 2 0 \"$SUFFIX\""
    assert re.search(r'\s6\s+1\s+"\$\{?SUFFIX\}?"', text), \
        "launcher must invoke <something> 6 1 \"$SUFFIX\""
    assert "downstream_sigreg.sh" in text, "launcher must reference downstream_sigreg.sh"
    assert "wait" in text, "launcher must wait on both GPU PIDs"


# ---- Iter-2 review: WT/OUT must be required, no hardcoded path defaults ----

@pytest.mark.parametrize("path", [BB_PATH, DL_PATH, LDS_PATH, SWEEP_PATH])
def test_launchers_require_wt_and_out(path):
    """Iter-1 review HIGH: WT and OUT must be required, not silently defaulted.
    A `WT=/tmp/...` default sent runs to a non-existent path on elisa while
    only WARNing on empty HF_TOKEN — the GPU then idled on anonymous rate
    limits. The launchers must abort fast (`${VAR:?…}` or explicit check)."""
    text = _read(path)
    # ${VAR:?msg} form, OR explicit `[ -d "$WT" ] || ... exit` check.
    has_wt_required = (
        re.search(r'\$\{WT:\?[^}]+\}', text)
        or re.search(r'\[\s*-d\s+"\$WT"\s*\]\s*\|\|', text))
    assert has_wt_required, f"{path}: WT must be required (`${{WT:?...}}` or `[ -d $WT ] || exit`)"
    has_out_required = re.search(r'\$\{OUT:\?[^}]+\}', text)
    assert has_out_required, f"{path}: OUT must be required (`${{OUT:?...}}`)"
    # No hardcoded default that resolves to /tmp/contrastive-forecasting-363
    # or .claude/worktrees — those are review-time paths that don't exist on
    # elisa and silently broke prior iter.
    assert "WT:-/tmp/contrastive-forecasting-363" not in text, (
        f"{path}: must not default WT to the review worktree path")
    assert "WT:-/home/jupyter/contrastive-forecasting/.claude/worktrees" not in text, (
        f"{path}: must not default WT to .claude/worktrees path")


@pytest.mark.parametrize("path,refs", [
    (BB_PATH,  ['"$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"',
                '"$WT/experiments/hf_token.txt"']),
    (DL_PATH,  ['"$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"',
                '"$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"',
                '"$WT/experiments/hf_token.txt"']),
    (LDS_PATH, ['"$WT/experiments/2026-06-24_sigreg_lambda_sweep/scripts/downstream_sigreg.sh"']),
    (SWEEP_PATH, ['"$WT/experiments/2026-06-24_sigreg_lambda_sweep/scripts/train_backbone_sigreg.sh"',
                  '"$WT/experiments/2026-06-24_sigreg_lambda_sweep/scripts/launch_downstream.sh"']),
])
def test_launcher_wt_derived_paths_are_referenced(path, refs):
    """Iter-1 review LOW: pin the exact WT-derived paths so a future rename
    can't silently re-introduce a non-existent path under the review default."""
    text = _read(path)
    for ref in refs:
        assert ref in text, f"{path}: expected reference {ref}"


@pytest.mark.parametrize("path", [BB_PATH, DL_PATH])
def test_launcher_fails_fast_on_missing_hf_token(path):
    """Iter-1 review HIGH (related): empty HF_TOKEN must abort, not WARN.
    Anonymous HF stream throttles the GPU to ~1 sps."""
    text = _read(path)
    # Must check token file exists, or abort on empty token.
    assert re.search(r'\[\s*-f\s+"\$HF_TOKEN_PATH"\s*\]\s*\|\|.*exit', text), (
        f"{path}: must check HF_TOKEN_PATH exists and exit if not")
    assert re.search(r'\[\s*-n\s+"\$HF_TOKEN"\s*\]\s*\|\|.*exit', text), (
        f"{path}: must exit on empty HF_TOKEN (not just WARN)")
