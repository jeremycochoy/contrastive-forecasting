"""#482: the trainer flags of the paper ablations (paper/sections/ablation.tex).

Each ablation run is the reference run with the one change of its row. Each
flag here is off by default, so the reference flags train the same model.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.models import ConfigurableModel

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")


@pytest.fixture(scope="module")
def train_py():
    spec = importlib.util.spec_from_file_location("train_py_482", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Built(Exception):
    """main() built its model: the test has what it needs."""


def model_config_of(train_py, monkeypatch, tmp_path, *argv):
    """The keyword arguments that main() gives to ConfigurableModel."""
    seen = {}

    def spy(**config):
        seen.update(config)
        raise Built

    monkeypatch.setattr(train_py, "ConfigurableModel", spy)
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--device", "cpu", "--weight-decay", "0.1",
        "--save-dir", str(tmp_path), *argv])
    with pytest.raises(Built):
        train_py.main()
    return seen


def tiny_model(**extra):
    return ConfigurableModel(C=1, H=16, W=16, encoder_type="gru",
                             num_layers=1, nhead=2, num_encoder_layers=1,
                             **extra)


# ---------------------------------------------------------------------------
# --dropout: the dropout of the blocks (row "Dropout of 0.1 in the blocks")
# ---------------------------------------------------------------------------

def test_the_blocks_keep_the_dropout_of_0_1_by_default(train_py, monkeypatch,
                                                         tmp_path):
    config = model_config_of(train_py, monkeypatch, tmp_path)
    assert config["dropout"] == 0.1


def test_dropout_0_reaches_the_model(train_py, monkeypatch, tmp_path):
    config = model_config_of(train_py, monkeypatch, tmp_path,
                             "--dropout", "0")
    assert config["dropout"] == 0.0


@pytest.mark.parametrize("value", ["-0.1", "1", "1.5"])
def test_a_dropout_outside_0_to_1_stops_the_run(train_py, monkeypatch,
                                                tmp_path, value):
    with pytest.raises(SystemExit, match="--dropout"):
        model_config_of(train_py, monkeypatch, tmp_path, "--dropout", value)


def test_a_model_with_dropout_0_has_no_dropout_in_its_blocks():
    model = tiny_model(dropout=0.0)
    rates = [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    rates += [m._attn_dropout_p for m in model.modules()
              if hasattr(m, "_attn_dropout_p")]
    assert rates and all(p == 0.0 for p in rates)


# ---------------------------------------------------------------------------
# --zero-time-scale-embedding (row "Time-scale embedding")
# ---------------------------------------------------------------------------

def test_the_time_scale_tables_train_by_default(train_py):
    args = train_py.build_parser().parse_args(["--weight-decay", "0.1"])
    assert args.zero_time_scale_embedding is False


def test_zero_time_scale_embedding_zeroes_and_freezes_both_tables(train_py):
    model = tiny_model(freq_emb_dim=3, seasonality_emb_dim=3)
    assert train_py.zero_time_scale_embedding_(model) == 2
    tables = [model.freq_embedding.embedding.weight,
              model.seasonality_embedding.embedding.weight]
    assert all(not t.requires_grad and torch.count_nonzero(t) == 0
               for t in tables)


def test_the_zero_tables_stay_at_zero_after_an_adamw_step(train_py):
    model = tiny_model(freq_emb_dim=3, seasonality_emb_dim=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2,
                                  weight_decay=0.1)
    train_py.zero_time_scale_embedding_(model)
    ids = torch.ones(2, dtype=torch.long)
    x = torch.randn(2, 64, 1)
    f, h = model(x, freq_ids=ids, seasonality_ids=ids)[:2]
    (f.square().mean() + h.square().mean()).backward()
    optimizer.step()
    assert torch.count_nonzero(model.freq_embedding.embedding.weight) == 0
    assert torch.count_nonzero(
        model.seasonality_embedding.embedding.weight) == 0


def test_zero_time_scale_embedding_needs_a_table(train_py, monkeypatch,
                                                 tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--device", "cpu", "--weight-decay", "0.1",
        "--save-dir", str(tmp_path), "--freq-emb-dim", "0",
        "--seasonality-emb-dim", "0", "--zero-time-scale-embedding"])
    with pytest.raises(SystemExit, match="--zero-time-scale-embedding"):
        train_py.main()


# ---------------------------------------------------------------------------
# --encoder-type timesfm (row "GRU patch embedding")
# ---------------------------------------------------------------------------

def test_the_timesfm_block_has_the_names_and_widths_of_timesfm():
    from src.encoders import create_encoder
    enc = create_encoder("timesfm", 22, 32)
    shapes = {k: tuple(v.shape) for k, v in enc.state_dict().items()}
    assert shapes == {
        "hidden_layer.0.weight": (32, 22), "hidden_layer.0.bias": (32,),
        "output_layer.weight": (32, 32), "output_layer.bias": (32,),
        "residual_layer.weight": (32, 22), "residual_layer.bias": (32,),
        "layer_norm.weight": (32,), "layer_norm.bias": (32,)}


def test_the_timesfm_block_is_an_mlp_plus_a_linear_map_then_a_layer_norm():
    from src.encoders import create_encoder
    enc = create_encoder("timesfm", 22, 32)
    x = torch.randn(2, 5, 1, 22)
    mlp = enc.output_layer(torch.nn.functional.silu(enc.hidden_layer[0](x)))
    want = torch.nn.functional.layer_norm(
        mlp + enc.residual_layer(x), (32,), enc.layer_norm.weight,
        enc.layer_norm.bias)
    assert torch.allclose(enc(x), want, atol=1e-6)
    assert enc(x).shape == (2, 5, 1, 32)


def test_encoder_type_timesfm_reaches_the_model(train_py, monkeypatch,
                                                tmp_path):
    config = model_config_of(train_py, monkeypatch, tmp_path,
                             "--encoder-type", "timesfm")
    assert config["encoder_type"] == "timesfm"
    model = ConfigurableModel(**dict(config, H=16, nhead=2, num_layers=1))
    assert type(model.encoder).__name__ == "TimesFMResidualEncoder"


# ---------------------------------------------------------------------------
# --ema-tau 0 (row "EMA teacher"): the teacher has the weights of the student
# ---------------------------------------------------------------------------

def teacher_model():
    return tiny_model(ema_embedding=True, ema_encoder=True)


def test_an_update_at_momentum_0_copies_the_student_into_the_teacher():
    model = teacher_model()
    with torch.no_grad():
        for p in model.transformer.parameters():
            p.add_(1.0)
    model.update_teacher(0.0)
    pairs = [(model.teacher_input_to_latent, model.transformer.input_to_latent)]
    pairs += list(zip(model.teacher_encoder_layers,
                      model.transformer.encoder_layers))
    for teacher, student in pairs:
        for t, s in zip(teacher.state_dict().values(),
                        student.state_dict().values()):
            assert torch.equal(t, s)


def test_the_trainer_accepts_momentum_0(train_py, monkeypatch, tmp_path):
    config = model_config_of(
        train_py, monkeypatch, tmp_path, "--ema-embedding", "--ema-encoder",
        "--num-encoder-layers", "1", "--ema-tau", "0",
        "--loss-shape", "cosine_similarity_batch_rep_only")
    assert config["ema_embedding"] and config["ema_encoder"]


@pytest.mark.parametrize("value", ["-0.1", "1"])
def test_a_momentum_outside_0_to_1_stops_the_run(train_py, monkeypatch,
                                                 tmp_path, value):
    with pytest.raises(SystemExit, match="--ema-tau"):
        model_config_of(
            train_py, monkeypatch, tmp_path, "--ema-embedding",
            "--ema-encoder", "--num-encoder-layers", "1", "--ema-tau", value,
            "--loss-shape", "cosine_similarity_batch_rep_only")


# ---------------------------------------------------------------------------
# --meanstd-no-leak (row "EWMA normalisation"): instance normalisation with no
# target leak (#475)
# ---------------------------------------------------------------------------

T_RAW, P = 1024, 16


def draws(lengths, n=4000):
    from src.forecasting_head import draw_short_context_ends
    lengths = torch.tensor(lengths).view(-1, 1, 1).repeat(n, 1, 1)
    sizes = torch.full((lengths.shape[0],), P)
    return draw_short_context_ends(lengths, sizes, T_RAW).view(n, -1)


def test_a_full_window_draws_a_context_of_1_to_32_patches():
    ends = draws([T_RAW])[:, 0]
    assert (ends % P == 0).all()
    assert set((ends // P).tolist()) == set(range(1, 33))


def test_a_padded_window_draws_1_patch_to_half_of_its_real_patches():
    # 500 real values: 31 whole patches after a first patch with padding.
    ends = draws([500])[:, 0]
    first_whole = T_RAW // P - 500 // P
    assert set((ends // P - first_whole).tolist()) == set(range(1, 16))


def test_a_window_shorter_than_two_patches_has_no_target():
    assert (draws([20, 31]) == T_RAW).all()


def test_the_draw_takes_one_number_per_window_as_the_uni2ts_draw():
    from src.forecasting_head import draw_context_ends, draw_short_context_ends
    lengths = torch.full((5, 1, 1), T_RAW)
    sizes = torch.full((5,), P)
    states = []
    for draw in (draw_context_ends, draw_short_context_ends):
        torch.manual_seed(3)
        draw(lengths, sizes, T_RAW)
        states.append(torch.get_rng_state())
    assert torch.equal(*states)


def meanstd_model():
    return ConfigurableModel(C=1, H=16, W=P, encoder_type="gru", num_layers=1,
                             nhead=2, rev_norm_kind="meanstd",
                             rev_norm_skip_leading_zeros=True)


def padded_walks(lengths=(T_RAW, 700, 300, T_RAW)):
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(len(lengths), T_RAW, 1)
    for b, n in enumerate(lengths):
        x[b, T_RAW - n:, 0] = 50.0 + torch.randn(n, generator=g).cumsum(0)
    return x


def test_the_statistics_read_the_context_only():
    from src.forecasting_head import mean_std_inputs
    m, x = meanstd_model(), padded_walks()
    torch.manual_seed(1)
    x_norm, _, target = mean_std_inputs(m, x, short_context=True)
    y = torch.where(target, 1e6 * torch.ones_like(x), x)
    torch.manual_seed(1)
    y_norm, _, target_y = mean_std_inputs(m, y, short_context=True)
    assert torch.equal(target, target_y)
    assert torch.equal(x_norm[~target], y_norm[~target])


def test_no_kept_anchor_reads_a_statistic_of_its_target(train_py):
    from src.forecasting_head import mean_std_inputs
    m, x = meanstd_model(), padded_walks()
    torch.manual_seed(2)
    _, _, target = mean_std_inputs(m, x, short_context=True)
    args = SimpleNamespace(meanstd_no_leak=True)
    start = train_py.step_start_mask(args, m, target)
    pad = train_py.group_padding(m.rev_norm.pad_mask, None, P, None,
                                 x.shape, start)
    ends = (~target).sum(dim=1).view(-1)          # the context of each row
    for b, end in enumerate(ends.tolist()):
        for t in range(T_RAW // P - 1):
            padding = bool(m.rev_norm.pad_mask[b, t * P:(t + 1) * P].all())
            # Kept: the next patch starts after the context.
            assert bool(pad[b, t, 0]) == (padding or (t + 1) * P < end)


def test_without_the_flag_the_start_mask_is_the_normaliser_s(train_py):
    m = meanstd_model()
    target = torch.zeros(2, T_RAW, 1, dtype=torch.bool)
    args = SimpleNamespace(meanstd_no_leak=False)
    assert train_py.step_start_mask(args, m, target) is None


def test_meanstd_no_leak_needs_meanstd(train_py, monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="--meanstd-no-leak"):
        model_config_of(train_py, monkeypatch, tmp_path, "--meanstd-no-leak")


# ---------------------------------------------------------------------------
# A few trainer steps of each new flag, on the recipe of the reference at a
# size the CPU trains in seconds, on the corpus of the #419 tests.
# ---------------------------------------------------------------------------

TINY_GREEN = (
    "--qk-norm", "--attn-out-norm", "--weight-decay", "0.1",
    "--gift-pretrain", "--freq-vocab", "v2",
    "--t-raw", "4096", "--n-channels", "1", "--d-model", "16",
    "--n-heads", "2", "--num-layers", "1", "--num-encoder-layers", "1",
    "--batch-size", "8", "--synth-kind", "forked-arma", "--mix-ratio", "0.25",
    "--crossfade-triplets", "1", "--mixup-p", "0.3", "--freq-emb-dim", "3",
    "--seasonality-emb-dim", "3",
    "--loss-shape", "cosine_similarity_batch_rep_only",
    "--align-loss-weight", "10", "--align-target", "teacher",
    "--moco-rep-keys", "--tau-rep", "1.0", "--ema-embedding", "--ema-encoder",
    "--sigreg-embedding", "--sigreg-encoding", "--train-rollout-depth", "3",
    "--rep-loss-weight", "0.1", "--latent-sphere", "--rev-norm-kind", "ewma",
    "--rev-norm-span", "128", "--mup-base-d-model", "8",
    "--mup-init-output", "0", "--mup-init-input", "0.3",
    "--total-steps", "2", "--log-every", "1", "--save-every", "1000000")


def tiny_green_run(tmp_path, *extra):
    """Two steps of TINY_GREEN with ``extra`` (the last value of a flag
    wins). Returns the stdout and the final state dict."""
    import os
    import subprocess
    pytest.importorskip("pyarrow")
    from tests.test_419_gift_pretrain import build_corpus, write_index
    folder, index, _ = build_corpus(tmp_path / "corpus")
    r = subprocess.run(
        [sys.executable, str(TRAIN_PY), "--device", "cpu", *TINY_GREEN,
         "--gift-pretrain-root", str(folder),
         "--gift-pretrain-index", str(write_index(tmp_path, index)), *extra,
         "--save-dir", str(tmp_path), "--run-name", "r"],
        capture_output=True, text=True, timeout=900,
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT),
                 CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout, torch.load(tmp_path / "r_final.pth", map_location="cpu")


def test_a_few_trainer_steps_with_meanstd_no_leak(tmp_path):
    out, _ = tiny_green_run(tmp_path, "--rev-norm-kind", "meanstd",
                            "--meanstd-no-leak", "--meanstd-z-max", "0")
    assert "no target leak" in out


def test_a_few_trainer_steps_with_momentum_0(tmp_path):
    _, sd = tiny_green_run(tmp_path, "--ema-tau", "0")
    teacher = {k: v for k, v in sd.items() if k.startswith("teacher_")}
    assert teacher
    for key, value in teacher.items():
        student = (key.replace("teacher_input_to_latent",
                               "transformer.input_to_latent")
                   .replace("teacher_encoder_layers",
                            "transformer.encoder_layers"))
        assert torch.equal(value, sd[student]), key


def test_a_few_trainer_steps_with_zero_time_scale_embedding(tmp_path):
    out, sd = tiny_green_run(tmp_path, "--zero-time-scale-embedding")
    assert "2 table(s) at 0, frozen" in out
    for key in ("freq_embedding.embedding.weight",
                "seasonality_embedding.embedding.weight"):
        assert sd[key].shape[1] == 3 and torch.count_nonzero(sd[key]) == 0


def test_a_few_trainer_steps_with_the_timesfm_block(tmp_path):
    _, sd = tiny_green_run(tmp_path, "--encoder-type", "timesfm")
    assert sd["encoder.residual_layer.weight"].shape == (16, 22)
    assert not any(k.startswith("encoder.gru.") for k in sd)


def test_a_few_trainer_steps_with_no_dropout(tmp_path):
    tiny_green_run(tmp_path, "--dropout", "0")
