"""Tests for #425: the reconstruction head on a copy of Moirai.

A copy of Moirai (#415, the value-space objective) has no contrastive loss,
so it has no encoder latent with a loss on it. Its one latent is the output
of the whole transformer at patch t: the vector that its own value head
reads. A reconstruction head of a copy of Moirai reads that vector, and
gives the values of patch t. The head of a run of ours reads the encoder
latent, as before.

Groups, all on the CPU:

1. The latent: the output latent is the input of the value head, it reads
   patch t and no later patch, and a checkpoint with a value head names it.
2. The loss and the score: the bank loss and strategy R read the latent
   that the caller names, and each patch is scored on its own values.
3. The head trainer: the head of a copy of Moirai trains on the output
   latent, the head of a run of ours on the encoder latent, alone or in a
   wave.
4. The eval script: strategy R scores a copy of Moirai on its output latent.
5. The proof on elisa: the parity script compares two per-config tables.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.forecasting_head as fh
from src.checkpoint import reconstruction_latent_of
from src.forecasting_head import (ForecastingHeadBank, bank_quantile_loss,
                                  bank_training_inputs,
                                  extract_encoder_latents,
                                  extract_forecaster_latents,
                                  extract_reconstruction_latents,
                                  reconstruct_horizon,
                                  reconstruction_quantile_loss,
                                  value_space_forward)
from src.freq_embedding import FREQ_NAMES_V2
from src.models import ConfigurableModel
from tests.test_425_encoder_reconstruction import (  # noqa: F401
    CPU, EVAL_PROTOCOL, HEAD_PROTOCOL, Q, SIZES, T, OracleHead,
    assert_same_run, bank_model, bank_of, corpus_flags, eval_args,
    ewma_model, job_argv, labels_of, load_eval_module, load_head_trainer,
    load_script, oracle_latents, saved, train_shared, train_solo, walk,
    windows)


def moirai_model(scaling="meanstd"):
    """A copy of Moirai at d_model 16 (MPM: mean/std, MPE: EWMA): patch
    sizes 8 to 128, zero padding, and the value heads of the value-space
    objective (#415), one for each patch size."""
    torch.manual_seed(4)
    norm = (dict(rev_norm_kind="meanstd") if scaling == "meanstd"
            else dict(rev_norm_kind="ewma", rev_norm_span=128))
    return ConfigurableModel(
        C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        rev_norm_skip_leading_zeros=True, num_encoder_layers=1,
        freq_emb_dim=3, num_freqs=len(FREQ_NAMES_V2), seasonality_emb_dim=3,
        multi_patch_sizes=SIZES, value_head_quantiles=Q, **norm).eval()


def moirai_one_size_model():
    """MOO: a copy of Moirai with one patch size, the EWMA and no padding.
    Its checkpoint holds one value head, ``value_head``."""
    torch.manual_seed(6)
    return ConfigurableModel(
        C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        rev_norm_kind="ewma", rev_norm_span=128, num_encoder_layers=1,
        freq_emb_dim=3, seasonality_emb_dim=3, num_freqs=10,
        value_head_quantiles=Q).eval()


def other_forecaster(model, by=0.05):
    """``model`` with each weight of its forecaster layers moved by ``by``.
    The encoder latent stays, and the output of the transformer changes."""
    with torch.no_grad():
        for p in model.transformer.layers.parameters():
            p.add_(by)
    return model


def normalised_batch(m, seed=5):
    """A batch as a head-training step reads it: the normalised windows,
    the mask of the scored values, and the labels of each row."""
    freq, seas = labels_of()
    torch.manual_seed(seed)
    x_norm, _, keep = bank_training_inputs(m, windows(), freq, SIZES)
    return x_norm, keep, freq, seas


# ---------------------------------------------------------------------------
# 1. The latent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scaling", ["meanstd", "ewma"])
@pytest.mark.parametrize("size", [8, 32])
def test_the_output_latent_is_the_vector_that_the_value_head_reads(scaling,
                                                                   size):
    """The value-space forward of the copy (#415) gives its value head the
    output of the transformer. The output latent is that tensor, so the
    value head of the copy decodes it into the forecast of the copy."""
    m = moirai_model(scaling)
    x_norm, _, freq, seas = normalised_batch(m)
    labels = dict(freq_ids=freq, seasonality_ids=seas, patch_size=size)
    with torch.no_grad():
        f_lat, o_lat, v_hat = value_space_forward(m, x_norm, **labels)
    B, n, C, H = f_lat.shape
    out, same = extract_reconstruction_latents(m, x_norm, "output",
                                               normalised=True, **labels)
    assert torch.equal(same, x_norm)
    assert torch.equal(out, f_lat.permute(0, 2, 1, 3).reshape(B * C, n, H))
    with torch.no_grad():
        decoded = m.value_head_for(size)(out).reshape(B, n, C, Q, size)
    assert torch.equal(decoded, v_hat)
    enc, _ = extract_reconstruction_latents(m, x_norm, "encoder",
                                            normalised=True, **labels)
    assert torch.allclose(enc, o_lat.permute(0, 2, 1, 3).reshape(B * C, n, H),
                          atol=1e-6)
    assert not torch.allclose(out, enc, atol=1e-3)


def test_the_two_latents_are_those_of_the_two_extract_functions():
    m = moirai_model()
    x_norm, _, freq, seas = normalised_batch(m)
    kw = dict(freq_ids=freq, seasonality_ids=seas, patch_size=16,
              normalised=True)
    for latent, extract in (("encoder", extract_encoder_latents),
                            ("output", extract_forecaster_latents)):
        got, _ = extract_reconstruction_latents(m, x_norm, latent, **kw)
        want, _ = extract(m, x_norm, **kw)
        assert torch.equal(got, want), latent
    default, _ = extract_reconstruction_latents(m, x_norm, **kw)
    assert torch.equal(default, extract_encoder_latents(m, x_norm, **kw)[0])


def test_an_unknown_latent_is_refused():
    m = moirai_model()
    with pytest.raises(ValueError, match="latent"):
        extract_reconstruction_latents(m, torch.zeros(1, 128, 1), "forecaster",
                                       normalised=True)


@pytest.mark.parametrize("latent", ["encoder", "output"])
def test_the_latent_of_a_patch_reads_that_patch_and_no_later_patch(latent):
    """Each of the two latents of patch t reads patch t and the patches
    before it. So a head can decode patch t from each one, and no value of
    a later patch reaches it."""
    m = moirai_model("ewma")
    x_norm = torch.randn(2, 256, 1, generator=torch.Generator().manual_seed(1))
    moved = x_norm.clone()
    moved[:, 5 * 16:6 * 16] += 1.0                           # patch 5
    kw = dict(patch_size=16, normalised=True)
    a, _ = extract_reconstruction_latents(m, x_norm, latent, **kw)
    b, _ = extract_reconstruction_latents(m, moved, latent, **kw)
    assert torch.equal(a[:, :5], b[:, :5])
    assert (a[:, 5] - b[:, 5]).abs().max() > 1e-3


def test_a_checkpoint_with_a_value_head_names_the_output_latent():
    """The value head marks a checkpoint of the value-space objective: a
    copy of Moirai. Each other checkpoint keeps the encoder latent."""
    assert reconstruction_latent_of(moirai_model().state_dict()) == "output"
    assert reconstruction_latent_of(
        moirai_one_size_model().state_dict()) == "output"
    assert reconstruction_latent_of(bank_model().state_dict()) == "encoder"
    ours = dict(ewma_model().state_dict(), **{
        "cpc_w1.weight": torch.zeros(1),
        "teacher_input_to_latent.skip.weight": torch.zeros(1)})
    assert reconstruction_latent_of(ours) == "encoder"


# ---------------------------------------------------------------------------
# 2. The loss and the score
# ---------------------------------------------------------------------------

def test_a_bank_reconstruction_reads_the_latent_that_the_caller_names(
        monkeypatch):
    """With ``latent='output'``, each group reads the output latent of its
    normalised rows at its size, and no encoder latent. The loss is the
    group losses weighted by their share of the rows, each on the values of
    the patch of its latent."""
    seen = []
    real = fh.extract_forecaster_latents

    def spy(backbone, x, **kw):
        seen.append((kw.get("patch_size"), kw.get("normalised")))
        return real(backbone, x, **kw)

    def no_encoder(*_, **__):
        raise AssertionError("this head reads the output latent only")

    m = moirai_model()
    bank = bank_of()
    x_norm, keep, freq, seas = normalised_batch(m)
    sizes = torch.tensor([8, 8, 32, 32, 8, 32, 32, 32])
    parts = []
    for size, rows in ((8, [0, 1, 4]), (32, [2, 3, 5, 6, 7])):
        f_bc, _ = extract_forecaster_latents(
            m, x_norm[rows], freq_ids=freq[rows], seasonality_ids=seas[rows],
            patch_size=size, normalised=True)
        parts.append(len(rows) / 8 * reconstruction_quantile_loss(
            bank.head_for(size)(f_bc), x_norm[rows], size, keep=keep[rows]))
    monkeypatch.setattr(fh, "extract_forecaster_latents", spy)
    monkeypatch.setattr(fh, "extract_encoder_latents", no_encoder)
    loss = bank_quantile_loss(m, bank, x_norm, sizes, keep, freq, seas,
                              reconstruction=True, latent="output")
    assert seen == [(8, True), (32, True)]
    assert torch.allclose(loss, parts[0] + parts[1])


def test_the_bank_loss_reads_the_encoder_latent_by_default():
    m = moirai_model()
    bank = bank_of()
    x_norm, keep, freq, seas = normalised_batch(m)
    sizes = torch.tensor([8, 16, 16, 64, 128, 8, 16, 64])
    args = (m, bank, x_norm, sizes, keep, freq, seas)
    default = bank_quantile_loss(*args, reconstruction=True)
    encoder = bank_quantile_loss(*args, reconstruction=True, latent="encoder")
    output = bank_quantile_loss(*args, reconstruction=True, latent="output")
    assert torch.equal(default, encoder)
    assert not torch.allclose(default, output)


def test_a_perfect_reconstruction_of_the_output_latent_has_no_loss(
        monkeypatch):
    """The target of the output latent of patch t is patch t itself, and not
    patch t + 1, the target of the forecast of the copy."""
    monkeypatch.setattr(fh, "extract_forecaster_latents", oracle_latents)
    m = moirai_model()
    freq, seas = labels_of()
    torch.manual_seed(5)
    x_norm, sizes, keep = bank_training_inputs(m, windows(), freq, SIZES)
    oracle = ForecastingHeadBank({p: OracleHead(p) for p in SIZES})
    loss = bank_quantile_loss(m, oracle, x_norm, sizes, keep, freq, seas,
                              reconstruction=True, latent="output")
    assert loss.item() == 0.0


@pytest.mark.parametrize("scaling,size,horizon", [
    ("meanstd", 32, 40), ("ewma", 16, 30), ("meanstd", 128, 300)])
def test_a_perfect_head_on_the_output_latent_gives_back_the_true_horizon(
        monkeypatch, scaling, size, horizon):
    monkeypatch.setattr(fh, "extract_forecaster_latents", oracle_latents)
    m = moirai_model(scaling)
    series = walk(T + horizon, seed=3)
    ctx, future = series[:T, None], series[T:]
    out = reconstruct_horizon(m, OracleHead(size), ctx, future.numpy(), CPU,
                              latent="output")
    assert out.shape == (Q, horizon, 1)
    for q in range(Q):
        assert np.allclose(out[q, :, 0], future.numpy(), atol=1e-3)


def test_r_reads_the_latent_that_the_caller_names(monkeypatch):
    """With ``latent='output'``, R reads the output latent of the context
    and the true horizon at the patch size of the head, and no encoder
    latent. By default it reads the encoder latent, as before."""
    m = moirai_model()
    head = bank_of().head_for(32)
    series = walk(T + 40, seed=3)
    ctx, future = series[:T, None], series[T:].numpy()
    default = reconstruct_horizon(m, head, ctx, future, CPU)
    encoder = reconstruct_horizon(m, head, ctx, future, CPU, latent="encoder")
    assert np.array_equal(default, encoder)
    seen = []
    real = fh.extract_forecaster_latents

    def spy(backbone, x, **kw):
        seen.append((x.shape[1], kw.get("patch_size"), kw.get("normalised")))
        return real(backbone, x, **kw)

    def no_encoder(*_, **__):
        raise AssertionError("this score reads the output latent only")

    monkeypatch.setattr(fh, "extract_forecaster_latents", spy)
    monkeypatch.setattr(fh, "extract_encoder_latents", no_encoder)
    output = reconstruct_horizon(m, head, ctx, future, CPU, latent="output")
    assert seen == [(T + 64, 32, True)]
    assert output.shape == default.shape and np.isfinite(output).all()
    assert not np.allclose(output, default)


# ---------------------------------------------------------------------------
# 3. The head trainer
# ---------------------------------------------------------------------------

def first_loss(trainer, tmp_path, name, model, batch):
    """The job of the head trainer on ``model``, and the loss of its first
    step on ``batch``."""
    bb = saved(tmp_path, f"{name}.pth", model)
    args = trainer.parse_args(["--backbone-path", bb, *HEAD_PROTOCOL,
                               "--save-dir", str(tmp_path / name),
                               "--run-name", name])
    job = trainer.HeadJob(args)
    job.start()
    job.train_step(1, tuple(t.clone() for t in batch))
    job.csv_logger.close()
    return job, job.ema_loss


def bank_batch():
    return (windows(), *labels_of())


def test_the_head_of_a_moirai_copy_trains_on_the_output_latent(tmp_path,
                                                               capsys):
    """Two copies of Moirai with the same encoder and another forecaster
    give the head another loss: the head reads the output of the
    transformer. The value heads of the checkpoint stay out of the frozen
    backbone."""
    trainer = load_head_trainer()
    job, a = first_loss(trainer, tmp_path, "a", moirai_model(), bank_batch())
    assert job.recon_latent == "output"
    assert job.backbone.value_heads is None
    assert "the output of the transformer" in capsys.readouterr().out
    _, same = first_loss(trainer, tmp_path, "same", moirai_model(),
                         bank_batch())
    _, b = first_loss(trainer, tmp_path, "b",
                      other_forecaster(moirai_model()), bank_batch())
    assert np.isfinite(a) and a == same
    assert abs(a - b) > 1e-6


def test_the_head_of_a_run_of_ours_trains_on_the_encoder_latent(tmp_path,
                                                                capsys):
    """Two runs of ours with the same encoder and another forecaster give
    the head the same loss: the head reads the encoder latent, as before."""
    trainer = load_head_trainer()
    job, a = first_loss(trainer, tmp_path, "a", bank_model(), bank_batch())
    assert job.recon_latent == "encoder"
    assert "the output of the transformer" not in capsys.readouterr().out
    _, b = first_loss(trainer, tmp_path, "b", other_forecaster(bank_model()),
                      bank_batch())
    assert np.isfinite(a) and a == b


def test_the_head_of_a_one_size_moirai_copy_trains_on_the_output_latent(
        tmp_path):
    """MOO: one patch size, the EWMA, no padding. Its head is one head, not
    a bank, and it reads the output latent too."""
    trainer = load_head_trainer()
    labels = torch.zeros(8, dtype=torch.long)
    batch = (windows(lengths=(T,) * 8), labels, labels)
    job, a = first_loss(trainer, tmp_path, "a", moirai_one_size_model(),
                        batch)
    assert job.recon_latent == "output" and not job.use_bank
    _, b = first_loss(trainer, tmp_path, "b",
                      other_forecaster(moirai_one_size_model()), batch)
    assert np.isfinite(a) and abs(a - b) > 1e-6


def test_a_wave_gives_a_moirai_job_the_steps_of_its_solo_run(tmp_path,
                                                             corpus_flags):
    """A copy of Moirai and a run of ours in one wave, on the stream of
    #419: each job writes the losses and the head of its solo run, to the
    last bit. The two jobs read one stream, and each one reads its own
    latent."""
    models = {"moirai": moirai_model(), "ours": bank_model()}
    argvs = []
    for name, model in models.items():
        bb = saved(tmp_path, f"{name}.pth", model)
        r = train_solo(job_argv(tmp_path / "solo", bb, name, *corpus_flags))
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        said = "the output of the transformer" in r.stdout
        assert said == (name == "moirai"), name
        argvs.append(job_argv(tmp_path / "shared", bb, name, *corpus_flags))
    r = train_shared(tmp_path, argvs)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for name in models:
        assert_same_run(tmp_path / "solo", tmp_path / "shared", name)


# ---------------------------------------------------------------------------
# 4. The eval script
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make,latent", [(moirai_model, "output"),
                                         (bank_model, "encoder")])
def test_the_eval_names_the_latent_of_the_checkpoint(tmp_path, monkeypatch,
                                                     make, latent):
    module = load_eval_module()
    bb = saved(tmp_path, "bb.pth", make())
    head = saved(tmp_path, "head.pth", bank_of())
    args = eval_args(module, monkeypatch, "--backbone-path", bb,
                     "--head-path", head, "--strategy", "R")
    backbone, bank = module.load_models(args, CPU)
    assert args.reconstruction_latent == latent
    assert isinstance(bank, ForecastingHeadBank)
    assert backbone.value_heads is None


def test_the_predictor_hands_its_latent_to_the_reconstruction(monkeypatch):
    import pandas as pd
    from gluonts.dataset.util import forecast_start
    module = load_eval_module()
    seen = []
    real = module.reconstruct_windows

    def spy(*args, **kw):
        seen.append(kw.get("latent"))
        return real(*args, **kw)

    monkeypatch.setattr(module, "reconstruct_windows", spy)
    m = moirai_model()
    item = {"target": walk(700).numpy(),
            "start": pd.Period("2020-01-01 00:00", freq="h")}
    label = {"target": walk(48, seed=7).numpy(), "start": forecast_start(item)}
    forecasts = {}
    for latent in ("output", "encoder"):
        predictor = module.ReconstructionPredictor(
            backbone=m, head=bank_of().head_for(32), prediction_length=48,
            device=CPU, strategy="R", context_pad="zeros", labels=[label],
            latent=latent)
        (forecast,) = list(predictor.predict([item]))
        forecasts[latent] = forecast.forecast_array
    assert seen == ["output", "encoder"]
    assert not np.allclose(forecasts["output"], forecasts["encoder"])


@pytest.mark.parametrize("make,latent", [(moirai_model, "output"),
                                         (bank_model, "encoder")])
def test_the_eval_scores_a_checkpoint_on_its_own_latent(tmp_path, monkeypatch,
                                                        make, latent):
    """The R score of the eval script: the predictor of each config gets the
    latent that the checkpoint names."""
    module = load_eval_module()
    bb = saved(tmp_path, "bb.pth", make())
    head = saved(tmp_path, "head.pth", bank_of())
    seen = []

    class Data:
        target_dim, freq, prediction_length = 1, "h", 24

        def __init__(self, **_):
            self.test_data = SimpleNamespace(label=[], input=[])

    def no_metrics(predictor, **_):
        seen.append((type(predictor).__name__, predictor.latent))
        raise RuntimeError("this test stops before the metrics")

    monkeypatch.setattr(module, "GiftDataset", Data)
    monkeypatch.setattr(module, "evaluate_model", no_metrics)
    monkeypatch.setattr(sys, "argv", [
        "eval", *EVAL_PROTOCOL, "--backbone-path", bb, "--head-path", head,
        "--strategy", "R", "--output-dir", str(tmp_path / "out"),
        "--config-filter", "^m4_hourly/short$"])
    module.main()
    assert seen == [("ReconstructionPredictor", latent)]


# ---------------------------------------------------------------------------
# 5. The proof on elisa
# ---------------------------------------------------------------------------

def per_config(path, rows):
    """A per-config table of the eval, with the two columns that the
    comparison reads."""
    path.write_text("dataset,model,eval_metrics/MASE[0.5]\n" + "".join(
        f"{config},x,{mase}\n" for config, mase in rows))
    return str(path)


def test_the_parity_script_compares_the_mase_of_two_tables(tmp_path):
    """The floor of a checkpoint on some configs, beside the floor table of
    a scaling setup: the same MASE on each config shows the same setup."""
    compare = load_script("parity_compare")
    floor = per_config(tmp_path / "floor.csv", [
        ("ett1/15T/long", 2.5), ("m4_hourly/H/short", 1.25),
        ("us_births/D/short", 0.75)])
    same = per_config(tmp_path / "same.csv", [
        ("m4_hourly/H/short", 1.25), ("ett1/15T/long", 2.5)])
    assert compare.compare_tables("MPM 100k", same, floor) == (
        "MPM 100k: 2 configs, 2 with the MASE of the second table, "
        "max relative difference 0")
    other = per_config(tmp_path / "other.csv", [
        ("m4_hourly/H/short", 1.25), ("ett1/15T/long", 2.0)])
    assert compare.compare_tables("x", other, floor) == (
        "x: 2 configs, 1 with the MASE of the second table, "
        "max relative difference 0.2")
    lost = per_config(tmp_path / "lost.csv", [("solar/H/short", 1.0)])
    with pytest.raises(KeyError):
        compare.compare_tables("x", lost, floor)


def test_the_parity_script_reads_tables_after_its_flag(tmp_path, capsys,
                                                       monkeypatch):
    compare = load_script("parity_compare")
    table = per_config(tmp_path / "a.csv", [("m4_hourly/H/short", 1.25)])
    monkeypatch.setattr(sys, "argv", ["parity_compare.py", "--tables", "MPM",
                                      table, table])
    compare.main()
    assert capsys.readouterr().out.strip() == (
        "MPM: 1 configs, 1 with the MASE of the second table, "
        "max relative difference 0")
