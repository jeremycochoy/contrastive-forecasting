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
6. The scripts of the report: the job table, the plan of a new start of a
   queue, the floors, the table of the scores and the figures.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.forecasting_head as fh
from src.checkpoint import reconstruction_latent_of
from src.forecasting_head import (ForecastingHeadBank,
                                  ZeroReconstructionHead, bank_quantile_loss,
                                  bank_training_inputs,
                                  extract_encoder_latents,
                                  extract_forecaster_latents,
                                  extract_reconstruction_latents,
                                  reconstruct_horizon, reconstruct_windows,
                                  reconstruction_quantile_loss,
                                  value_space_forward)
from src.freq_embedding import FREQ_NAMES_V2
from src.models import ConfigurableModel
from tests import test_425_encoder_reconstruction as base
from tests import test_425_linear_head as linear
from tests.test_425_encoder_reconstruction import (  # noqa: F401
    CPU, EVAL_PROTOCOL, HEAD_PROTOCOL, Q, SIZES, T, OracleHead,
    assert_same_run, bank_model, bank_of, corpus_flags, eval_args,
    ewma_bank_model, ewma_model, job_argv, labels_of, load_eval_module,
    load_head_trainer, load_script, meanstd_model, oracle_latents, saved,
    train_shared, train_solo, walk, windows)


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


@pytest.mark.parametrize("scaling,ours", [("meanstd", meanstd_model),
                                          ("ewma", ewma_bank_model)])
def test_the_floor_of_a_moirai_copy_is_the_floor_of_its_scaling_setup(
        scaling, ours):
    """The floor of R reads the scaling and the padding of a run, and no
    latent and no weight. So MPM has the floor of the mean/std runs of ours
    (BMS), and MPE has the floor of the EWMA runs with zero padding (OEF),
    with each of the two latents."""
    m = moirai_model(scaling)
    ctx, future = walk(T)[:, None], walk(100, seed=6)[:, None]
    want = reconstruct_windows(ours(), ZeroReconstructionHead(16), ctx[None],
                               future[None], CPU)
    for size in (16, 64):
        for latent in ("output", "encoder"):
            got = reconstruct_windows(m, ZeroReconstructionHead(size),
                                      ctx[None], future[None], CPU,
                                      latent=latent)
            assert np.allclose(got, want, rtol=1e-6, atol=1e-4), (size, latent)


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


# ---------------------------------------------------------------------------
# 6. The scripts of the report
# ---------------------------------------------------------------------------

# The two copies of Moirai of the card: the code of each run, its arm (the
# #412 run key) and its folder in the mirror of the box.
MOIRAI = {"MPM": ("cf421f_moirai_native", "cf-421f"),
          "MPE": ("cf421ew_moirai_native", "cf-421ew")}
GM_412 = (base.REPO_ROOT / "reports" / "2026-09-06_moirai_small_size"
          / "results" / "gm_trajectories.tsv")


def forecast_stops(arm):
    """The stops of a run with a forecast score in the #412 table."""
    rows = (row.split() for row in open(GM_412))
    return sorted(int(stop_k) for name, stop_k, _ in rows if name == arm)


def test_the_job_table_holds_each_scored_stop_of_the_two_moirai_copies():
    """One job for each stop of MPM and of MPE with a forecast score, on the
    stream of the run. The checkpoint of MPM 20k is the file that its
    forecast score read, in the folder of the third try of its leg."""
    rows = base.job_rows()
    for code, (arm, folder) in MOIRAI.items():
        mine = [row for row in rows if row[0] == code]
        assert {row[1] for row in mine} == {arm}
        assert [int(row[2]) for row in mine] == forecast_stops(arm)
        assert all(row[4].startswith(f"{folder}/value_space/leg_")
                   for row in mine)
        assert {row[6] for row in mine} == {"gift_pretrain"}
    assert sum(row[0] in MOIRAI for row in rows) == 17
    (ckpt,) = [row[4] for row in rows if row[0] == "MPM" and row[2] == "20"]
    assert ckpt == "cf-421f/value_space/leg_25k_try3/cf421f_moirai_k3_20k.pth"


def score_each_job_of_ours(res, suffix):
    """A score file for each job of the real job table but those of the two
    copies of Moirai: the state of a queue when the card got these runs."""
    res.mkdir(parents=True, exist_ok=True)
    for code, arm, stop_k, *_ in base.job_rows():
        if code not in MOIRAI:
            (res / f"score_{arm}_bb{stop_k}k_h30k_{suffix}.txt").write_text(
                "0.5000\n")


@pytest.mark.parametrize("arch,suffix,waves", [
    ("transformer", "recon", [12, 5]), ("linear", "recon_lin", [17])])
def test_a_new_start_trains_the_moirai_jobs_and_no_job_with_a_score(
        tmp_path, arch, suffix, waves):
    """Each job of ours has its score. So a new start of a queue trains the
    17 Moirai jobs and no other job, in GiftEvalPretrain waves, with the
    first and the last stop of each run first."""
    res = tmp_path / "res"
    score_each_job_of_ours(res, suffix)
    plan = linear.plan_of(dict(linear.real_job_box(tmp_path),
                               CF425_HEAD_ARCH=arch), res)
    assert len(plan) == 17
    assert {p[0] for p in plan} == {"gift_pretrain"}
    assert {p[3] for p in plan} == set(MOIRAI)
    assert all(p[5].endswith(f"_moirai_native_bb{p[4]}_h30k_{suffix}")
               for p in plan)
    assert [(p[3], p[4]) for p in plan[:4]] == [
        ("MPM", "10k"), ("MPM", "166k"), ("MPE", "10k"), ("MPE", "166k")]
    in_wave = [int(p[1]) for p in plan]
    assert in_wave == sorted(in_wave)
    assert [in_wave.count(wave) for wave in sorted(set(in_wave))] == waves


def test_the_floor_table_gives_each_moirai_copy_the_floor_of_its_setup():
    """MPM has the mean/std scaling and MPE the EWMA, each with zero
    padding: the setups of two floors that exist. floors.sh names the two
    runs in those setups, and floors.tsv gives each run its floor."""
    rows = [line.rstrip("\n").split("\t")
            for line in open(base.STUDY / "results" / "floors.tsv")][1:]
    arms = {setup: listed.split(",") for setup, _, listed, _ in rows}
    assert MOIRAI["MPM"][0] in arms["meanstd"]
    assert MOIRAI["MPE"][0] in arms["ewma_zero_pad"]
    setups = (base.SCRIPTS / "floors.sh").read_text().split('SETUPS="')[1]
    runs = {line.split()[0]: line.split()[3].split(",")
            for line in setups.split('"')[0].strip().splitlines()}
    assert "MPM" in runs["meanstd"] and "MPE" in runs["ewma_zero_pad"]


def test_the_score_table_names_the_forecast_of_each_run(tmp_path,
                                                        monkeypatch):
    """The forecast of a copy of Moirai is the forecast of its own heads
    (#412), not B4. So the column of the forecast scores has the name
    ``forecast``, and a Moirai job has its #412 score there."""
    collect = base.load_script("collect")
    monkeypatch.setattr(collect, "RESULTS", tmp_path)
    assert collect.JOB_COLUMNS[2] == "forecast"
    scores = {(row[0], row[1]): row[2] for row in collect.job_scores()}
    assert scores["MPM", "166"] == "0.9250"
    assert scores["MPE", "10"] == "1.0247"
    assert scores["OMB", "10"] == "1.3782"


def moirai_scores(tmp_path):
    """R scores of the two copies of Moirai and of one run of ours, and the
    forecast scores of #412."""
    plot = base.load_script("plot_recon")
    table = tmp_path / "recon.tsv"
    table.write_text(
        "cf421f_moirai_native\t10\t0.4100\ncf421f_moirai_native\t166\t0.3000\n"
        "cf421ew_moirai_native\t10\t0.0900\ncf421ew_moirai_native\t166\t0.0800\n"
        "cf412om\t10\t0.3100\ncf412om\t25\t0.2900\n")
    return plot, plot.load([table]), plot.load(plot.FORECAST[:1])


def key_of(fig):
    """The entries of the key of a figure, under its name."""
    (key,) = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    return [text.get_text() for text in key.get_texts()][1:]


def test_the_moirai_figure_draws_the_two_copies_as_the_412_report(tmp_path):
    """The Moirai graph of #412, with the R scores: MPM and MPE in their
    colour, with the dashed line of a copy of Moirai, and each step at
    batch 256 counts 4. The run of ours is not in this graph."""
    pytest.importorskip("matplotlib")
    plot, points, forecast = moirai_scores(tmp_path)
    assert points["cf421f_moirai_native"] == {40000: 0.41, 664000: 0.30}
    fig = plot.draw_figure(plot.GRAPHS["moirai"], forecast, points,
                           tmp_path / "m.png", False)
    assert (tmp_path / "m.png").stat().st_size > 10_000
    (ax,) = fig.axes
    for arm, scores in (("cf421f_moirai_native", [0.41, 0.30]),
                        ("cf421ew_moirai_native", [0.09, 0.08])):
        (curve,) = [line for line in ax.get_lines()
                    if line.get_color() == plot.colour(arm)]
        assert list(curve.get_ydata()) == scores
        assert curve.get_linestyle() == "--"
    texts = base.legend_texts(fig)
    assert any("MPM" in t and t.endswith("R 0.4100 → 0.3000, ×0.73")
               for t in texts)
    assert any("MPE" in t and t.endswith("R 0.0900 → 0.0800, ×0.89")
               for t in texts)
    assert not any("OMB" in t for t in texts)
    assert ax.get_xlabel().endswith("A step at batch 256 (MPM and MPE) "
                                    "counts 4.")


def test_the_key_names_the_latent_and_the_forecast_of_a_moirai_copy(tmp_path):
    """The report holds the figures only. So the key of a figure with a copy
    of Moirai says that its R reads the output of the transformer, and that
    its forecast is the forecast of its own heads, not B4. Each entry is one
    line of 52 characters or less. A figure with no copy of Moirai keeps
    its key."""
    pytest.importorskip("matplotlib")
    plot, points, forecast = moirai_scores(tmp_path)
    latent = "MPM, MPE: R reads the output of the transformer"
    own = "MPM, MPE: the forecast of their own heads"
    # The Moirai graph: no run of ours, so no B4.
    fig = plot.draw_figure(plot.GRAPHS["moirai"], forecast, points,
                           tmp_path / "m.png", True)
    key = key_of(fig)
    assert latent in key and own in key
    assert not any("B4" in label for label in key)
    assert "One checkpoint: its forecast and its R" in key
    top, _ = fig.axes
    assert top.get_ylabel() == "Forecast (own heads)"
    # A graph with runs of ours and copies of Moirai: the two forecasts.
    fig = plot.draw_figure(plot.GRAPHS["all"], forecast, points,
                           tmp_path / "a.png", True)
    key = key_of(fig)
    assert latent in key and own in key
    assert "B4: the forecast of the run" in key
    assert key.index(own) == key.index("B4: the forecast of the run") + 1
    assert fig.axes[0].get_ylabel() == "Forecast"
    # With no forecast panel, the key names the latent only.
    fig = plot.draw_figure(plot.GRAPHS["all"], forecast, points,
                           tmp_path / "r.png", False)
    key = key_of(fig)
    assert latent in key and own not in key
    # A graph of ours keeps its key.
    fig = plot.draw_figure(plot.GRAPHS["ours_patch_sizes"], forecast, points,
                           tmp_path / "o.png", True)
    key = key_of(fig)
    assert not any("MPM" in label for label in key)
    assert "B4: the forecast of the run" in key
    assert "One checkpoint: its B4 and its R" in key
    assert fig.axes[0].get_ylabel() == "Forecast (B4)"
    for name in ("moirai", "all"):
        for overlay in (False, True):
            fig = plot.draw_figure(plot.GRAPHS[name], forecast, points,
                                   tmp_path / "k.png", overlay)
            assert all("\n" not in label and len(label) <= 52
                       for label in key_of(fig))


def test_one_moirai_copy_in_a_figure_has_its_own_key_lines(tmp_path):
    pytest.importorskip("matplotlib")
    plot, points, forecast = moirai_scores(tmp_path)
    del points["cf421ew_moirai_native"]
    fig = plot.draw_figure(plot.GRAPHS["moirai"], forecast, points,
                           tmp_path / "m.png", True)
    key = key_of(fig)
    assert "MPM: R reads the output of the transformer" in key
    assert "MPM: the forecast of its own heads" in key


def test_the_title_of_the_moirai_overlay_names_no_b4(tmp_path, monkeypatch):
    """main() gives each overlay its title. The Moirai graph holds no B4
    score, so its title says forecast."""
    pytest.importorskip("matplotlib")
    plot, points, forecast = moirai_scores(tmp_path)
    titles = {}

    def record(groups, forecast, recon, out, overlay, floors=(), title=None,
               *args, **kwargs):
        titles[out.name] = title

    monkeypatch.setattr(plot, "draw_figure", record)
    monkeypatch.setattr(plot, "PLOTS", tmp_path / "plots")
    plot.main()
    assert titles["overlay_moirai.png"] == (
        "R and forecast, transformer head: Moirai, our copy")
    assert titles["overlay_moirai_linear.png"] == (
        "R and forecast, linear head: Moirai, our copy")
    assert titles["overlay_all.png"] == "R and B4, transformer head: all runs"
    assert titles["recon_moirai.png"] == (
        "R, transformer head: Moirai, our copy")


def test_the_legend_names_mpe_in_its_floor(tmp_path):
    """MPE has the floor of the EWMA runs with zero padding, as BLK and
    OEF. The legend of a figure with MPE names the three runs. A figure
    with no MPE keeps the name of its floor."""
    pytest.importorskip("matplotlib")
    plot, points, forecast = moirai_scores(tmp_path)
    floors = plot.load_floors(plot.FLOORS)
    (floor,) = [f for f in floors if f["setup"] == "ewma_zero_pad"]
    assert plot.floor_legend(floor) == "EWMA, new data (BLK, OEF): 1.2092"
    assert plot.floor_legend(floor, ["MPM"]) == (
        "EWMA, new data (BLK, OEF): 1.2092")
    assert plot.floor_legend(floor, ["MPM", "MPE"]) == (
        "EWMA, new data (BLK, OEF, MPE): 1.2092")
    fig = plot.draw_figure(plot.GRAPHS["moirai"], forecast, points,
                           tmp_path / "m.png", False, floors)
    key = key_of(fig)
    assert "EWMA, new data (BLK, OEF, MPE): 1.2092" in key
    assert "mean/std: 1.5721" in key
    assert not any("old data" in label for label in key)
    fig = plot.draw_figure(plot.GRAPHS["ours_patch_sizes"], forecast,
                           {"cf412oe2": {40000: 0.09}}, tmp_path / "o.png",
                           False, floors)
    assert "EWMA, new data (BLK, OEF): 1.2092" in key_of(fig)
