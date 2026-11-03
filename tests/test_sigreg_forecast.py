"""--sigreg-forecast: SIGReg on the forecaster output f, the one-step forecast
latent (the user, 11-03).

The term is the statistic of --sigreg-embedding and --sigreg-encoding
(src.loss.sigreg_loss, the window of docs/sigreg.md) on f at the real
positions. The loss adds --sigreg-forecast-weight times it. The losses CSV
gets a trailing column ``sigreg_f`` only when the flag is on, so the CSV of
every other run keeps its columns.
"""
import csv
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "experiments" / "2026-04-27_freq-embedding" / "scripts" / "train.py"
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def train():
    spec = importlib.util.spec_from_file_location("train_sigreg_forecast", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse(train, *flags):
    return train.parse_args(["--weight-decay", "0.1", *flags])


def latents():
    torch.manual_seed(0)
    return SimpleNamespace(f=torch.randn(4, 6, 1, 8), o=torch.randn(4, 6, 1, 8),
                           teacher=None, e=torch.randn(4, 6, 1, 8), rollout=None)


def fake_sigreg(lat):
    """3.0 on f and 5.0 on any other latent."""
    return lambda latent, args, pad: torch.tensor(3.0 if latent is lat.f else 5.0)


def single_size_terms(train, monkeypatch, args):
    lat = latents()
    monkeypatch.setattr(train, "main_contrastive_term",
                        lambda lat, args, rep_w, tau, pad, terms: lat.o.new_zeros(()))
    monkeypatch.setattr(train, "align_moco_term", lambda lat, args, tau: None)
    monkeypatch.setattr(train, "cpc_aux_term", lambda model, lat, args, pad: None)
    monkeypatch.setattr(train, "sigreg_term", fake_sigreg(lat))
    return train.contrastive_terms(None, lat, args, 0.0, None, None)


def test_flags_are_off_by_default(train):
    args = parse(train)
    assert args.sigreg_forecast is False
    assert args.sigreg_forecast_weight == pytest.approx(0.1)


def test_the_term_reads_the_forecaster_output(train, monkeypatch):
    args = parse(train, "--sigreg-forecast", "--sigreg-forecast-weight", "0.5")
    loss, values = single_size_terms(train, monkeypatch, args)
    assert loss.item() == pytest.approx(0.5 * 3.0)
    assert values["sigreg_f"] == pytest.approx(3.0)


def test_no_term_without_the_flag(train, monkeypatch):
    loss, values = single_size_terms(train, monkeypatch, parse(train))
    assert loss.item() == 0.0
    assert math.isnan(values["sigreg_f"])


def test_the_patch_size_groups_add_the_term_on_the_common_grid(train, monkeypatch):
    lat = latents()
    grid = SimpleNamespace(f=lat.f, o=lat.o, e=lat.e, teacher=None, pad=None, token=None)
    monkeypatch.setattr(train, "common_grid", lambda lats, pads: grid)
    monkeypatch.setattr(train, "grid_main_terms", lambda *a: lat.o.new_zeros(()))
    monkeypatch.setattr(train, "group_align_terms", lambda *a: lat.o.new_zeros(()))
    monkeypatch.setattr(train, "sigreg_term", fake_sigreg(lat))
    args = parse(train, "--sigreg-forecast", "--sigreg-forecast-weight", "2.0")
    loss, values = train.multi_size_contrastive_terms(
        None, {16: lat}, {16: None}, {16: 4}, args, 0.0, None)
    assert loss.item() == pytest.approx(2.0 * 3.0)
    assert values["sigreg_f"] == pytest.approx(3.0)


def logged_rows(train, tmp_path, **options):
    path = tmp_path / "losses.csv"
    logger = train.CSVLogger(str(path), flush_every=1, **options)
    logger.log(1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, False,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sigreg_f=0.25)
    logger.close()
    return list(csv.reader(open(path)))


def test_the_csv_gets_the_column_with_the_flag(train, tmp_path):
    header, row = logged_rows(train, tmp_path, sigreg_f_column=True)
    assert header[-1] == "sigreg_f"
    assert float(row[-1]) == pytest.approx(0.25)


def test_the_csv_keeps_its_columns_without_the_flag(train, tmp_path):
    header, row = logged_rows(train, tmp_path)
    assert "sigreg_f" not in header
    assert len(row) == len(header)


def run_trainer(tmp_path, *flags):
    from tests.test_412_ours_moirai import CYAN_EWMA, corpus, run
    return run(REPO_ROOT, tmp_path / "run", *CYAN_EWMA, *flags, *corpus(tmp_path))


def test_a_negative_weight_stops_the_run(tmp_path):
    result = run_trainer(tmp_path, "--sigreg-forecast", "--sigreg-forecast-weight", "-1")
    assert result.returncode != 0
    assert "--sigreg-forecast-weight" in result.stdout + result.stderr


def test_a_run_logs_the_term(tmp_path):
    result = run_trainer(tmp_path, "--sigreg-forecast", "--sigreg-forecast-weight", "1.0")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    rows = list(csv.DictReader(open(tmp_path / "run" / "r_losses.csv")))
    assert rows and all(math.isfinite(float(r["sigreg_f"])) for r in rows)
