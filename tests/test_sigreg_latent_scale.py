"""--sigreg-latent-scale: SIGReg reads each latent times a fixed factor, so the
latents of plain muP (coordinates of about 1) meet its target (variance 1/K)."""
import types

import torch

from src.loss import sigreg_loss
from tests.test_388_align_teacher_ema_schedule import load_train_module
from tests.test_mup import trainer


def test_the_factor_1_over_sqrt_k_puts_unit_latents_at_the_target():
    torch.manual_seed(20261105)
    z = torch.randn(4096, 16)  # coordinates of variance 1, as plain muP latents
    wide = sigreg_loss(z, M=256).item()
    at_target = sigreg_loss(z / 4.0, M=256).item()  # 1/sqrt(16)
    assert at_target < 0.1 < wide


def test_inv_sqrt_d_is_1_over_sqrt_of_the_latent_size():
    train = load_train_module()
    assert train.sigreg_scale(types.SimpleNamespace(sigreg_latent_scale="inv_sqrt_d"), 64) == 0.125
    assert train.sigreg_scale(types.SimpleNamespace(sigreg_latent_scale=0.5), 64) == 0.5
    assert train.sigreg_scale_arg("inv_sqrt_d") == "inv_sqrt_d"
    assert train.sigreg_scale_arg("0.25") == 0.25


def test_the_flag_reaches_the_training(tmp_path):
    run = trainer(tmp_path, "scale", 8,
                  extra=["--sigreg-embedding", "--sigreg-encoding", "--sigreg-latent-scale", "inv_sqrt_d"])
    assert "SIGReg latent scale: inv_sqrt_d (0.353553 at d_model 8)" in run.stdout
