"""--latent-sphere: e, h and f on one sphere of radius sqrt(K) after SIGReg has
read them. The encoder and the forecaster read only vectors on the sphere, also
in the EMA teacher, the rollout and a checkpoint loaded without the flag."""
import csv
import math

import pytest
import torch

from src.forecasting_head import extract_forecaster_latents, rollout_latent
from src.models import ConfigurableModel
from tests.test_mup import trainer

H = 16
CFG = dict(C=1, H=H, W=8, nhead=2, num_layers=2, num_encoder_layers=2,
           encoder_type="gru", dropout=0.0, rev_norm_kind="ewma", rev_norm_span=8)


def model(**over):
    torch.manual_seed(20261105)
    return ConfigurableModel(**dict(CFG, **over)).eval()


def encoder_input(net, seed=7):
    x = torch.randn(2, 64, 1, generator=torch.Generator().manual_seed(seed))
    return net.prepare_encoder_input(net.rev_norm(x, mode="norm"))


def norms(x):
    return x.float().norm(dim=-1)


def first_input(layer):
    """A hook list that keeps the input of each call of ``layer``."""
    seen = []
    layer.register_forward_pre_hook(lambda _, args, kwargs: seen.append(args[0]),
                                    with_kwargs=True)
    return seen


def test_the_default_keeps_the_model_unchanged():
    net = model()
    assert "transformer.latent_sphere" not in net.state_dict()
    x = torch.randn(3, H)
    assert net.transformer.sphere(x) is x


def test_the_encoder_and_the_forecaster_read_vectors_on_the_sphere():
    net = model(latent_sphere=True)
    enc_in = first_input(net.transformer.encoder_layers[0])
    fcst_in = first_input(net.transformer.layers[0])
    with torch.no_grad():
        f, h, e = net.transformer(encoder_input(net), return_embed=True)
    radius = torch.full_like(norms(f), math.sqrt(H))
    torch.testing.assert_close(norms(enc_in[0]), radius)
    torch.testing.assert_close(norms(fcst_in[0]), radius)
    torch.testing.assert_close(norms(f), radius)
    # SIGReg reads e and h before the projection: they keep their own norms.
    assert not torch.allclose(norms(e), torch.full_like(norms(e), math.sqrt(H)))
    assert not torch.allclose(norms(h), radius)


def test_the_forecaster_reads_only_the_direction_of_its_input():
    net = model(latent_sphere=True)
    x = torch.randn(2, 5, H)
    with torch.no_grad():
        torch.testing.assert_close(net.transformer.forecaster_forward(x),
                                   net.transformer.forecaster_forward(3.7 * x))


def test_the_rollout_feeds_back_vectors_on_the_sphere():
    net = model(latent_sphere=True)
    fcst_in = first_input(net.transformer.layers[0])
    with torch.no_grad():
        _, h = net.transformer(encoder_input(net))
        for feedback in ("raw", "normalised"):
            fcst_in.clear()
            rollout_latent(net, h, 3, feedback=feedback)
            for tokens in fcst_in:
                torch.testing.assert_close(
                    norms(tokens), torch.full_like(norms(tokens), math.sqrt(H)))


def test_the_teacher_encoder_reads_e_on_the_sphere():
    net = model(latent_sphere=True, ema_embedding=True, ema_encoder=True)
    teacher_in = first_input(net.teacher_encoder_layers[0])
    with torch.no_grad():
        net.teacher_forward(encoder_input(net))
    torch.testing.assert_close(norms(teacher_in[0]),
                               torch.full_like(norms(teacher_in[0]), math.sqrt(H)))


def test_a_checkpoint_gives_the_sphere_to_a_model_built_without_the_flag():
    trained = model(latent_sphere=True)
    loaded = model()
    loaded.load_state_dict(trained.state_dict())  # strict: the key is known
    x = torch.randn(2, 64, 1, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        want, _ = extract_forecaster_latents(trained, x)
        got, _ = extract_forecaster_latents(loaded, x)
    assert loaded.transformer._latent_sphere
    torch.testing.assert_close(got, want)


def test_a_cpc_forecaster_refuses_the_sphere():
    with pytest.raises(ValueError, match="latent-sphere"):
        model(latent_sphere=True, forecaster_kind="linear_cpc")


def test_the_flag_reaches_the_training(tmp_path):
    run = trainer(tmp_path, "sphere", 8, mup_base=8,
                  extra=["--latent-sphere", "--sigreg-embedding", "--sigreg-encoding",
                         "--sigreg-latent-scale", "inv_sqrt_d"])
    assert "Latent sphere: e, h and f at radius sqrt(8) = 2.82843" in run.stdout
    state = torch.load(tmp_path / "sphere_step20.pth", map_location="cpu", weights_only=False)
    weights = state.get("model_state_dict", state)
    assert "transformer.latent_sphere" in weights
    with open(tmp_path / "sphere_losses.csv") as f:
        losses = [float(row["loss"]) for row in csv.DictReader(f)]
    assert len(losses) == 20 and all(math.isfinite(v) for v in losses)
