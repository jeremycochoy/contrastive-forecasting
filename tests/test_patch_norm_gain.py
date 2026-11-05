"""--patch-norm-gain-init: the start gain of the LayerNorm at the end of each
patch encoder sets the scale of the patch embedding e."""
import copy

import torch

from src.encoders import GRUEncoder, set_patch_norm_gain_


def two_encoders():
    """A GRU patch encoder and its EMA copy, as the student and the teacher."""
    torch.manual_seed(20261105)
    student = GRUEncoder(W=8, H=16, intermediate_dim=4, num_gru_layers=1)
    return torch.nn.ModuleDict({"student": student, "teacher": copy.deepcopy(student)})


def test_the_gain_is_the_scale_of_each_patch_embedding():
    encoders = two_encoders()
    x = torch.randn(2, 3, 1, 8)
    assert set_patch_norm_gain_(encoders, 0.25) == 2
    for encoder in encoders.values():
        e = encoder(x)
        std = e.std(dim=-1, unbiased=False)
        assert torch.allclose(std, torch.full_like(std, 0.25), atol=1e-3)


def test_the_gain_0_starts_the_patch_embedding_at_0():
    encoders = two_encoders()
    set_patch_norm_gain_(encoders, 0.0)
    x = torch.randn(2, 3, 1, 8)
    for encoder in encoders.values():
        assert torch.equal(encoder(x), torch.zeros(2, 3, 1, 16))
