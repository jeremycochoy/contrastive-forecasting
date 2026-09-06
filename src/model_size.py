"""How many parameters a contrastive backbone has, and which width to train.

One `ConfigurableModel` gives three different numbers. A card that compares
this project against a published model must use the first of them.

- The TRAINABLE parameters. This is the number a paper gives.
- The FROZEN parameters. The EMA teacher is a copy of the patch encoder and of
  the encoder stack. No gradient touches it, and it does no work at test time.
- The SUM over the checkpoint file. This is larger than the two together,
  because the patch encoder is one module under two names, `encoder.*` and
  `transformer.input_to_latent.*`. The sum counts that module two times.

At the published width, `d_model` 64, the three numbers are 720,668 trainable,
563,760 frozen and 1,699,534 in the file. A reader who drops the `teacher_*`
keys from the file gets 1,135,774, which is neither the size of the model nor
the size of the teacher.

`parameter_counts` gives the three numbers from a model. `state_dict_counts`
gives the file number from a checkpoint, and splits the teacher from the
student the same way a reader of that file does. `backbone_counts` builds this
project's backbone at a given shape and counts it, which is what a card that
must reach a target size asks for.
"""

from __future__ import annotations

from .models import ConfigurableModel

# The shape of every cell of the rollout-depth family (#373, #393, #401, #404,
# #409), from `reports/2026-08-08_rollout_depth/scripts/run_leg_k.sh`. Only the
# three shape keys `H`, `num_layers` and `num_encoder_layers` change between
# the cards, so they have no value here and each caller gives them.
#
# `tests/test_412_moirai_small_size.py` holds this against that runner.
CELL_BACKBONE = dict(
    C=1, W=16, encoder_type="gru", nhead=8, ffn_mult=4.0, activation="gelu",
    depthwise_conv=3, dropout=0.1,
    freq_emb_dim=3, seasonality_emb_dim=3,
    rev_norm_kind="ewma", rev_norm_span=128,
    encoder_dropkey=0.70, encoder_dropkey_share_heads=True,
    encoder_dropkey_share_layers=True,
    qk_norm=True, attn_out_norm=True,
    ema_embedding=True, ema_encoder=True,
)


def parameter_counts(model) -> dict[str, int]:
    """The three counts of one model, as a dictionary.

    Keys: `trainable`, `frozen`, `total` and `state_dict`. See the module
    docstring for why the last one is larger than the sum of the first two.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": trainable + frozen,
        "state_dict": sum(t.numel() for t in model.state_dict().values()),
    }


def state_dict_counts(state_dict, teacher_prefix: str = "teacher_") -> dict[str, int]:
    """Count a checkpoint file, and split the teacher from the student.

    This is the count a reader gets from the file alone. It is not the
    parameter count of the model: the patch encoder is in the file two times.
    """
    total = 0
    teacher = 0
    for key, tensor in state_dict.items():
        if not hasattr(tensor, "numel"):
            continue
        total += tensor.numel()
        if key.startswith(teacher_prefix):
            teacher += tensor.numel()
    return {"total": total, "teacher": teacher, "student": total - teacher}


def build_backbone(**overrides) -> ConfigurableModel:
    """Build this project's backbone at one shape, on the CPU.

    Give the shape as `H`, `num_layers` and `num_encoder_layers`. Any other
    key replaces the value in `CELL_BACKBONE`.
    """
    config = dict(CELL_BACKBONE)
    config.update(overrides)
    return ConfigurableModel(**config)


def backbone_counts(**overrides) -> dict[str, int]:
    """The counts of this project's backbone at one shape, with the shape.

    The result holds the keys of `parameter_counts`, and also `H`,
    `num_layers` and `num_encoder_layers`, so a table of shapes reads in one
    pass.
    """
    model = build_backbone(**overrides)
    counts = parameter_counts(model)
    config = dict(CELL_BACKBONE)
    config.update(overrides)
    for key in ("H", "num_layers", "num_encoder_layers"):
        counts[key] = config[key]
    return counts


def nearest_to_target(rows, target: int, key: str = "trainable"):
    """The row whose count is nearest to `target`.

    `rows` is a list of the dictionaries `backbone_counts` gives. Raises
    `ValueError` on an empty list, because "no shape" is not an answer a
    caller can train.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("nearest_to_target needs at least one row")
    return min(rows, key=lambda row: abs(row[key] - target))
