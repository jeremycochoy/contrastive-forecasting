"""
Training checkpoint utilities for saving/loading optimizer state,
step counter, and best-tracking metadata alongside model checkpoints.

The model checkpoint format is NOT changed. Optimizer state is saved
in a companion file (model.pth -> model_optimizer.pth).

Also exposes :func:`load_backbone_from_checkpoint` — a state-dict-driven
backbone loader that autodetects the architecture flags visible in the
weights (freq / seasonality embeddings, encoder-layer count, QK-norm,
attn-out RMSNorm, CPC forecaster kind, learnable τ, patch-stats width)
so downstream consumers (offline drift probe, ad-hoc inference scripts)
don't have to re-derive the arch from CLI args.
"""

import os
import re
import torch


_DEFAULTS = {
    "step": 0,
    "best_val_ff": float("-inf"),
    "best_step": 0,
    "best_loss": float("inf"),
    "best_loss_step": 0,
    "ema_loss": None,
    "ema_gap": None,
    "hf_rows_consumed": 0,
    "synth_rows_consumed": 0,
    "rng_state_torch": None,
    "rng_state_numpy": None,
}


def get_optimizer_state_path(model_path: str) -> str:
    """Derive optimizer state file path from model checkpoint path."""
    root, ext = os.path.splitext(model_path)
    return f"{root}_optimizer{ext}"


def save_training_state(optimizer, model_path: str, step: int,
                        best_val_ff: float, best_step: int, *,
                        best_loss: float = float("inf"),
                        best_loss_step: int = 0,
                        ema_loss=None, ema_gap=None,
                        hf_rows_consumed: int = 0,
                        synth_rows_consumed: int = 0,
                        rng_state_torch=None,
                        rng_state_numpy=None) -> str:
    """Save optimizer state and training metadata to companion file.

    Returns the path where the state was saved.
    """
    state = {
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_ff": best_val_ff,
        "best_step": best_step,
        "best_loss": best_loss,
        "best_loss_step": best_loss_step,
        "ema_loss": ema_loss,
        "ema_gap": ema_gap,
        "hf_rows_consumed": hf_rows_consumed,
        "synth_rows_consumed": synth_rows_consumed,
        "rng_state_torch": rng_state_torch,
        "rng_state_numpy": rng_state_numpy,
    }
    optim_path = get_optimizer_state_path(model_path)
    torch.save(state, optim_path)
    return optim_path


def safe_save_path(save_path: str, resume_path: str) -> str:
    """Ensure save_path won't overwrite the resume checkpoint or its companions.

    If save_path would conflict with resume_path (same base name), appends
    a _runN suffix to branch out. This prevents accidental overwriting of
    trained checkpoints when resuming training.

    Returns a safe save_path (unchanged if no conflict, suffixed if conflict).
    """
    if resume_path is None:
        return save_path

    # Normalize paths for comparison
    save_dir = os.path.dirname(os.path.abspath(save_path))
    resume_dir = os.path.dirname(os.path.abspath(resume_path))
    save_base = os.path.splitext(os.path.basename(save_path))[0]
    resume_base = os.path.splitext(os.path.basename(resume_path))[0]
    save_ext = os.path.splitext(save_path)[1]

    # Check if save would conflict with resume or its companions (_best, _optimizer)
    resume_bases = {resume_base, resume_base.replace("_best", ""),
                    resume_base.replace("_optimizer", "")}
    save_bases = {save_base, save_base + "_best", save_base + "_optimizer"}

    if save_dir == resume_dir and (resume_bases & save_bases):
        # Conflict detected — find next available _runN suffix
        parent = os.path.dirname(save_path)
        n = 1
        while True:
            candidate = os.path.join(parent, f"{save_base}_run{n}{save_ext}")
            if not os.path.exists(candidate):
                print(f"  [checkpoint] WARNING: save_path '{save_path}' would "
                      f"conflict with resume checkpoint.")
                print(f"  [checkpoint] Branching to: {candidate}")
                return candidate
            n += 1

    return save_path


def load_training_state(optimizer, model_path: str, device=None) -> dict:
    """Load optimizer state from companion file. Graceful fallback if missing.

    Returns dict with keys: step, best_val_ff, best_step.
    """
    optim_path = get_optimizer_state_path(model_path)

    if not os.path.exists(optim_path):
        print(f"  [checkpoint] No optimizer state at {optim_path}, "
              f"starting fresh.")
        return dict(_DEFAULTS)

    try:
        state = torch.load(optim_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"  [checkpoint] Restored optimizer from {optim_path} "
              f"(step={state['step']}, best_ff={state['best_val_ff']:.4f})")
        # Return all fields, falling back to defaults for old checkpoints
        result = dict(_DEFAULTS)
        for key in _DEFAULTS:
            if key in state:
                result[key] = state[key]
        return result
    except Exception as e:
        print(f"  [checkpoint] WARNING: Failed to load {optim_path}: {e}")
        print(f"  [checkpoint] Continuing with fresh optimizer.")
        return dict(_DEFAULTS)


def _detect_backbone_config(sd: dict, base_cfg: dict) -> dict:
    """Fill in ConfigurableModel kwargs from a state_dict.

    Mirrors the autodetect block in ``train_forecasting_head.py`` so both
    the head trainer and out-of-loop consumers (e.g. the offline latent-
    drift probe) build a backbone that strictly matches the checkpoint.
    ``base_cfg`` supplies the fields the state_dict cannot disambiguate:
    ``C``, ``H``, ``W``, ``nhead``, ``num_layers``, ``encoder_type``,
    ``ffn_mult``, ``activation``, ``depthwise_conv``, ``dropout``,
    ``rev_norm_kind``, ``rev_norm_span``. Autodetected fields overwrite
    any value the caller placed in ``base_cfg``.
    """
    from src.norm import PATCH_STATS_DIM
    cfg = dict(base_cfg)
    w = sd.get("freq_embedding.embedding.weight")
    cfg["freq_emb_dim"] = int(w.shape[1]) if w is not None else 0
    w = sd.get("seasonality_embedding.embedding.weight")
    cfg["seasonality_emb_dim"] = int(w.shape[1]) if w is not None else 0
    if "log_inv_tau" in sd:
        cfg["learnable_tau"] = True
    enc_layer_idxs = set()
    for k in sd:
        if k.startswith("transformer.encoder_layers."):
            try:
                enc_layer_idxs.add(int(k.split(".")[2]))
            except (IndexError, ValueError):
                continue
    if enc_layer_idxs:
        cfg["num_encoder_layers"] = max(enc_layer_idxs) + 1
    if any(k.endswith(".q_norm.weight") for k in sd):
        cfg["qk_norm"] = True
    if any(k.endswith(".attn_out_rms.weight") for k in sd):
        cfg["attn_out_norm"] = True
    lin_idxs = set()
    for k in sd:
        if k.startswith("transformer.cpc_heads."):
            try:
                lin_idxs.add(int(k.split(".")[2]))
            except (IndexError, ValueError):
                continue
    if lin_idxs:
        cfg["forecaster_kind"] = "linear_cpc"
        cfg["cpc_k_steps"] = max(lin_idxs) + 1
    cpc_head_idxs = set()
    for k in sd:
        if k.startswith("transformer.cpc_layers."):
            try:
                cpc_head_idxs.add(int(k.split(".")[2]))
            except (IndexError, ValueError):
                continue
    if cpc_head_idxs:
        cfg["forecaster_kind"] = "cpc"
        cfg["cpc_k_steps"] = max(cpc_head_idxs) + 1
        wc = sd.get("transformer.cpc_down.0.weight")
        if wc is not None:
            cfg["forecaster_d_model"] = int(wc.shape[0])
    # patch_stats width: encoder in-features = W + freq_emb + seasonality_emb
    # + (PATCH_STATS_DIM if patch_stats else 0). GRU encoder puts it in
    # ``encoder.skip.weight``; MLP-style in ``encoder.linear1.weight``.
    W = cfg["W"]
    ref = sd.get("encoder.skip.weight")
    if ref is None:
        ref = sd.get("encoder.linear1.weight")
    if ref is None:
        cfg["patch_stats_kind"] = "none"
    else:
        extra = (int(ref.shape[1]) - W
                 - cfg["freq_emb_dim"] - cfg["seasonality_emb_dim"])
        if extra == 0:
            cfg["patch_stats_kind"] = "none"
        elif extra == PATCH_STATS_DIM:
            cfg["patch_stats_kind"] = "diff"
        else:
            raise ValueError(
                f"Backbone loader: encoder in_features={int(ref.shape[1])} "
                f"leaves extra width={extra}, which doesn't match W ({W}) + "
                f"freq_emb_dim ({cfg['freq_emb_dim']}) + seasonality_emb_dim "
                f"({cfg['seasonality_emb_dim']}) + 0 or {PATCH_STATS_DIM}.")
    return cfg


def load_backbone_from_checkpoint(
    checkpoint_path: str,
    device,
    *,
    C: int = 1,
    H: int = 384,
    W: int = 16,
    nhead: int = 6,
    num_layers: int = 6,
    encoder_type: str = "gru",
    rev_norm_kind: str = "ewma",
    rev_norm_span: int = 128,
    ffn_mult: float = 4.0,
    activation: str = "gelu",
    depthwise_conv: int = 3,
    dropout: float = 0.1,
    verbose: bool = False,
):
    """Load a frozen ``ConfigurableModel`` from a backbone checkpoint,
    autodetecting architecture flags visible in the state_dict.

    Fields the caller MUST match to the training-time backbone
    (state_dict does not disambiguate): ``C``, ``H``, ``W``, ``nhead``,
    ``num_layers``, ``encoder_type``, ``rev_norm_kind``,
    ``rev_norm_span``, and the transformer FFN / activation / dropout
    knobs. Fields autodetected from the state_dict: ``freq_emb_dim``,
    ``seasonality_emb_dim``, ``num_encoder_layers``, ``qk_norm``,
    ``attn_out_norm``, ``forecaster_kind`` (transformer / cpc /
    linear_cpc) with ``cpc_k_steps`` and ``forecaster_d_model``,
    ``learnable_tau``, ``patch_stats_kind``.

    Non-load state_dict keys (``cpc_w1.*`` from the CPC-InfoNCE
    auxiliary, ``teacher_*`` from the EMA-target teacher — training-only
    branches with no downstream role) are stripped so ``load_state_dict``
    with the default ``strict=True`` still succeeds.

    Returns:
        (backbone, cfg): ``backbone`` is a ``ConfigurableModel`` in eval
        mode, on ``device``, with ``requires_grad=False`` on every
        parameter. ``cfg`` is the resolved kwargs dict, for logging.
    """
    from src.models import ConfigurableModel
    base_cfg = dict(
        C=C, H=H, W=W,
        encoder_type=encoder_type, num_layers=num_layers, nhead=nhead,
        ffn_mult=ffn_mult, activation=activation,
        depthwise_conv=depthwise_conv, dropout=dropout,
        rev_norm_kind=rev_norm_kind,
    )
    if rev_norm_kind == "ewma":
        base_cfg["rev_norm_span"] = rev_norm_span
    sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = _detect_backbone_config(sd, base_cfg)
    if verbose:
        print(f"  [load_backbone] {checkpoint_path}: cfg={cfg}")
    backbone = ConfigurableModel(**cfg)
    sd = {k: v for k, v in sd.items()
          if not k.startswith("cpc_w1")
          and not k.startswith("teacher_")}
    backbone.load_state_dict(sd)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, cfg
