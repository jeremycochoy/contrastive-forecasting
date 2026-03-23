"""
Training checkpoint utilities for saving/loading optimizer state,
step counter, and best-tracking metadata alongside model checkpoints.

The model checkpoint format is NOT changed. Optimizer state is saved
in a companion file (model.pth -> model_optimizer.pth).
"""

import os
import torch


_DEFAULTS = {
    "step": 0,
    "best_val_ff": float("-inf"),
    "best_step": 0,
}


def get_optimizer_state_path(model_path: str) -> str:
    """Derive optimizer state file path from model checkpoint path."""
    root, ext = os.path.splitext(model_path)
    return f"{root}_optimizer{ext}"


def save_training_state(optimizer, model_path: str, step: int,
                        best_val_ff: float, best_step: int) -> str:
    """Save optimizer state and training metadata to companion file.

    Returns the path where the state was saved.
    """
    state = {
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_ff": best_val_ff,
        "best_step": best_step,
    }
    optim_path = get_optimizer_state_path(model_path)
    torch.save(state, optim_path)
    return optim_path


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
        return {
            "step": state["step"],
            "best_val_ff": state["best_val_ff"],
            "best_step": state["best_step"],
        }
    except Exception as e:
        print(f"  [checkpoint] WARNING: Failed to load {optim_path}: {e}")
        print(f"  [checkpoint] Continuing with fresh optimizer.")
        return dict(_DEFAULTS)
