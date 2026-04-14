"""TDD tests for checkpoint.py — optimizer state save/load utilities."""

import os
import tempfile
import shutil

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.checkpoint import (
    get_optimizer_state_path,
    save_training_state,
    load_training_state,
    safe_save_path,
)


# ── Fixtures ──

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def simple_model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


@pytest.fixture
def trained_optimizer(simple_model):
    """Optimizer with non-trivial internal state (momentum buffers)."""
    opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
    x = torch.randn(2, 4)
    for _ in range(10):
        simple_model(x).sum().backward()
        opt.step()
        opt.zero_grad()
    return opt


# ── Tests: get_optimizer_state_path ──

class TestGetOptimizerStatePath:
    def test_basic_pth(self):
        assert get_optimizer_state_path("model.pth") == "model_optimizer.pth"

    def test_with_directory(self):
        assert get_optimizer_state_path("/tmp/ckpt/model.pth") == \
            "/tmp/ckpt/model_optimizer.pth"

    def test_best_checkpoint(self):
        assert get_optimizer_state_path("arch_search_best.pth") == \
            "arch_search_best_optimizer.pth"

    def test_no_extension(self):
        assert get_optimizer_state_path("model") == "model_optimizer"

    def test_relative_path(self):
        assert get_optimizer_state_path("./checkpoints/model.pth") == \
            "./checkpoints/model_optimizer.pth"


# ── Tests: save_training_state ──

class TestSaveTrainingState:
    def test_creates_file(self, tmp_dir, simple_model, trained_optimizer):
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)

        result_path = save_training_state(
            trained_optimizer, model_path,
            step=5000, best_val_ff=0.190, best_step=4500,
        )
        expected = os.path.join(tmp_dir, "model_optimizer.pth")
        assert result_path == expected
        assert os.path.exists(expected)

    def test_saved_contents(self, tmp_dir, simple_model, trained_optimizer):
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)

        save_training_state(
            trained_optimizer, model_path,
            step=5000, best_val_ff=0.190, best_step=4500,
        )
        loaded = torch.load(
            os.path.join(tmp_dir, "model_optimizer.pth"),
            map_location="cpu", weights_only=False,
        )
        assert "optimizer_state_dict" in loaded
        assert loaded["step"] == 5000
        assert loaded["best_val_ff"] == pytest.approx(0.190)
        assert loaded["best_step"] == 4500

    def test_overwrites_existing(self, tmp_dir, simple_model, trained_optimizer):
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)

        save_training_state(trained_optimizer, model_path,
                            step=1000, best_val_ff=0.1, best_step=500)
        save_training_state(trained_optimizer, model_path,
                            step=2000, best_val_ff=0.2, best_step=1500)

        loaded = torch.load(
            os.path.join(tmp_dir, "model_optimizer.pth"),
            map_location="cpu", weights_only=False,
        )
        assert loaded["step"] == 2000
        assert loaded["best_val_ff"] == pytest.approx(0.2)


# ── Tests: load_training_state ──

class TestLoadTrainingState:
    def test_roundtrip(self, tmp_dir, simple_model, trained_optimizer):
        """Save then load restores optimizer state identically."""
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)

        orig_state = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in trained_optimizer.state_dict()["state"][0].items()
        }

        save_training_state(
            trained_optimizer, model_path,
            step=5000, best_val_ff=0.190, best_step=4500,
        )

        fresh_opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        result = load_training_state(fresh_opt, model_path)

        assert result["step"] == 5000
        assert result["best_val_ff"] == pytest.approx(0.190)
        assert result["best_step"] == 4500

        restored = fresh_opt.state_dict()["state"][0]
        for key in orig_state:
            if isinstance(orig_state[key], torch.Tensor):
                assert torch.allclose(orig_state[key], restored[key]), \
                    f"Mismatch for key {key}"
            else:
                assert orig_state[key] == restored[key]

    def test_missing_file_returns_defaults(self, tmp_dir, simple_model):
        model_path = os.path.join(tmp_dir, "nonexistent.pth")
        opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        result = load_training_state(opt, model_path)

        assert result["step"] == 0
        assert result["best_val_ff"] == float("-inf")
        assert result["best_step"] == 0

    def test_corrupted_file(self, tmp_dir, simple_model):
        model_path = os.path.join(tmp_dir, "model.pth")
        optim_path = os.path.join(tmp_dir, "model_optimizer.pth")
        with open(optim_path, "wb") as f:
            f.write(b"not a valid pytorch file")

        opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        result = load_training_state(opt, model_path)

        assert result["step"] == 0
        assert result["best_val_ff"] == float("-inf")
        assert result["best_step"] == 0

    def test_incompatible_architecture(self, tmp_dir):
        """Optimizer state from different param group count falls back gracefully."""
        # Model A: 1 layer = 2 param groups (weight + bias)
        model_a = nn.Linear(4, 8)
        opt_a = optim.AdamW(model_a.parameters(), lr=1e-3)
        x = torch.randn(2, 4)
        for _ in range(5):
            model_a(x).sum().backward()
            opt_a.step()
            opt_a.zero_grad()

        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(model_a.state_dict(), model_path)
        save_training_state(opt_a, model_path, step=100,
                            best_val_ff=0.5, best_step=50)

        # Model B: 3 layers = 6 param groups (different count -> load fails)
        model_b = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                                nn.Linear(8, 16), nn.ReLU(),
                                nn.Linear(16, 4))
        opt_b = optim.AdamW(model_b.parameters(), lr=1e-3)
        result = load_training_state(opt_b, model_path)

        assert result["step"] == 0

    def test_backward_compat_old_checkpoint(self, tmp_dir, simple_model):
        """Old checkpoints without companion file work fine."""
        model_path = os.path.join(tmp_dir, "old_model.pth")
        torch.save(simple_model.state_dict(), model_path)

        opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        result = load_training_state(opt, model_path)

        assert result["step"] == 0
        assert result["best_val_ff"] == float("-inf")
        assert result["best_step"] == 0
        assert len(opt.state_dict()["state"]) == 0


# ── Integration: full save/load cycle ──

class TestFullCycle:
    def test_restored_vs_fresh_optimizer_differ(self, tmp_dir, simple_model):
        """Proves restored optimizer momentum affects parameter updates."""
        x = torch.randn(2, 4)
        target = torch.randn(2, 4)

        # Train 10 steps
        opt1 = optim.AdamW(simple_model.parameters(), lr=1e-3)
        for _ in range(10):
            nn.functional.mse_loss(simple_model(x), target).backward()
            opt1.step()
            opt1.zero_grad()

        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)
        save_training_state(opt1, model_path, step=10,
                            best_val_ff=0.5, best_step=8)

        params_10 = {n: p.clone() for n, p in simple_model.named_parameters()}

        # Step 11 with RESTORED optimizer
        opt_restored = optim.AdamW(simple_model.parameters(), lr=1e-3)
        load_training_state(opt_restored, model_path)
        nn.functional.mse_loss(simple_model(x), target).backward()
        opt_restored.step()
        opt_restored.zero_grad()
        params_restored = {n: p.clone() for n, p in simple_model.named_parameters()}

        # Reset to step 10, step 11 with FRESH optimizer
        with torch.no_grad():
            for n, p in simple_model.named_parameters():
                p.copy_(params_10[n])
        opt_fresh = optim.AdamW(simple_model.parameters(), lr=1e-3)
        nn.functional.mse_loss(simple_model(x), target).backward()
        opt_fresh.step()
        opt_fresh.zero_grad()
        params_fresh = {n: p.clone() for n, p in simple_model.named_parameters()}

        any_different = any(
            not torch.allclose(params_restored[n], params_fresh[n])
            for n in params_restored
        )
        assert any_different, "Restored and fresh optimizers should produce different updates"

    def test_step_counter_resumes(self, tmp_dir, simple_model):
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)

        opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        save_training_state(opt, model_path, step=5000,
                            best_val_ff=0.190, best_step=4500)

        fresh_opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        state = load_training_state(fresh_opt, model_path)
        assert state["step"] + 1 == 5001


# ── Tests: safe_save_path ──

class TestSafeSavePath:
    def test_no_conflict_different_dir(self, tmp_dir):
        resume = os.path.join(tmp_dir, "checkpoints", "model.pth")
        save = os.path.join(tmp_dir, "output", "model.pth")
        assert safe_save_path(save, resume) == save

    def test_no_conflict_different_name(self, tmp_dir):
        resume = os.path.join(tmp_dir, "model_a.pth")
        save = os.path.join(tmp_dir, "model_b.pth")
        assert safe_save_path(save, resume) == save

    def test_no_resume(self, tmp_dir):
        save = os.path.join(tmp_dir, "model.pth")
        assert safe_save_path(save, None) == save

    def test_conflict_same_path(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pth")
        result = safe_save_path(path, path)
        assert result != path
        assert "_run1" in result

    def test_conflict_save_overwrites_resume_best(self, tmp_dir):
        """save_path 'model.pth' would create 'model_best.pth' which
        conflicts with resume from 'model_best.pth'."""
        resume = os.path.join(tmp_dir, "model_best.pth")
        save = os.path.join(tmp_dir, "model.pth")
        result = safe_save_path(save, resume)
        assert result != save
        assert "_run1" in result

    def test_increments_run_number(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pth")
        # Create _run1 so it must go to _run2
        open(os.path.join(tmp_dir, "model_run1.pth"), "w").close()
        result = safe_save_path(path, path)
        assert "_run2" in result

    def test_conflict_with_optimizer_companion(self, tmp_dir):
        """Resuming from model_optimizer.pth base should detect conflict."""
        resume = os.path.join(tmp_dir, "model.pth")
        save = os.path.join(tmp_dir, "model.pth")
        result = safe_save_path(save, resume)
        assert "_run" in result


# ── Tests: complete training state (best_loss, EMA, RNG, hf_rows) ──

class TestCompleteTrainingState:
    """Tests for saving/loading ALL training state, not just optimizer+gap."""

    def test_best_loss_roundtrip(self, tmp_dir, simple_model, trained_optimizer):
        """best_loss and best_loss_step are saved and restored."""
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)

        save_training_state(
            trained_optimizer, model_path,
            step=5000, best_val_ff=0.19, best_step=4500,
            best_loss=-1.5, best_loss_step=4800,
        )
        fresh_opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        result = load_training_state(fresh_opt, model_path)

        assert result["best_loss"] == pytest.approx(-1.5)
        assert result["best_loss_step"] == 4800

    def test_ema_roundtrip(self, tmp_dir, simple_model, trained_optimizer):
        """ema_loss and ema_gap are saved and restored."""
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)

        save_training_state(
            trained_optimizer, model_path,
            step=5000, best_val_ff=0.19, best_step=4500,
            ema_loss=0.42, ema_gap=0.35,
        )
        fresh_opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        result = load_training_state(fresh_opt, model_path)

        assert result["ema_loss"] == pytest.approx(0.42)
        assert result["ema_gap"] == pytest.approx(0.35)

    def test_ema_none_roundtrip(self, tmp_dir, simple_model, trained_optimizer):
        """ema_loss=None (initial state) is preserved."""
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)

        save_training_state(
            trained_optimizer, model_path,
            step=0, best_val_ff=-float("inf"), best_step=0,
            ema_loss=None, ema_gap=None,
        )
        fresh_opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        result = load_training_state(fresh_opt, model_path)

        assert result["ema_loss"] is None
        assert result["ema_gap"] is None

    def test_hf_rows_consumed_roundtrip(self, tmp_dir, simple_model, trained_optimizer):
        """hf_rows_consumed is saved and restored."""
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)

        save_training_state(
            trained_optimizer, model_path,
            step=5000, best_val_ff=0.19, best_step=4500,
            hf_rows_consumed=479904,
        )
        fresh_opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        result = load_training_state(fresh_opt, model_path)

        assert result["hf_rows_consumed"] == 479904

    def test_rng_state_torch_roundtrip(self, tmp_dir, simple_model, trained_optimizer):
        """Torch RNG state is saved and can be restored for reproducibility."""
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)

        # Set a specific RNG state and generate reference values
        torch.manual_seed(12345)
        rng_state = torch.get_rng_state()
        ref_values = torch.randn(5)

        save_training_state(
            trained_optimizer, model_path,
            step=5000, best_val_ff=0.19, best_step=4500,
            rng_state_torch=rng_state,
        )

        # Mess up the RNG
        torch.manual_seed(99999)
        _ = torch.randn(100)

        # Restore
        fresh_opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        result = load_training_state(fresh_opt, model_path)

        assert result["rng_state_torch"] is not None
        torch.set_rng_state(result["rng_state_torch"])
        restored_values = torch.randn(5)
        assert torch.allclose(ref_values, restored_values)

    def test_rng_state_numpy_roundtrip(self, tmp_dir, simple_model, trained_optimizer):
        """Numpy RNG state is saved and can be restored for reproducibility."""
        import numpy as np
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)

        np.random.seed(12345)
        rng_state = np.random.get_state()
        ref_values = np.random.randn(5)

        save_training_state(
            trained_optimizer, model_path,
            step=5000, best_val_ff=0.19, best_step=4500,
            rng_state_numpy=rng_state,
        )

        # Mess up numpy RNG
        np.random.seed(99999)
        _ = np.random.randn(100)

        # Restore
        fresh_opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        result = load_training_state(fresh_opt, model_path)

        assert result["rng_state_numpy"] is not None
        np.random.set_state(result["rng_state_numpy"])
        restored_values = np.random.randn(5)
        np.testing.assert_array_almost_equal(ref_values, restored_values)

    def test_backward_compat_old_checkpoint_new_fields(self, tmp_dir, simple_model):
        """Old checkpoints (without new fields) load with safe defaults."""
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)

        # Manually create an old-format companion file (only old fields)
        optim_path = get_optimizer_state_path(model_path)
        opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        old_state = {
            "optimizer_state_dict": opt.state_dict(),
            "step": 1000,
            "best_val_ff": 0.15,
            "best_step": 900,
        }
        torch.save(old_state, optim_path)

        fresh_opt = optim.AdamW(simple_model.parameters(), lr=1e-3)
        result = load_training_state(fresh_opt, model_path)

        # Old fields work
        assert result["step"] == 1000
        assert result["best_val_ff"] == pytest.approx(0.15)
        assert result["best_step"] == 900
        # New fields get defaults
        assert result["best_loss"] == float("inf")
        assert result["best_loss_step"] == 0
        assert result["ema_loss"] is None
        assert result["ema_gap"] is None
        assert result["hf_rows_consumed"] == 0
        assert result["rng_state_torch"] is None
        assert result["rng_state_numpy"] is None

    def test_all_fields_present_in_saved_file(self, tmp_dir, simple_model, trained_optimizer):
        """Verify the saved companion file contains all expected keys."""
        import numpy as np
        model_path = os.path.join(tmp_dir, "model.pth")
        torch.save(simple_model.state_dict(), model_path)

        save_training_state(
            trained_optimizer, model_path,
            step=5000, best_val_ff=0.19, best_step=4500,
            best_loss=-1.2, best_loss_step=4700,
            ema_loss=0.42, ema_gap=0.35,
            hf_rows_consumed=480000,
            rng_state_torch=torch.get_rng_state(),
            rng_state_numpy=np.random.get_state(),
        )

        optim_path = get_optimizer_state_path(model_path)
        loaded = torch.load(optim_path, map_location="cpu", weights_only=False)

        expected_keys = {
            "optimizer_state_dict", "step", "best_val_ff", "best_step",
            "best_loss", "best_loss_step", "ema_loss", "ema_gap",
            "hf_rows_consumed", "rng_state_torch", "rng_state_numpy",
        }
        assert expected_keys.issubset(set(loaded.keys()))
