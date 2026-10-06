"""#412ow2: --lr-decay-shape linear makes the anneal after the warmup a
straight line, and the cosine stays the default."""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")


@pytest.fixture(scope="module")
def train_py():
    spec = importlib.util.spec_from_file_location("train_py_412ow2", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_linear_anneal_after_the_warmup(train_py):
    def lr(step):
        return train_py.scheduled_lr(step, 1e-3, 5.6e-5, 20000, 10000, "linear")
    assert lr(0) == 0.0
    assert lr(5000) == pytest.approx(5e-4)
    assert lr(10000) == pytest.approx(1e-3)
    assert lr(15000) == pytest.approx((1e-3 + 5.6e-5) / 2)
    assert lr(20000) == pytest.approx(5.6e-5)
    assert lr(40000) == pytest.approx(5.6e-5)


def test_the_cosine_is_the_default(train_py):
    for step in (0, 50000, 100000, 200000, 400000):
        assert (train_py.scheduled_lr(step, 5.6e-5, 1e-6, 200000)
                == train_py.cosine_lr(step, 5.6e-5, 1e-6, 200000))
