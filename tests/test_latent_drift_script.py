"""Contract tests for scripts/latent_drift.py.

Model-free — exercises the manifest reader and the row-formatter so a
malformed manifest fails loudly and the emitted CSV shape matches what
the plot script consumes. The full end-to-end test (loading a real
backbone and running drift over checkpoints) needs a checkpoint on
disk and is covered by the elisa run under
``experiments/2026-07-22_latent-drift/``.
"""

import csv
import importlib.util
import os
import sys
import tempfile

import pytest

_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "latent_drift.py")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("latent_drift", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["latent_drift"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_read_manifest_parses_and_sorts(tmp_path):
    mod = _load_script_module()
    ckpts = [tmp_path / f"c{i}.pth" for i in range(4)]
    for p in ckpts:
        p.write_bytes(b"stub")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "arm,step,path\n"
        f"arm1,25000,{ckpts[2]}\n"
        f"arm1,2000,{ckpts[0]}\n"
        f"arm2,12500,{ckpts[3]}\n"
        f"arm1,12500,{ckpts[1]}\n"
    )
    out = mod._read_manifest(str(manifest))
    assert set(out.keys()) == {"arm1", "arm2"}
    assert [s for s, _ in out["arm1"]] == [2000, 12500, 25000]
    assert [s for s, _ in out["arm2"]] == [12500]


def test_read_manifest_missing_columns_errors(tmp_path):
    mod = _load_script_module()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("arm,path\narm1,/nowhere\n")
    with pytest.raises(SystemExit, match="missing required columns"):
        mod._read_manifest(str(manifest))


def test_read_manifest_missing_ckpt_errors(tmp_path):
    mod = _load_script_module()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "arm,step,path\narm1,2000,/tmp/definitely-not-a-file-abc123.pth\n")
    with pytest.raises(SystemExit, match="missing file"):
        mod._read_manifest(str(manifest))


def test_row_formatter_shape():
    """Every emitted row must be [arm, step_a, step_b, delta_step,
    kind, drift_cos, drift_cos_aligned, rot_gap, cka] — 9 columns —
    matching the writer header."""
    mod = _load_script_module()
    import torch
    m = {
        "drift_cos": torch.tensor(0.1234),
        "drift_cos_aligned": torch.tensor(0.05),
        "rot_gap": torch.tensor(0.0734),
        "cka": torch.tensor(0.99),
    }
    row = mod._row("arm3", 12500, 25000, "adjacent", m)
    assert len(row) == 9
    assert row[0] == "arm3"
    assert row[1:5] == [12500, 25000, 12500, "adjacent"]
    # numeric columns serialised as strings
    assert all(isinstance(x, str) for x in row[5:])
