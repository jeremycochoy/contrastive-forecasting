"""Tests for #419: every series of GiftEvalPretrain, in 1,024-value windows.

Seven groups, all on the CPU and on a corpus the tests write themselves in
the layout of `Salesforce/GiftEvalPretrain` (Arrow IPC stream files):

1. The index: the walk finds every record batch, its rows and its values,
   and the frequency of the source, without reading a body.
2. The sampler: sources in proportion to uni2ts's slots, the ERA5 and CMIP6
   years as one family, `SampleDimension` over the variates.
3. The windows: short and long series, left zero padding, missing values,
   one window per variate of a multivariate row, a label per window.
4. The normaliser: with zero padding its statistics come from the real
   values, and without the new mode nothing changes.
5. The value-space loss skips padded targets.
6. The vocabulary v2, and v1 checkpoints as before, in the eval too.
7. A few trainer steps on the stream.
"""

from __future__ import annotations

import collections
import gzip
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import gift_pretrain as gp  # noqa: E402
from src.forecasting_head import (QUANTILE_LEVELS, masked_quantile_loss,  # noqa: E402
                                  quantile_loss, shift_pad,
                                  value_space_objective)
from src.freq_embedding import (FREQ_NAMES, FREQ_NAMES_V2, FREQ_VOCABS,  # noqa: E402
                                freq_labels, freq_to_id, gluonts_freq_to_id,
                                seasonality_to_id, vocab_of_rows)
from src.models import ConfigurableModel  # noqa: E402
from src.norm import RevEWMNorm, leading_zero_count  # noqa: E402

BUILDER = REPO_ROOT / "scripts" / "build_gift_pretrain_index.py"
TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
EVAL_PY = (REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
           / "eval_gift_eval_official.py")
T = 1024


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script(BUILDER, "build_gift_pretrain_index_419")


# ---------------------------------------------------------------------------
# A small corpus in the layout of GiftEvalPretrain
# ---------------------------------------------------------------------------

def arrow_schema(target_dim, cov_dim, freq_last=False):
    import pyarrow as pa
    flt = pa.list_(pa.float32())
    fields = [("item_id", pa.string()), ("start", pa.timestamp("s")),
              ("freq", pa.string()),
              ("target", flt if target_dim == 1 else pa.list_(flt, target_dim))]
    if cov_dim:
        fields.append(("past_feat_dynamic_real", pa.list_(flt, cov_dim)))
    if freq_last:
        fields.append(fields.pop(2))
    return pa.schema(fields)


def arrow_rows(rows, freq, schema):
    import pyarrow as pa
    cols = {"item_id": [f"s{i}" for i in range(len(rows))],
            "start": [0] * len(rows), "freq": [freq] * len(rows),
            "target": [r["target"].tolist() for r in rows]}
    if "past_feat_dynamic_real" in schema.names:
        cols["past_feat_dynamic_real"] = [r["cov"].tolist() for r in rows]
    return pa.record_batch([pa.array(cols[n], type=schema.field(n).type)
                            for n in schema.names], schema=schema)


def write_source(root, name, rows, freq, batch_rows=2, freq_last=False):
    """``<root>/<name>/data-00000-of-00001.arrow``, ``batch_rows`` a batch."""
    import pyarrow as pa
    first = rows[0]
    dt = 1 if first["target"].ndim == 1 else first["target"].shape[0]
    dc = first["cov"].shape[0] if "cov" in first else 0
    schema = arrow_schema(dt, dc, freq_last)
    (root / name).mkdir(parents=True, exist_ok=True)
    with pa.ipc.new_stream(str(root / name / "data-00000-of-00001.arrow"),
                           schema) as w:
        for i in range(0, len(rows), batch_rows):
            w.write_batch(arrow_rows(rows[i:i + batch_rows], freq, schema))


def index_sources(root, weights):
    """The index `build_gift_pretrain_index.py` writes, for local sources."""
    sources = {}
    for name in sorted(weights):
        path = f"{name}/data-00000-of-00001.arrow"
        size = (root / path).stat().st_size
        walk = builder.walk_file(path, None, 0, size, root=str(root))
        sources[name] = builder.summarize(name, [path], [walk], weights)
    return {"repo": gp.HF_REPO, "revision": None, "max_dim": gp.MAX_DIM,
            "sources": sources}


def series(rng, length, level=0.0):
    return (level + rng.standard_normal(length)).astype(np.float32)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Four sources: short yearly series, a multivariate hourly source with
    covariates (freq stored last), and two ERA5-like years."""
    root = tmp_path_factory.mktemp("gep")
    rng = np.random.default_rng(0)
    short = [{"target": series(rng, n, 5000.0)} for n in (12, 20, 31, 7, 25)]
    multi = [{"target": np.stack([series(rng, 1500, 10.0 * j) for j in range(3)]),
              "cov": np.stack([series(rng, 1500, -5.0) for _ in range(2)])}
             for _ in range(4)]
    write_source(root, "tiny_yearly", short, "A-DEC")
    write_source(root, "tiny_hourly", multi, "H", freq_last=True)
    for year in (1990, 1991):
        era = [{"target": np.stack([series(rng, 1100) for _ in range(4)])}
               for _ in range(3)]
        write_source(root, f"era5_{year}", era, "H", batch_rows=3)
    weights = {"tiny_yearly": 2.0, "tiny_hourly": 1.5, "era5_1990": 1.0,
               "era5_1991": 1.0}
    return root, index_sources(root, weights), {"short": short, "multi": multi}


def write_index(tmp_path, index):
    path = tmp_path / "index.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(index, f)
    return path


# ---------------------------------------------------------------------------
# 1. The index
# ---------------------------------------------------------------------------

def test_the_walk_finds_every_record_batch(corpus):
    _, index, _ = corpus
    short = index["sources"]["tiny_yearly"]
    assert [b[3] for b in short["batches"]] == [2, 2, 1]
    assert sum(b[4] for b in short["batches"]) == 12 + 20 + 31 + 7 + 25
    assert short["freq"] == "A-DEC" and short["rows"] == 5
    assert (short["target_dim"], short["cov_dim"]) == (1, 0)


def test_the_walk_reads_dims_and_a_freq_stored_last(corpus):
    _, index, _ = corpus
    multi = index["sources"]["tiny_hourly"]
    assert (multi["target_dim"], multi["cov_dim"]) == (3, 2)
    assert multi["freq"] == "H"
    assert sum(b[4] for b in multi["batches"]) == 4 * 3 * 1500
    assert sum(b[5] for b in multi["batches"]) == 4 * 2 * 1500


def test_a_batch_read_by_its_offsets_decodes_to_the_rows(corpus):
    root, index, rows = corpus
    src = index["sources"]["tiny_yearly"]
    ref = gp._block_refs("tiny_yearly", src)[1]
    whole = dict(ref, rows_from=0, rows_to=ref["rows"])
    block = gp.BlockLoader(None, str(root)).chunk(whole, uniform=False)
    assert block.rows == 2
    np.testing.assert_array_equal(block.series(0, 0, 0), rows["short"][2]["target"])
    np.testing.assert_array_equal(block.series(0, 1, 0), rows["short"][3]["target"])


def test_a_whole_file_decodes_to_every_batch(corpus):
    root, index, rows = corpus
    refs = gp._block_refs("tiny_hourly", index["sources"]["tiny_hourly"])
    blocks = gp.BlockLoader(None, str(root)).whole_file(refs, uniform=False)
    assert [b.rows for b in blocks] == [2, 2]
    np.testing.assert_array_equal(blocks[1].series(1, 1, 0), rows["multi"][3]["cov"][0])
    np.testing.assert_array_equal(blocks[0].series(0, 0, 2), rows["multi"][0]["target"][2])


def rows_of(block):
    return [[block.series(f, r, j).tolist() for f, (_, st, _) in
             enumerate(block.fields) for j in range(st.shape[1])]
            for r in range(block.rows)]


def test_a_chunk_reads_the_rows_of_the_whole_batch(corpus):
    """A large source is read a run of rows at a time, from three ranges of
    its record batch. The rows must be the ones pyarrow decodes."""
    root, index, _ = corpus
    refs = gp._block_refs("tiny_hourly", index["sources"]["tiny_hourly"])
    loader = gp.BlockLoader(None, str(root))
    whole = [r for b in loader.whole_file(refs, False) for r in rows_of(b)]
    chunks = gp.chunk_refs(refs, chunk_bytes=1)
    assert [(c["rows_from"], c["rows_to"]) for c in chunks] == [(0, 1), (1, 2)] * 2
    got = [r for c in chunks for r in rows_of(loader.chunk(c, False))]
    assert got == whole


def test_a_chunk_turns_nulls_into_nan(tmp_path):
    import pyarrow as pa
    schema = arrow_schema(1, 0)
    cols = [pa.array(["a", "b"]), pa.array([0, 0], pa.timestamp("s")),
            pa.array(["D", "D"]),
            pa.array([[1.0, None, 3.0], [4.0, 5.0]], schema.field("target").type)]
    (tmp_path / "nul").mkdir()
    path = tmp_path / "nul" / "data-00000-of-00001.arrow"
    with pa.ipc.new_stream(str(path), schema) as w:
        w.write_batch(pa.record_batch(cols, schema=schema))
    index = index_sources(tmp_path, {"nul": 1.0})
    ref = gp.chunk_refs(gp._block_refs("nul", index["sources"]["nul"]))[0]
    block = gp.BlockLoader(None, str(tmp_path)).chunk(ref, False)
    assert np.isnan(block.series(0, 0, 0)[1])
    assert block.series(0, 1, 0).tolist() == [4, 5]


def test_the_shipped_index_covers_every_source():
    index = gp.load_index()
    assert len(index["sources"]) == 152
    for name, src in index["sources"].items():
        assert src["weight"] > 0 and src["rows"] > 0 and src["batches"], name
        assert src["freq"] and src["target_dim"] >= 1, name
    fams = gp.build_families(index)
    assert np.isclose(gp.family_probabilities(fams).sum(), 1.0)


# ---------------------------------------------------------------------------
# 2. The sampler
# ---------------------------------------------------------------------------

def test_era5_years_share_one_family(corpus):
    _, index, _ = corpus
    fams = {f["name"]: f for f in gp.build_families(index)}
    assert set(fams) == {"era5", "tiny_hourly", "tiny_yearly"}
    assert fams["era5"]["members"] == ["era5_1990", "era5_1991"]
    assert fams["era5"]["uniform"] and not fams["tiny_yearly"]["uniform"]


def test_families_are_drawn_in_proportion_to_uni2ts_slots(corpus):
    _, index, _ = corpus
    fams = gp.build_families(index)
    slots = {"era5": 3 + 3, "tiny_hourly": 6, "tiny_yearly": 10}
    want = np.array([slots[f["name"]] for f in fams], dtype=float)
    np.testing.assert_allclose(gp.family_probabilities(fams), want / want.sum())


def test_sample_dimension_draws_one_to_d_distinct_variates():
    rng = np.random.default_rng(1)
    seen = collections.Counter()
    for _ in range(2000):
        picks = gp.sample_variates(rng, (3, 2))
        target = [j for f, j in picks if f == 0]
        cov = [j for f, j in picks if f == 1]
        assert 1 <= len(target) <= 3 and 1 <= len(cov) <= 2
        assert len(set(target)) == len(target)
        seen[len(target)] += 1
    assert set(seen) == {1, 2, 3} and min(seen.values()) > 500


def test_sample_dimension_caps_at_max_dim():
    rng = np.random.default_rng(2)
    counts = [len(gp.sample_variates(rng, (300, 0))) for _ in range(500)]
    assert max(counts) <= gp.MAX_DIM and min(counts) >= 1


# ---------------------------------------------------------------------------
# 3. The windows
# ---------------------------------------------------------------------------

def test_a_short_series_is_left_padded_with_zeros():
    s = np.arange(1, 21, dtype=np.float32) + 5000
    w = gp.cut_window(s, 0, T)
    assert w.shape == (T,) and (w[:T - 20] == 0).all()
    np.testing.assert_array_equal(w[T - 20:], s)


def test_a_long_series_gives_its_exact_slice():
    s = np.arange(5000, dtype=np.float32) + 1
    np.testing.assert_array_equal(gp.cut_window(s, 1234, T), s[1234:1234 + T])
    rng = np.random.default_rng(3)
    starts = [gp.crop_start(rng, 5000, T) for _ in range(500)]
    assert min(starts) >= 0 and max(starts) <= 5000 - T


def test_missing_values_pad_before_the_first_and_fill_after():
    s = np.array([np.nan, np.nan, 3.0, np.nan, 5.0, np.inf], dtype=np.float32)
    w = gp.cut_window(s, 0, 8)
    np.testing.assert_array_equal(w, [0, 0, 0, 0, 3, 3, 5, 5])
    assert gp.cut_window(np.full(10, np.nan, np.float32), 0, 8) is None


def coded(row, dims, base=0, length=1500):
    """Variates whose values name their row, variate and time step."""
    t = np.arange(length)
    return np.stack([row * 1e6 + (base + j) * 1e4 + t
                     for j in range(dims)]).astype(np.float32)


def test_every_variate_is_its_own_window():
    """A draw of a [3, 1500] row with 2 covariates gives one window per
    sampled variate, every one cut at the same position of the same row."""
    rows = [{"target": coded(r, 3), "cov": coded(r, 2, base=3)} for r in (1, 2)]
    block = gp.Block.from_batch(arrow_rows(rows, "H", arrow_schema(3, 2)),
                                False, "x")
    rng = np.random.default_rng(4)
    for _ in range(50):
        got = gp.draw_windows(rng, block, (3, 2), T)
        assert 2 <= len(got) <= 5
        assert len({int(w[0] // 1e6) for w in got}) == 1       # one row
        assert len({int(w[0] % 1e4) for w in got}) == 1        # one crop
        assert len({int(w[0] // 1e4) for w in got}) == len(got)
        assert all((np.diff(w) == 1).all() for w in got)


def stream_of(corpus, **kw):
    root, index, _ = corpus
    opts = dict(batch_size=8, C=1, seed=0, index=index, root=str(root),
                freq_vocab="v2")
    opts.update(kw)
    return gp.GiftPretrainStream(**opts)


def test_a_batch_carries_a_label_per_window(corpus, monkeypatch):
    monkeypatch.setattr(gp, "SHUFFLE_WINDOWS", 64)
    x, freq, seas = next(stream_of(corpus)._batches())
    assert x.shape == (8, T, 1) and freq.shape == seas.shape == (8,)
    yearly = freq_labels("A-DEC", "v2")
    hourly = freq_labels("H", "v2")
    pairs = set(zip(freq.tolist(), seas.tolist()))
    assert pairs <= {yearly, hourly}
    for b in range(8):
        pad = int(leading_zero_count(x[b:b + 1])[0, 0, 0])
        assert (pad >= T - 31) == ((freq[b].item(), seas[b].item()) == yearly)


def test_the_stream_is_a_function_of_its_seed(corpus, monkeypatch):
    monkeypatch.setattr(gp, "SHUFFLE_WINDOWS", 32)
    a = [b[0] for _, b in zip(range(3), stream_of(corpus, seed=5)._batches())]
    b = [b[0] for _, b in zip(range(3), stream_of(corpus, seed=5)._batches())]
    c = [b[0] for _, b in zip(range(3), stream_of(corpus, seed=6)._batches())]
    assert all(torch.equal(p, q) for p, q in zip(a, b))
    assert not all(torch.equal(p, q) for p, q in zip(a, c))


def test_streaming_pools_serve_the_same_windows_as_resident_ones(corpus, monkeypatch):
    """With every source over the resident size, the pools keep one record
    batch and replace it; the stream still runs and labels every window."""
    monkeypatch.setattr(gp, "SHUFFLE_WINDOWS", 32)
    monkeypatch.setattr(gp, "RESIDENT_BYTES", 0)
    monkeypatch.setattr(gp, "POOL_BYTES", 1)
    monkeypatch.setattr(gp, "CHUNK_BYTES", 1)
    stream = stream_of(corpus, block_use=0.01)
    batches = [b for _, b in zip(range(20), stream._batches())]
    assert all(b[0].shape == (8, T, 1) for b in batches)
    assert sum(stream.counts.values()) == 20 * 8


def test_without_labels_the_stream_yields_bare_tensors(corpus, monkeypatch):
    monkeypatch.setattr(gp, "SHUFFLE_WINDOWS", 16)
    x = next(stream_of(corpus, emit_labels=False, C=2, batch_size=3)._batches())
    assert isinstance(x, torch.Tensor) and x.shape == (3, T, 2)


def test_a_mixed_loader_passes_the_stream_labels_through(corpus, monkeypatch):
    from src.dataloader import create_mixed_forked_arma_dataloader
    monkeypatch.setattr(gp, "SHUFFLE_WINDOWS", 16)
    make = lambda bs, emit: stream_of(corpus, batch_size=bs, emit_labels=emit)
    loader = create_mixed_forked_arma_dataloader(
        repo_id=None, batch_size=6, C=1, mix_ratio=1 / 3, T_raw=4096, seed=0,
        emit_freq_ids=True, cross_triplets=1, real_rows=make)
    x, freq, seas = next(iter(loader))
    assert x.shape == (4 + 2 + 3, T, 1) and freq.shape == (9,)
    assert set(freq[:4].tolist()) <= {freq_to_id("A-DEC", "v2"), freq_to_id("H", "v2")}


# ---------------------------------------------------------------------------
# 4. The normaliser
# ---------------------------------------------------------------------------

def norm(skip, C=1):
    return RevEWMNorm(C, span=128, patch_size=16, skip_leading_zeros=skip)


def padded(length, level=5000.0, seed=0):
    """A level-5,000 series of ``length`` values after ``T - length`` zeros."""
    g = torch.Generator().manual_seed(seed)
    s = level + 50 * torch.randn(1, length, 1, generator=g)
    return torch.cat([torch.zeros(1, T - length, 1), s], dim=1), s


def test_zero_padding_throws_the_plain_norm_off():
    """The finding: under the plain EWMA (span 128) a 20-point series padded
    with 1,004 zeros starts from mean 0 and variance 0, and its first value
    normalises to about +8."""
    x, _ = padded(20)
    first = norm(False)(x, "norm")[0, T - 20, 0].item()
    assert 7.0 < first < 9.0


@pytest.mark.parametrize("length", [20, 5, 700])
def test_padded_statistics_are_those_of_the_series_alone(length):
    x, s = padded(length)
    skip, plain = norm(True), norm(False)
    out, ref = skip(x, "norm"), plain(s, "norm")
    assert torch.allclose(out[:, T - length:], ref, atol=1e-4)
    assert (out[:, :T - length] == 0).all()
    assert torch.allclose(skip.stdev[:, T - length:], plain.stdev, rtol=1e-4)
    assert skip.pad_mask.sum() == T - length


def test_a_series_with_no_padding_normalises_as_before():
    x = torch.randn(3, T, 2) + 3
    assert torch.allclose(norm(True, C=2)(x, "norm"), norm(False, C=2)(x, "norm"),
                          atol=1e-5)


def test_rows_with_different_padding_share_a_batch():
    xs = [padded(n, seed=n)[0] for n in (30, 700, T)]
    out = norm(True)(torch.cat(xs, dim=0), "norm")
    for i, x in enumerate(xs):
        assert torch.allclose(out[i:i + 1], norm(True)(x, "norm"), atol=1e-5)


def test_an_all_zero_series_stays_zero_without_nan():
    skip = norm(True)
    out = skip(torch.zeros(2, T, 1), "norm")
    assert (out == 0).all() and skip.pad_mask.all()
    assert torch.isfinite(skip.mean).all() and torch.isfinite(skip.stdev).all()


def test_denorm_gives_back_the_real_values():
    x, s = padded(40)
    skip = norm(True)
    back = skip(skip(x, "norm"), "denorm")
    assert torch.allclose(back[:, T - 40:], s, rtol=1e-4)


def backbone_kwargs(**over):
    kw = dict(C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
              ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
              rev_norm_kind="ewma", rev_norm_span=128)
    kw.update(over)
    return kw


def test_the_mode_is_one_buffer_in_the_state_dict():
    plain = ConfigurableModel(**backbone_kwargs()).state_dict()
    zero_pad = ConfigurableModel(**backbone_kwargs(
        rev_norm_skip_leading_zeros=True)).state_dict()
    assert set(zero_pad) - set(plain) == {"rev_norm.leading_zero_pad"}
    assert list(norm(False).state_dict()) == []


def test_the_mode_needs_the_ewma_norm():
    with pytest.raises(ValueError):
        ConfigurableModel(**backbone_kwargs(rev_norm_kind="revin",
                                            rev_norm_skip_leading_zeros=True))


# ---------------------------------------------------------------------------
# 5. The value-space loss
# ---------------------------------------------------------------------------

def test_masked_loss_with_every_value_kept_is_the_plain_loss():
    pred, target = torch.randn(2, 5, 1, 9, 16), torch.randn(2, 5, 1, 16)
    keep = torch.ones_like(target, dtype=torch.bool)
    assert torch.allclose(masked_quantile_loss(pred, target, keep),
                          quantile_loss(pred, target))


def test_padded_targets_do_not_move_the_loss():
    pred, target = torch.randn(2, 5, 1, 9, 16), torch.randn(2, 5, 1, 16)
    keep = torch.rand(2, 5, 1, 16) > 0.3
    moved = target + 100.0 * (~keep)
    assert torch.allclose(masked_quantile_loss(pred, target, keep),
                          masked_quantile_loss(pred, moved, keep))


def test_shift_pad_moves_the_padding_left_by_j_patches():
    pad = torch.arange(64).view(1, 64, 1) < 40
    assert shift_pad(pad, 8, 0).equal(pad)
    assert int(shift_pad(pad, 8, 1).sum()) == 32
    assert int(shift_pad(pad, 8, 5).sum()) == 0


def value_model():
    torch.manual_seed(0)
    return ConfigurableModel(
        C=1, H=16, W=8, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=2.0, dropout=0.0, rev_norm_kind="ewma", rev_norm_span=16,
        num_encoder_layers=1, enc_transformer_use_grad_checkpoint=False,
        value_head_quantiles=len(QUANTILE_LEVELS),
        rev_norm_skip_leading_zeros=True)


def test_no_padding_gives_the_plain_objective():
    m = value_model()
    x_norm = m.rev_norm(torch.randn(2, 64, 1) + 2, "norm")
    a = value_space_objective(m, x_norm, depth=2, pad_mask=m.rev_norm.pad_mask)
    b = value_space_objective(m, x_norm, depth=2)
    assert torch.allclose(a[0], b[0])


def test_the_rolled_inputs_keep_their_padding_at_zero(monkeypatch):
    import src.forecasting_head as fh
    seen, real = [], fh.value_space_forward

    def spy(model, x, **kw):
        seen.append(x.detach().clone())
        return real(model, x, **kw)
    monkeypatch.setattr(fh, "value_space_forward", spy)
    m = value_model()
    x = torch.cat([torch.zeros(1, 40, 1), torch.randn(1, 24, 1) + 5], dim=1)
    value_space_objective(m, m.rev_norm(x, "norm"), depth=3,
                          pad_mask=m.rev_norm.pad_mask)
    assert len(seen) == 4
    assert all((x_in[0, :40 - 8 * j] == 0).all() for j, x_in in enumerate(seen))


# ---------------------------------------------------------------------------
# 6. The vocabulary, and the eval
# ---------------------------------------------------------------------------

def test_v2_keeps_every_v1_class_at_its_id():
    assert FREQ_NAMES_V2[:len(FREQ_NAMES)] == FREQ_NAMES
    assert vocab_of_rows(10) == "v1"
    assert vocab_of_rows(len(FREQ_NAMES_V2)) == "v2"
    assert FREQ_VOCABS["v1"] is FREQ_NAMES


@pytest.mark.parametrize("freq", ["H", "D", "W", "10S", "5T", "15min", "M",
                                  "Q-DEC", "A-DEC", "W-SUN", "6H", None, ""])
def test_v1_maps_every_string_as_before(freq):
    assert freq_to_id(freq, "v1") == gluonts_freq_to_id(freq)


@pytest.mark.parametrize("freq,name", [
    ("M", "1M"), ("MS", "1M"), ("Q-DEC", "1Q"), ("A-DEC", "1Y"), ("Y", "1Y"),
    ("W-SUN", "1w"), ("W-WED", "1w"), ("H", "1h"), ("5T", "5min"),
    ("10S", "10s"), ("D", "1d"), ("6H", "6h"), ("15min", "15min")])
def test_v2_names_every_frequency(freq, name):
    assert FREQ_NAMES_V2[freq_to_id(freq, "v2")] == name


def test_every_corpus_frequency_has_a_v2_class():
    for name, src in gp.load_index()["sources"].items():
        assert freq_to_id(src["freq"], "v2") != 0, (name, src["freq"])


def test_labels_follow_gluonts_seasonality():
    assert freq_labels("H", "v2") == (freq_to_id("H", "v2"), seasonality_to_id(24))
    assert freq_labels("A-DEC", "v2") == (FREQ_NAMES_V2.index("1Y"), 0)
    assert freq_labels("M", "v1") == (0, seasonality_to_id(12))


def eval_module(tag):
    pytest.importorskip("gift_eval")
    return load_script(EVAL_PY, f"eval_419_{tag}")


def load_in_eval(tmp_path, model, tag, *extra):
    path = tmp_path / f"{tag}.pth"
    torch.save(model.state_dict(), path)
    ev = eval_module(tag)
    argv = ["eval", "--backbone-path", str(path), "--native-value-head",
            "--strategy", "A2", "--device", "cpu", "--n-channels", "1",
            "--d-model", "16", "--n-heads", "2", "--num-layers", "1",
            "--rev-norm-span", "128", *extra]
    from unittest.mock import patch
    with patch.object(sys, "argv", argv):
        args = ev.parse_args()
    backbone, _ = ev.load_models(args, torch.device("cpu"))
    return args, backbone


def eval_backbone(num_freqs, zero_pad):
    return ConfigurableModel(**backbone_kwargs(
        freq_emb_dim=3, num_freqs=num_freqs, seasonality_emb_dim=3,
        value_head_quantiles=len(QUANTILE_LEVELS),
        rev_norm_skip_leading_zeros=zero_pad))


def test_a_v1_checkpoint_loads_and_pads_as_before(tmp_path):
    args, bb = load_in_eval(tmp_path, eval_backbone(10, False), "v1")
    assert bb._freq_vocab == "v1" and args.context_pad == "first"
    assert bb.freq_embedding.embedding.weight.shape[0] == 10
    assert not bb.rev_norm.skip_leading_zeros


def test_a_zero_pad_v2_checkpoint_loads_the_way_it_trained(tmp_path):
    args, bb = load_in_eval(tmp_path, eval_backbone(len(FREQ_NAMES_V2), True), "v2")
    assert bb._freq_vocab == "v2" and args.context_pad == "zeros"
    assert bb.rev_norm.skip_leading_zeros


def test_the_pad_option_overrides_auto(tmp_path):
    args, _ = load_in_eval(tmp_path, eval_backbone(10, False), "over",
                           "--context-pad", "zeros")
    assert args.context_pad == "zeros"


def test_the_predictor_pads_as_asked():
    ev = eval_module("pred")
    cpu = torch.device("cpu")
    first = ev.ContrastiveForecasterPredictor(None, None, 4, cpu, t_raw=8)
    zeros = ev.ContrastiveForecasterPredictor(None, None, 4, cpu, t_raw=8,
                                              context_pad="zeros")
    target = np.array([5, 6, 7], dtype=np.float32)
    assert first._prepare_context(target)[0, :, 0].tolist() == [5] * 6 + [6, 7]
    assert zeros._prepare_context(target)[0, :, 0].tolist() == [0] * 5 + [5, 6, 7]


def test_missing_values_before_the_first_become_padding_under_zeros():
    ev = eval_module("fill")
    cpu = torch.device("cpu")
    first = ev.ContrastiveForecasterPredictor(None, None, 4, cpu, t_raw=6)
    zeros = ev.ContrastiveForecasterPredictor(None, None, 4, cpu, t_raw=6,
                                              context_pad="zeros")
    target = np.array([np.nan, np.nan, 5, np.nan, 7], dtype=np.float32)
    assert first._fill_missing(target).tolist() == [5, 5, 5, 5, 7]
    assert zeros._fill_missing(target).tolist() == [5, 5, 7]
    ctx = zeros._prepare_context(zeros._fill_missing(target))[0, :, 0]
    assert ctx.tolist() == [0, 0, 0, 5, 5, 7]


# ---------------------------------------------------------------------------
# 7. The trainer
# ---------------------------------------------------------------------------

# The #415 cell's data path at a size the CPU trains in seconds.
TINY_GIFT_RUN = (
    "--value-space-objective", "--gift-pretrain", "--freq-vocab", "v2",
    "--t-raw", "4096", "--n-channels", "1", "--d-model", "16",
    "--n-heads", "2", "--num-layers", "1", "--num-encoder-layers", "1",
    "--enc-num-layers", "1", "--enc-nhead", "2", "--batch-size", "4",
    "--synth-kind", "forked-arma", "--mix-ratio", "0.25",
    "--crossfade-triplets", "1", "--mixup-p", "0.3",
    "--freq-emb-dim", "3", "--seasonality-emb-dim", "3",
    "--rev-norm-kind", "ewma", "--rev-norm-span", "128",
    "--train-rollout-depth", "1", "--log-every", "1",
    "--save-every", "1000000")


def run_trainer(*extra):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.run(
        [sys.executable, str(TRAIN_PY), "--device", "cpu",
         "--weight-decay", "0.1", *extra],
        capture_output=True, text=True, env=env, timeout=900)


def test_a_few_trainer_steps_on_the_stream(corpus, tmp_path):
    root, index, _ = corpus
    r = run_trainer(*TINY_GIFT_RUN, "--total-steps", "3",
                    "--gift-pretrain-root", str(root),
                    "--gift-pretrain-index", str(write_index(tmp_path, index)),
                    "--save-dir", str(tmp_path), "--run-name", "g419")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "Salesforce/GiftEvalPretrain (#419)" in r.stdout
    sd = torch.load(tmp_path / "g419_final.pth", map_location="cpu")
    assert "rev_norm.leading_zero_pad" in sd
    assert sd["freq_embedding.embedding.weight"].shape[0] == len(FREQ_NAMES_V2)


@pytest.mark.parametrize("extra,why", [
    (("--gift-pretrain", "--hf-repo", "x"), "replaces --hf-repo"),
    (("--gift-pretrain", "--rev-norm-kind", "revin"), "EWMA normaliser"),
    (("--gift-pretrain-root", "/x"), "add --gift-pretrain")])
def test_the_trainer_refuses_a_line_it_cannot_honour(tmp_path, extra, why):
    r = run_trainer(*extra, "--total-steps", "1", "--save-dir", str(tmp_path))
    assert r.returncode != 0 and why in r.stdout + r.stderr
