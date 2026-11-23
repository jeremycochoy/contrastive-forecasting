"""--batch-size counts every row of a training step, the crossfade triplets too
(#425): the loader samples batch_size - 3 * triplets rows and appends the
triplet rows. --crossfade-triplets-on-top keeps the count of the older runs."""

import pytest

from src.dataloader import step_rows


def test_the_triplet_rows_count_in_the_batch():
    rows = step_rows(256, 1 / 128, 0.0, 1)
    assert rows == {"sampled": 253, "real": 251, "synth": 2, "cross": 0,
                    "triplet": 3, "total": 256}


def test_on_top_keeps_the_count_of_the_older_runs():
    rows = step_rows(256, 1 / 128, 0.0, 1, triplets_on_top=True)
    assert rows == {"sampled": 256, "real": 254, "synth": 2, "cross": 0,
                    "triplet": 3, "total": 259}


def test_a_batch_with_no_triplet_does_not_change():
    rows = step_rows(256, 1 / 128)
    assert (rows["sampled"], rows["real"], rows["total"]) == (256, 254, 256)


def test_synthetic_and_crossfade_rows_are_fractions_of_the_sampled_rows():
    rows = step_rows(256, 0.25, 0.25, 2)
    assert rows["sampled"] == 250 and rows["total"] == 256
    assert rows["real"] + rows["synth"] + rows["cross"] == rows["sampled"]


def test_a_batch_with_no_real_row_is_refused():
    with pytest.raises(ValueError):
        step_rows(3, 0.0, 0.0, 1)
