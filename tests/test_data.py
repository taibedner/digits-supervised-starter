import numpy as np
import pytest

from digits_starter.data import load_split


def test_splits_are_disjoint_and_complete(split):
    train = set(split.train_indices)
    validation = set(split.validation_indices)
    test = set(split.test_indices)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == set(range(len(split.targets)))


@pytest.mark.parametrize("part", ["train", "validation", "test"])
def test_each_split_contains_all_classes(split, part):
    x, y = split.arrays(part)
    assert x.shape == (len(y), 64)
    assert set(y) == set(range(10))
    assert np.isfinite(x).all()
    assert x.min() >= 0 and x.max() <= 16


def test_same_seed_reproduces_split(split):
    again = load_split(42)
    np.testing.assert_array_equal(split.train_indices, again.train_indices)
    np.testing.assert_array_equal(split.test_indices, again.test_indices)


def test_different_seed_changes_split(split):
    assert not np.array_equal(split.test_indices, load_split(7).test_indices)


@pytest.mark.parametrize("seed", [-1, 2**32, True, "42"])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        load_split(seed)

