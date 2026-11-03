import pytest

from digits_starter.data import load_split
from digits_starter.model import make_model


@pytest.fixture(scope="session")
def split():
    return load_split(seed=42)


@pytest.fixture(scope="session")
def fitted_model(split):
    model = make_model()
    x, y = split.arrays("train")
    model.fit(x, y)
    return model

