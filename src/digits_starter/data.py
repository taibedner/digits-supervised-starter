"""Read this first: image inputs X, correct labels y, and independent splits."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Split:
    images: np.ndarray
    targets: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray

    def indices(self, part):
        if part not in {"train", "validation", "test"}:
            raise ValueError("part must be train, validation or test")
        return getattr(self, f"{part}_indices")

    def arrays(self, part):
        """Turn (N, 8, 8) images into (N, 64) feature vectors."""
        indices = self.indices(part)
        return self.images[indices].reshape(-1, 64), self.targets[indices]


def load_split(seed=42):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer from 0 through 2**32 - 1")
    dataset = load_digits()
    indices = np.arange(len(dataset.target))
    # Reserve 20% for the final test; split the remainder into 60% / 20%.
    train_validation, test = train_test_split(
        indices, test_size=0.2, stratify=dataset.target, random_state=seed,
    )
    train, validation = train_test_split(
        train_validation, test_size=0.25, stratify=dataset.target[train_validation],
        random_state=seed,
    )
    return Split(dataset.images, dataset.target, train, validation, test)

