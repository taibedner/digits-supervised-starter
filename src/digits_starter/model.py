"""A readable baseline: standardize pixels, then classify with logistic regression."""

import math

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(c=1.0, max_iter=2000, seed=42):
    if not math.isfinite(c) or c <= 0:
        raise ValueError("C must be a positive finite number")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    return Pipeline([
        # fit() estimates each pixel's mean and scale using TRAINING images only.
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=c, max_iter=max_iter, solver="lbfgs", random_state=seed)),
    ])

