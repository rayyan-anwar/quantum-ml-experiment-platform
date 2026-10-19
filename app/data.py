"""Small 2-D binary classification datasets (sklearn generators and the bundled Iris data) with angle scaling.

Author: Rayyan Anwar
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATASETS = ("moons", "circles", "xor_blobs", "iris_vv")


def load_dataset(name: str, seed: int = 0, n: int = 240):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset '{name}'; choose from {DATASETS}")
    if not 40 <= n <= 2000:
        raise ValueError("n must be in [40, 2000]")
    if name == "moons":
        return make_moons(n, noise=0.2, random_state=seed)
    if name == "circles":
        return make_circles(n, noise=0.1, factor=0.5, random_state=seed)
    if name == "xor_blobs":
        rng = np.random.default_rng(seed)
        centers = rng.choice([-1.0, 1.0], size=(n, 2))
        X = centers + rng.normal(0, 0.55, size=(n, 2))
        return X, (centers[:, 0] * centers[:, 1] > 0).astype(int)
    iris = load_iris()  # versicolor (1) vs virginica (2) on petal length and width
    keep = iris.target > 0
    return iris.data[keep][:, 2:4], (iris.target[keep] == 2).astype(int)


def split_scale(X, y, seed: int = 0, test_size: float = 0.4):
    """Stratified split; features min-max scaled to [0, pi] using TRAIN statistics only."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    sc = MinMaxScaler(feature_range=(0, np.pi)).fit(Xtr)
    return sc.transform(Xtr), np.clip(sc.transform(Xte), 0, np.pi), ytr, yte
