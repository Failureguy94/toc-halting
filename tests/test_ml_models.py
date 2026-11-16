"""Tests for ML models and dataset/experiment helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_models import BalancedMLP, make_models


def test_balanced_mlp_handles_imbalance():
    rng = np.random.default_rng(0)
    X_neg = rng.normal(loc=0.0, scale=1.0, size=(180, 6))
    X_pos = rng.normal(loc=2.5, scale=1.0, size=(20, 6))
    X = np.vstack([X_neg, X_pos])
    y = np.array([0] * 180 + [1] * 20)

    clf = BalancedMLP(hidden_layer_sizes=(16, 8), max_iter=100, random_state=42)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    preds = clf.predict(X)
    assert probs.shape == (200, 2)
    assert preds.shape == (200,)
    # Balanced training should recover the minority class instead of predicting all zeros
    assert preds[180:].sum() >= 10


def test_make_models_returns_all_three():
    models = make_models(seed=42)
    assert set(models.keys()) == {"Decision Tree", "Random Forest", "Neural Net (MLP)"}

