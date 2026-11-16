"""Model definitions for bounded halting prediction."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class BalancedMLP(BaseEstimator, ClassifierMixin):
    """StandardScaler + MLPClassifier with training-fold minority oversampling.

    Oversampling happens strictly inside `fit(X, y)` so test folds remain untouched.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        max_iter: int = 350,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BalancedMLP":
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            c0, c1 = self.classes_[0], self.classes_[1]
            idx0 = np.where(y == c0)[0]
            idx1 = np.where(y == c1)[0]
            if len(idx0) > 0 and len(idx1) > 0 and len(idx0) != len(idx1):
                if len(idx1) < len(idx0):
                    extra = rng.choice(idx1, size=len(idx0) - len(idx1), replace=True)
                    idx = np.concatenate([idx0, idx1, extra])
                else:
                    extra = rng.choice(idx0, size=len(idx1) - len(idx0), replace=True)
                    idx = np.concatenate([idx0, idx1, extra])
                rng.shuffle(idx)
                X = X[idx]
                y = y[idx]

        self.pipe_ = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                early_stopping=True,
                random_state=self.random_state,
            ),
        )
        self.pipe_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipe_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipe_.predict_proba(X)


def make_models(seed: int = 42) -> dict:
    """Return fresh classifier instances for Decision Tree, Random Forest, and Balanced MLP."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "Neural Net (MLP)": BalancedMLP(
            hidden_layer_sizes=(64, 32), max_iter=350, random_state=seed
        ),
    }

