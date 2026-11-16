"""Model definitions for bounded halting prediction.

All three classifiers must cope with class imbalance: in the 'hard' population only
~9% of pairs halt within L steps.
  Decision Tree / Random Forest : class_weight="balanced" (errors on the rare class cost more)
  Neural Net (MLP)              : sklearn's MLPClassifier has no class_weight option, so
                                  BalancedMLP fixes imbalance itself (see below).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _oversample_minority(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Duplicate random minority-class rows until both classes are equally frequent."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or counts[0] == counts[1]:
        return X, y
    minority = classes[np.argmin(counts)]
    idx_min = np.where(y == minority)[0]
    extra = rng.choice(idx_min, size=abs(counts[0] - counts[1]), replace=True)
    idx = rng.permutation(np.concatenate([np.arange(len(y)), extra]))
    return X[idx], y[idx]


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Probability cut-off that maximises F1 on (y_true, prob). Falls back to 0.5."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # precision/recall have one more entry than thresholds; drop the last point
    return float(thresholds[int(np.argmax(f1[:-1]))])


class BalancedMLP(BaseEstimator, ClassifierMixin):
    """StandardScaler + MLPClassifier (64-32 hidden units, 60 epochs), made imbalance-aware.

    tune_threshold=True (default, used in the experiments):
        1. split the TRAINING data into 80% fit / 20% validation (stratified)
        2. train the network on the 80%
        3. pick the probability threshold that maximises F1 on the 20% validation part
        `predict` then uses that threshold instead of the usual 0.5.
    oversample=True:
        duplicate minority-class rows of the training data before fitting.

    Both happen strictly inside `fit`, so the test fold is never touched.
    With both options off this is a plain, unweighted MLP (kept for the ablation study).
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        max_iter: int = 60,
        random_state: int = 42,
        tune_threshold: bool = True,
        oversample: bool = False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.tune_threshold = tune_threshold
        self.oversample = oversample

    def _new_pipeline(self):
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                # No early stopping: sklearn's early stopping watches validation ACCURACY,
                # which on 9%-positive data plateaus at "predict all 0" and stops training
                # after a few epochs (unstable results). A fixed 60 epochs is stable and fast.
                early_stopping=False,
                random_state=self.random_state,
            ),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BalancedMLP":
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        self.threshold_ = 0.5

        # Hold out a validation part only if both classes have enough members for it
        can_tune = (
            self.tune_threshold
            and len(self.classes_) == 2
            and np.bincount(y.astype(int)).min() >= 5
        )
        if can_tune:
            fit_idx, val_idx = train_test_split(
                np.arange(len(y)), test_size=0.2, random_state=self.random_state, stratify=y
            )
        else:
            fit_idx, val_idx = np.arange(len(y)), None

        X_fit, y_fit = X[fit_idx], y[fit_idx]
        if self.oversample:
            X_fit, y_fit = _oversample_minority(X_fit, y_fit, rng)

        self.pipe_ = self._new_pipeline()
        self.pipe_.fit(X_fit, y_fit)

        if val_idx is not None:
            val_prob = self.pipe_.predict_proba(X[val_idx])[:, 1]
            self.threshold_ = best_f1_threshold(y[val_idx], val_prob)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipe_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(self.classes_) < 2:
            return np.full(len(X), self.classes_[0])
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(self.classes_.dtype)


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
            hidden_layer_sizes=(64, 32), max_iter=60, random_state=seed
        ),
    }


# Variants compared in the MLP imbalance ablation (experiment.py)
MLP_VARIANTS = {
    "MLP, unweighted (threshold 0.5)": dict(tune_threshold=False, oversample=False),
    "MLP, oversampled (threshold 0.5)": dict(tune_threshold=False, oversample=True),
    "MLP, threshold tuned on validation": dict(tune_threshold=True, oversample=False),
}
