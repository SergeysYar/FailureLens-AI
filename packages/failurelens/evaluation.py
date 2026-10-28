from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import make_features


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def chronological_split(dates, train_ratio: float = .6, validation_ratio: float = .2) -> Split:
    """Split by timestamps, never by shuffled rows."""
    order = np.argsort(np.asarray(dates, dtype="datetime64[ns]"))
    n = len(order)
    train_end = max(1, int(n * train_ratio))
    validation_end = max(train_end + 1, int(n * (train_ratio + validation_ratio)))
    return Split(order[:train_end], order[train_end:validation_end], order[validation_end:])


def _metrics(y_true, probabilities, threshold: float = .5) -> dict[str, float]:
    result: dict[str, float] = {"rows": float(len(y_true)), "positive_rate": float(np.mean(y_true))}
    predictions = (probabilities >= threshold).astype(int)
    result.update({"precision": float(precision_score(y_true, predictions, zero_division=0)),
                   "recall": float(recall_score(y_true, predictions, zero_division=0)),
                   "brier_score": float(brier_score_loss(y_true, probabilities))})
    if len(np.unique(y_true)) > 1:
        result["pr_auc"] = float(average_precision_score(y_true, probabilities))
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return result


def _probabilities(model, X):
    probabilities = model.predict_proba(X)
    if probabilities.shape[1] == 1:
        return np.full(len(X), float(model.classes_[0]))
    return probabilities[:, 1]


def run_experiments(frame, horizon: int = 7) -> dict[str, Any]:
    X, y = make_features(frame, horizon=horizon)
    dates = pd_to_dates(frame, len(X))
    split = chronological_split(dates)
    output: dict[str, Any] = {"horizon_days": horizon, "protocol": "chronological 60/20/20", "models": {}}
    models = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500, random_state=7)),
    }
    for name, model in models.items():
        model.fit(X.iloc[split.train], y.iloc[split.train])
        validation_prob = _probabilities(model, X.iloc[split.validation])
        test_prob = _probabilities(model, X.iloc[split.test])
        output["models"][name] = {"validation": _metrics(y.iloc[split.validation].to_numpy(), validation_prob),
                                  "test": _metrics(y.iloc[split.test].to_numpy(), test_prob)}
    return output


def pd_to_dates(frame, n: int):
    import pandas as pd
    dates = frame.copy()
    dates["date"] = pd.to_datetime(dates["date"])
    dates = dates.sort_values(["device_id", "date"]).drop_duplicates(["device_id", "date"])
    return dates["date"].iloc[:n].to_numpy()

