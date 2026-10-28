from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import _probabilities, chronological_split, supervised_data


def train(frame: pd.DataFrame, artifact: str | Path, horizon: int = 7) -> dict:
    X, y, dates = supervised_data(frame, horizon)
    split = chronological_split(dates)
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500, random_state=7))
    model.fit(X.iloc[split.train], y.iloc[split.train])
    probabilities = _probabilities(model, X.iloc[split.test]) if len(split.test) else []
    metrics = {"rows": len(X), "positive_rate": float(y.mean()), "horizon_days": horizon, "protocol": "chronological holdout"}
    if len(set(y.iloc[split.test])) > 1:
        metrics.update({"pr_auc": float(average_precision_score(y.iloc[split.test], probabilities)),
                        "roc_auc": float(roc_auc_score(y.iloc[split.test], probabilities))})
    target = Path(artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, target)
    target.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

