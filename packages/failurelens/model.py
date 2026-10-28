from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import make_features


def train(frame: pd.DataFrame, artifact: str | Path) -> dict:
    X, y = make_features(frame)
    split = max(1, int(len(X) * 0.8))
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500, random_state=7))
    model.fit(X.iloc[:split], y.iloc[:split])
    probabilities = model.predict_proba(X.iloc[split:])[:, 1] if len(X) > split else []
    metrics = {"rows": len(X), "positive_rate": float(y.mean())}
    if len(set(y.iloc[split:])) > 1:
        metrics.update({"pr_auc": float(average_precision_score(y.iloc[split:], probabilities)),
                        "roc_auc": float(roc_auc_score(y.iloc[split:], probabilities))})
    target = Path(artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, target)
    target.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

