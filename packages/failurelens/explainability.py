from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


def explain_linear(artifact: str | Path, values: dict[str, float], limit: int = 5) -> list[dict[str, float | str]]:
    """Return signed, model-space contributions for the supported baseline."""
    bundle = joblib.load(artifact)
    model = bundle["model"]
    columns = bundle["feature_columns"]
    estimator = model[-1]
    scaler = model[0]
    vector = np.array([float(values.get(column, 0.0)) for column in columns])
    contributions = estimator.coef_[0] * ((vector - scaler.mean_) / scaler.scale_)
    ranked = np.argsort(np.abs(contributions))[::-1][:limit]
    return [{"feature": columns[i], "contribution": round(float(contributions[i]), 6), "direction": "increases_risk" if contributions[i] > 0 else "reduces_risk"} for i in ranked]

