from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
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

from .evaluation import _probabilities, chronological_split
from .features import feature_groups, make_supervised
from .survival import kaplan_meier


def _model(name: str):
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500, random_state=7))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=120, min_samples_leaf=3, class_weight="balanced", random_state=7, n_jobs=1)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=160, depth=5, learning_rate=.05, loss_function="Logloss", eval_metric="PRAUC", random_seed=7, verbose=False, thread_count=1)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=160, learning_rate=.05, num_leaves=15, random_state=7, verbosity=-1)
    raise ValueError(f"Unsupported model: {name}")


def _metrics(y, p) -> dict[str, float]:
    result = {"rows": len(y), "positive_rate": float(np.mean(y)), "brier_score": float(brier_score_loss(y, p))}
    result.update({"precision": float(precision_score(y, p >= .5, zero_division=0)), "recall": float(recall_score(y, p >= .5, zero_division=0))})
    if len(np.unique(y)) > 1:
        result.update({"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p))})
    return result


def _bootstrap_pr_auc(y, p, seed: int = 7, rounds: int = 200) -> dict[str, float] | None:
    if len(np.unique(y)) < 2:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) > 1:
            values.append(average_precision_score(y[indices], p[indices]))
    if not values:
        return None
    return {"lower_95": float(np.quantile(values, .025)), "median": float(np.quantile(values, .5)), "upper_95": float(np.quantile(values, .975)), "rounds": len(values)}


def _fingerprint(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.sort_index(axis=1).to_csv(index=False).encode()).hexdigest()[:16]


def run_advanced(frame: pd.DataFrame, horizon: int = 7, quick: bool = True) -> dict[str, Any]:
    X, y, eligible = make_supervised(frame, horizon=horizon)
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized = normalized.sort_values(["device_id", "date"]).drop_duplicates(["device_id", "date"])
    dates = normalized.loc[eligible, "date"].reset_index(drop=True).to_numpy()
    split = chronological_split(dates)
    groups = feature_groups()
    model_names = ["dummy", "logistic_regression"] if quick else ["dummy", "logistic_regression", "random_forest"]
    if not quick and importlib.util.find_spec("catboost"):
        model_names.append("catboost")
    if not quick and importlib.util.find_spec("lightgbm"):
        model_names.append("lightgbm")
    result: dict[str, Any] = {"status": "executed", "horizon_days": horizon, "mode": "quick" if quick else "full", "protocol": "chronological 60/20/20", "dataset_fingerprint": _fingerprint(frame), "eligible_rows": len(y), "censoring": "right-censored rows excluded from forward-label evaluation", "models": {}, "optional_models": {}, "tracking": "MLflow available" if importlib.util.find_spec("mlflow") else "MLflow not installed; JSON manifest used"}
    for package, label in (("catboost", "CatBoost"), ("lightgbm", "LightGBM"), ("torch", "GRU/TCN"), ("lifelines", "Cox survival")):
        result["optional_models"][label] = "available" if importlib.util.find_spec(package) else "not_installed / not_executed"
    for name in model_names:
        model = _model(name)
        model.fit(X.iloc[split.train], y.iloc[split.train])
        validation_p = _probabilities(model, X.iloc[split.validation])
        test_p = _probabilities(model, X.iloc[split.test])
        test_y = y.iloc[split.test].to_numpy()
        result["models"][name] = {"validation": _metrics(y.iloc[split.validation].to_numpy(), validation_p), "test": _metrics(test_y, test_p), "test_pr_auc_ci": _bootstrap_pr_auc(test_y, test_p)}
        if name == "logistic_regression":
            result["alert_budget"] = {f"top_{budget}_percent_recall": float(recall_score(test_y, np.isin(np.arange(len(test_p)), np.argsort(test_p)[-max(1, int(len(test_p) * budget / 100)):]), zero_division=0)) for budget in (1, 5, 10)}
    result["ablation"] = _ablation(X, y, groups, split)
    result["backtest"] = _backtest(frame, normalized, eligible, dates, split, X, y)
    result["survival"] = kaplan_meier(frame)
    result["reproducibility"] = {"seed": 7, "git_commit": _git_commit(), "feature_groups": sorted(set(groups.values())), "feature_count": int(X.shape[1])}
    return result


def _ablation(X, y, groups, split) -> dict[str, Any]:
    output = {}
    for group in sorted(set(groups.values())):
        columns = [column for column in X.columns if groups.get(column) == group]
        if not columns:
            continue
        model = _model("logistic_regression")
        model.fit(X.loc[:, columns].iloc[split.train], y.iloc[split.train])
        p = _probabilities(model, X.loc[:, columns].iloc[split.test])
        output[group] = _metrics(y.iloc[split.test].to_numpy(), p)
    return output


def _backtest(frame, normalized, eligible, dates, split, X, y) -> dict[str, Any]:
    model = _model("logistic_regression")
    model.fit(X.iloc[split.train], y.iloc[split.train])
    p = _probabilities(model, X.iloc[split.test])
    test = normalized.reset_index(drop=True).loc[eligible].reset_index(drop=True).iloc[split.test].copy()
    test["score"] = p
    alerts = test["score"] >= test["score"].quantile(.95)
    forward_events = y.iloc[split.test].to_numpy().astype(bool)
    return {"status": "executed", "alert_budget": "top 5% score threshold on test", "alerts": int(alerts.sum()), "test_rows": len(test), "forward_failure_rows_in_test": int(forward_events.sum()), "alert_precision_at_threshold": float(forward_events[alerts.to_numpy()].mean()) if alerts.any() else 0.0, "lead_time_definition": "days from prediction timestamp to first registered failure within horizon"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:12]
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_report(result: dict[str, Any], root: str | Path) -> list[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"research_h{result['horizon_days']}_{result['mode']}.json"
    md_path = root / f"research_h{result['horizon_days']}_{result['mode']}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [f"# FailureLens AI Research — {result['horizon_days']}-day horizon", "", f"Status: **{result['status']}**", f"Dataset fingerprint: `{result['dataset_fingerprint']}`", f"Protocol: {result['protocol']}", "", "## Model comparison", "", "| Model | Test PR-AUC | Test ROC-AUC | Recall | Brier |", "|---|---:|---:|---:|---:|"]
    for name, values in result["models"].items():
        test = values["test"]
        lines.append(f"| {name} | {test.get('pr_auc', 'n/a')} | {test.get('roc_auc', 'n/a')} | {test.get('recall', 'n/a')} | {test.get('brier_score', 'n/a')} |")
    lines += ["", "## Scientific status", "", "Optional CatBoost, LightGBM, deep sequence, and survival packages are reported explicitly in the JSON. No unavailable method is represented as executed."]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return [json_path, md_path]

