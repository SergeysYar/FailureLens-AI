from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RuntimeMonitor:
    requests: int = 0
    errors: int = 0
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    risk_levels: Counter[str] = field(default_factory=Counter)
    model_versions: Counter[str] = field(default_factory=Counter)

    def observe(self, duration_ms: float, risk_level: str, model_version: str, error: bool = False) -> None:
        self.requests += 1
        self.errors += int(error)
        self.latencies_ms.append(duration_ms)
        self.risk_levels[risk_level] += 1
        self.model_versions[model_version] += 1

    def snapshot(self) -> dict[str, Any]:
        values = np.array(self.latencies_ms, dtype=float)
        return {"requests": self.requests, "errors": self.errors, "error_rate": self.errors / self.requests if self.requests else 0.0,
                "latency_ms": {"p50": float(np.percentile(values, 50)) if len(values) else None, "p95": float(np.percentile(values, 95)) if len(values) else None, "p99": float(np.percentile(values, 99)) if len(values) else None},
                "risk_distribution": dict(self.risk_levels), "model_versions": dict(self.model_versions)}


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with stable quantile bins."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    expected = np.histogram(ref, edges)[0] / len(ref)
    actual = np.histogram(cur, edges)[0] / len(cur)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], threshold: float = .2) -> dict[str, Any]:
    return {"threshold": threshold, "features": {column: {"psi": psi(reference[column], current[column]), "status": "drift" if psi(reference[column], current[column]) >= threshold else "stable", "reference_rows": int(reference[column].notna().sum()), "current_rows": int(current[column].notna().sum())} for column in columns if column in reference and column in current}}


def timer() -> float:
    return time.perf_counter()

