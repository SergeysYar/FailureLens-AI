from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo(n_devices: int = 24, days: int = 120, seed: int = 7) -> pd.DataFrame:
    """Create deterministic telemetry with failures caused by prior degradation."""
    rng = np.random.default_rng(seed)
    rows = []
    for device_idx in range(n_devices):
        device = f"demo-{device_idx:03d}"
        low = max(3, days // 3)
        high = max(low + 1, days - 3)
        failure_day = int(rng.integers(low, high)) if device_idx % 4 == 0 else None
        model = ["DemoDrive X", "DemoDrive Pro", "DemoDrive Lite"][device_idx % 3]
        for day in range(days):
            degradation = max(0.0, day - (failure_day or days + 10)) / 30
            smart_5 = max(0, rng.normal(2 + 35 * degradation, 1.5))
            smart_187 = max(0, rng.normal(1 + 22 * degradation, 1.0))
            temperature = rng.normal(32 + 5 * degradation, 1.8)
            errors = max(0, rng.poisson(0.4 + 4 * degradation))
            failed = int(failure_day is not None and day >= failure_day)
            rows.append({"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day), "device_id": device,
                         "model": model, "smart_5": smart_5, "smart_187": smart_187,
                         "temperature": temperature, "read_errors": errors, "failed": failed})
    return pd.DataFrame(rows)


def write_demo(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    generate_demo().to_csv(target, index=False)
    return target

