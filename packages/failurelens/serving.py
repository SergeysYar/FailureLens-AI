from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class ModelStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._cache: dict[int, tuple[float, dict[str, Any]]] = {}

    def load(self, horizon: int) -> dict[str, Any]:
        path = self.root / f"baseline_{horizon}d.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model artifact is missing: {path}")
        stamp = path.stat().st_mtime
        cached = self._cache.get(horizon)
        if cached and cached[0] == stamp:
            return cached[1]
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "model" not in bundle or "feature_columns" not in bundle:
            raise ValueError(f"Invalid model artifact: {path}")
        self._cache[horizon] = (stamp, bundle)
        return bundle

    def status(self, horizon: int) -> dict[str, Any]:
        path = self.root / f"baseline_{horizon}d.joblib"
        try:
            bundle = self.load(horizon)
            return {"ready": True, "horizon_days": horizon, "artifact": str(path), "feature_count": len(bundle["feature_columns"]), "modified_at": path.stat().st_mtime}
        except (FileNotFoundError, ValueError, OSError) as exc:
            return {"ready": False, "horizon_days": horizon, "artifact": str(path), "error": str(exc)}

