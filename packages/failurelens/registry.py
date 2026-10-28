from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATES = {"candidate", "staging", "production", "archived"}


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "models": [], "active": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def register(path: str | Path, artifact: str | Path, metadata: dict[str, Any], version: str | None = None) -> dict[str, Any]:
    path = Path(path)
    payload = _read(path)
    version = version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    record = {"model_id": f"failurelens-{version}", "version": version, "artifact": str(artifact), "state": "candidate", "registered_at": datetime.now(timezone.utc).isoformat(), **metadata}
    payload["models"] = [item for item in payload["models"] if item["version"] != version] + [record]
    _write(path, payload)
    return record


def transition(path: str | Path, version: str, state: str) -> dict[str, Any]:
    if state not in STATES:
        raise ValueError(f"Unknown model state: {state}")
    path = Path(path)
    payload = _read(path)
    matches = [item for item in payload["models"] if item["version"] == version]
    if not matches:
        raise KeyError(f"Model version not found: {version}")
    if state == "production":
        for item in payload["models"]:
            if item.get("state") == "production":
                item["state"] = "archived"
        payload.setdefault("active", {})[str(matches[0].get("horizon_days", 7))] = version
    matches[0]["state"] = state
    _write(path, payload)
    return matches[0]


def list_models(path: str | Path) -> list[dict[str, Any]]:
    return _read(Path(path))["models"]


def active(path: str | Path, horizon: int = 7) -> dict[str, Any] | None:
    payload = _read(Path(path))
    version = payload.get("active", {}).get(str(horizon))
    return next((item for item in payload["models"] if item["version"] == version), None)

