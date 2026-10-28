from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .demo import generate_demo
from .explainability import explain_linear
from .features import make_features
from .monitoring import RuntimeMonitor, timer
from .registry import list_models
from .serving import ModelStore

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("failurelens.api")
app = FastAPI(title="FailureLens AI", version="0.4.0", description="Predictive equipment-failure intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins), allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
DATA = Path(__file__).resolve().parents[2] / "data/demo/telemetry.csv"
ROOT = Path(__file__).resolve().parents[2]
store = ModelStore(ROOT / settings.model_dir)
monitor = RuntimeMonitor()


class Telemetry(BaseModel):
    device_id: str = "api-device"
    model: str = "unknown"
    date: str
    smart_5: float = Field(ge=0)
    smart_187: float = Field(ge=0)
    temperature: float
    read_errors: float = Field(ge=0)
    horizon_days: Literal[7, 30] = 7


class BatchTelemetry(BaseModel):
    records: list[Telemetry] = Field(min_length=1, max_length=1000)


def dataset() -> pd.DataFrame:
    return pd.read_csv(DATA) if DATA.exists() else generate_demo()


@app.get("/health")
def health(): return {"status": "ok", "demo": not DATA.exists(), "environment": settings.environment}


@app.get("/health/live")
def live(): return {"status": "alive"}


@app.get("/health/ready")
def ready():
    statuses = [store.status(horizon) for horizon in (7, 30)]
    return {"status": "ready" if any(item["ready"] for item in statuses) else "not_ready", "models": statuses}


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    started = timer()
    try:
        response = await call_next(request)
        error = response.status_code >= 500
        return response
    except Exception:
        error = True
        raise
    finally:
        duration = (timer() - started) * 1000
        monitor.observe(duration, "unknown", "unknown", error=error)
        logger.info("request_id=%s method=%s path=%s duration_ms=%.2f", request_id, request.method, request.url.path, duration)


def _predict(payload: Telemetry) -> dict:
    row = pd.DataFrame([payload.model_dump()])
    horizon = payload.horizon_days
    row = row.drop(columns=["horizon_days"])
    X, _ = make_features(row.assign(failed=0), horizon=horizon)
    bundle = store.load(horizon)
    score = float(bundle["model"].predict_proba(X[bundle["feature_columns"]])[:, 1][0])
    level = "high" if score >= .6 else "medium" if score >= .3 else "low"
    monitor.risk_levels[level] += 1
    model_version = f"baseline-{horizon}d"
    monitor.model_versions[model_version] += 1
    return {"device_id": payload.device_id, "horizon_days": horizon, "risk": round(score, 4), "risk_level": level, "model_version": model_version, "feature_schema_version": "temporal-v3", "explanations": explain_linear(ROOT / settings.model_dir / f"baseline_{horizon}d.joblib", X.iloc[0].to_dict()), "demo": not DATA.exists()}


@app.get("/api/v1/summary")
def summary():
    df = dataset()
    return {"devices": int(df.device_id.nunique()), "observations": len(df), "demo": not DATA.exists()}


@app.get("/api/v1/model")
def model_summary():
    artifact = ROOT / settings.model_dir / "baseline_7d.json"
    return json.loads(artifact.read_text(encoding="utf-8")) if artifact.exists() else {"status": "not_trained", "message": "Run `make train` first."}


@app.get("/api/v1/experiments")
def experiments():
    import json
    root = Path(__file__).resolve().parents[2] / "artifacts/experiments"
    files = sorted(root.glob("horizon_*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


@app.get("/api/v1/research")
def research_results():
    root = Path(__file__).resolve().parents[2] / "artifacts/research"
    files = sorted(root.glob("research_h*_*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


@app.get("/api/v1/model/status")
def model_status(): return {"models": [store.status(horizon) for horizon in (7, 30)]}


@app.get("/api/v1/model/versions")
def model_versions(): return list_models(ROOT / settings.registry_path)


@app.get("/api/v1/model/metadata")
def model_metadata():
    return {"feature_schema_version": "temporal-v3", "horizons": [7, 30], "artifacts": [store.status(horizon) for horizon in (7, 30)]}


@app.get("/api/v1/monitoring/status")
def monitoring_status(): return monitor.snapshot()


@app.get("/metrics")
def metrics():
    snapshot = monitor.snapshot()
    lines = [f"failurelens_requests_total {snapshot['requests']}", f"failurelens_errors_total {snapshot['errors']}"]
    if snapshot["latency_ms"]["p95"] is not None:
        lines.append(f"failurelens_request_latency_p95_ms {snapshot['latency_ms']['p95']}")
    for level, count in snapshot["risk_distribution"].items(): lines.append(f'failurelens_predictions_total{{risk_level="{level}"}} {count}')
    return "\n".join(lines) + "\n"


@app.get("/api/v1/devices")
def devices():
    df = dataset()
    return df.groupby("device_id").tail(1)[["device_id", "model", "date"]].to_dict("records")


@app.get("/api/v1/devices/{device_id}")
def device(device_id: str):
    rows = dataset().query("device_id == @device_id")
    if rows.empty: raise HTTPException(404, "Device not found")
    return {"device_id": device_id, "model": rows.model.iloc[0], "telemetry": rows.tail(30).to_dict("records")}


@app.post("/api/v1/predict")
def predict(payload: Telemetry):
    try:
        return _predict(payload)
    except FileNotFoundError as exc:
        raise HTTPException(503, "Model is not trained. Run `make train` first.") from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, f"Inference schema error: {exc}") from exc


@app.post("/api/v1/predict/batch")
def predict_batch(payload: BatchTelemetry):
    try:
        return {"predictions": [_predict(record) for record in payload.records], "count": len(payload.records)}
    except FileNotFoundError as exc:
        raise HTTPException(503, "Model is not trained. Run `make train` first.") from exc

