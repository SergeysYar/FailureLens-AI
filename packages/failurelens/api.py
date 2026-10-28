from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .demo import generate_demo
from .explainability import explain_linear
from .features import make_features

app = FastAPI(title="FailureLens AI", version="0.2.0", description="Predictive equipment-failure intelligence API")
DATA = Path(__file__).resolve().parents[2] / "data/demo/telemetry.csv"


class Telemetry(BaseModel):
    device_id: str = "api-device"
    model: str = "unknown"
    date: str
    smart_5: float = Field(ge=0)
    smart_187: float = Field(ge=0)
    temperature: float
    read_errors: float = Field(ge=0)
    horizon_days: Literal[7, 30] = 7


def dataset() -> pd.DataFrame:
    return pd.read_csv(DATA) if DATA.exists() else generate_demo()


@app.get("/health")
def health(): return {"status": "ok", "demo": not DATA.exists()}


@app.get("/api/v1/summary")
def summary():
    df = dataset()
    return {"devices": int(df.device_id.nunique()), "observations": len(df), "demo": not DATA.exists()}


@app.get("/api/v1/model")
def model_summary():
    import json
    artifact = Path(__file__).resolve().parents[2] / "artifacts/baseline_7d.json"
    return json.loads(artifact.read_text(encoding="utf-8")) if artifact.exists() else {"status": "not_trained", "message": "Run `make train` first."}


@app.get("/api/v1/experiments")
def experiments():
    import json
    root = Path(__file__).resolve().parents[2] / "artifacts/experiments"
    files = sorted(root.glob("horizon_*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


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
    row = pd.DataFrame([payload.model_dump()])
    horizon = payload.horizon_days
    row = row.drop(columns=["horizon_days"])
    X, _ = make_features(row.assign(failed=0), horizon=horizon)
    artifact = Path(__file__).resolve().parents[2] / f"artifacts/baseline_{horizon}d.joblib"
    if not artifact.exists():
        raise HTTPException(503, "Model is not trained. Run `make train` first.")
    import joblib
    bundle = joblib.load(artifact)
    score = float(bundle["model"].predict_proba(X[bundle["feature_columns"]])[:, 1][0])
    return {"device_id": payload.device_id, "horizon_days": horizon, "risk": round(score, 4), "risk_level": "high" if score >= .6 else "medium" if score >= .3 else "low", "explanations": explain_linear(artifact, X.iloc[0].to_dict()), "demo": not DATA.exists()}

