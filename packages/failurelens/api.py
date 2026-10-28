from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .demo import generate_demo
from .features import make_features

app = FastAPI(title="FailureLens AI", version="0.1.0")
DATA = Path(__file__).resolve().parents[2] / "data/demo/telemetry.csv"


class Telemetry(BaseModel):
    device_id: str = "api-device"
    model: str = "unknown"
    date: str
    smart_5: float = Field(ge=0)
    smart_187: float = Field(ge=0)
    temperature: float
    read_errors: float = Field(ge=0)


def dataset() -> pd.DataFrame:
    return pd.read_csv(DATA) if DATA.exists() else generate_demo()


@app.get("/health")
def health(): return {"status": "ok", "demo": not DATA.exists()}


@app.get("/api/v1/summary")
def summary():
    df = dataset()
    return {"devices": int(df.device_id.nunique()), "observations": len(df), "demo": not DATA.exists()}


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
    X, _ = make_features(row.assign(failed=0))
    score = min(0.99, max(0.01, float((X["smart_5"] / 50 + X["smart_187"] / 40 + X["read_errors"] / 20) / 3)))
    return {"device_id": payload.device_id, "risk_7d": round(score, 4), "risk_level": "high" if score >= .6 else "medium" if score >= .3 else "low", "demo": True}

