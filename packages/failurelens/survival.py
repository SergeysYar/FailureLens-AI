from __future__ import annotations

import pandas as pd


def kaplan_meier(frame: pd.DataFrame) -> dict:
    """Small dependency-free Kaplan–Meier estimator for device-level lifetimes.

    A device with no registered failure is right-censored at its last observation.
    This curve is descriptive and is not directly comparable to classification PR-AUC.
    """
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    for device, group in df.sort_values("date").groupby("device_id"):
        start = group["date"].min()
        failures = group.loc[group.get("failed", 0).eq(1), "date"] if "failed" in group else pd.Series(dtype="datetime64[ns]")
        event_date = failures.min() if not failures.empty else group["date"].max()
        rows.append({"device_id": device, "duration_days": max(0, int((event_date - start).days)), "event": int(not failures.empty)})
    durations = pd.DataFrame(rows)
    curve = []
    survival = 1.0
    for time in sorted(durations["duration_days"].unique()):
        at_risk = int((durations["duration_days"] >= time).sum())
        events = int(((durations["duration_days"] == time) & durations["event"].eq(1)).sum())
        if at_risk and events:
            survival *= 1 - events / at_risk
        curve.append({"time_days": int(time), "at_risk": at_risk, "events": events, "survival": round(float(survival), 8)})
    return {"status": "executed", "devices": len(durations), "events": int(durations["event"].sum()), "right_censored": int((1 - durations["event"]).sum()), "curve": curve}
