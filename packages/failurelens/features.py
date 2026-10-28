from __future__ import annotations

import pandas as pd

BASE = ["smart_5", "smart_187", "temperature", "read_errors"]


def make_features(frame: pd.DataFrame, horizon: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Build point-in-time features and forward failure labels per device."""
    required = {"date", "device_id", *BASE}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["device_id", "date"]).drop_duplicates(["device_id", "date"])
    grouped = df.groupby("device_id", group_keys=False)
    for col in BASE:
        df[f"{col}_roll7_mean"] = grouped[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
        df[f"{col}_delta7"] = grouped[col].transform(lambda s: s - s.shift(7))
    df["observations"] = grouped.cumcount() + 1
    failure_flag = df["failed"] if "failed" in df.columns else pd.Series(0, index=df.index)
    failure_dates = df.loc[failure_flag.eq(1)].groupby("device_id")["date"].min()
    y = pd.Series(0, index=df.index, dtype="int64")
    for device, failure_date in failure_dates.items():
        y.loc[(df["device_id"] == device) & (df["date"] < failure_date) &
              (df["date"] >= failure_date - pd.Timedelta(days=horizon))] = 1
    feature_cols = [c for c in df.columns if c.endswith(("_roll7_mean", "_delta7"))] + BASE + ["observations"]
    X = df[feature_cols].replace([float("inf"), -float("inf")], pd.NA).fillna(0.0)
    return X, y

