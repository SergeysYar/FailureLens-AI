from __future__ import annotations

import pandas as pd

BASE = ["smart_5", "smart_187", "temperature", "read_errors"]
WINDOWS = (3, 7, 14, 30)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "device_id", *BASE}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["device_id", "date"]).drop_duplicates(["device_id", "date"])


def feature_groups() -> dict[str, str]:
    groups = {column: "current_state" for column in BASE}
    groups["observations"] = "device_metadata"
    for column in BASE:
        for window in WINDOWS:
            for suffix in ("mean", "median", "std", "min", "max", "ema", "delta", "relative", "iqr", "missing_rate"):
                groups[f"{column}_{suffix}{window}"] = "historical_statistics" if suffix in {"mean", "median", "std", "min", "max", "iqr", "missing_rate"} else "temporal_trends"
        groups[f"{column}_trend_acceleration"] = "degradation"
        groups[f"{column}_cumulative_change"] = "degradation"
    return groups


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("device_id", group_keys=False)
    feature_data: dict[str, pd.Series] = {}
    for col in BASE:
        for window in WINDOWS:
            series = grouped[col]
            feature_data[f"{col}_mean{window}"] = series.transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
            feature_data[f"{col}_median{window}"] = series.transform(lambda s, w=window: s.rolling(w, min_periods=1).median())
            feature_data[f"{col}_std{window}"] = series.transform(lambda s, w=window: s.rolling(w, min_periods=2).std())
            feature_data[f"{col}_min{window}"] = series.transform(lambda s, w=window: s.rolling(w, min_periods=1).min())
            feature_data[f"{col}_max{window}"] = series.transform(lambda s, w=window: s.rolling(w, min_periods=1).max())
            feature_data[f"{col}_ema{window}"] = series.transform(lambda s, w=window: s.ewm(span=w, adjust=False, min_periods=1).mean())
            feature_data[f"{col}_delta{window}"] = series.transform(lambda s, w=window: s - s.shift(w))
            feature_data[f"{col}_relative{window}"] = series.transform(lambda s, w=window: (s - s.shift(w)) / s.shift(w).abs().clip(lower=1e-6))
            feature_data[f"{col}_iqr{window}"] = series.transform(lambda s, w=window: s.rolling(w, min_periods=2).quantile(.75) - s.rolling(w, min_periods=2).quantile(.25))
            feature_data[f"{col}_missing_rate{window}"] = series.transform(lambda s, w=window: s.isna().rolling(w, min_periods=1).mean())
        delta7 = feature_data[f"{col}_delta7"]
        feature_data[f"{col}_trend_acceleration"] = delta7.groupby(df["device_id"]).transform(lambda s: s.diff())
        feature_data[f"{col}_cumulative_change"] = grouped[col].transform(lambda s: s - s.iloc[0])
    feature_data["observations"] = grouped.cumcount() + 1
    return pd.DataFrame(feature_data, index=df.index).replace([float("inf"), -float("inf")], pd.NA).fillna(0.0).astype(float)


def make_supervised(frame: pd.DataFrame, horizon: int = 7) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return only at-risk rows with a fully observed forward label.

    The third return value is an eligibility mask aligned to the de-duplicated frame.
    Rows censored before the forward horizon and rows after a registered failure are excluded.
    """
    df = _prepare(frame)
    X = _build_features(df)
    last_dates = df.groupby("device_id")["date"].transform("max")
    failure_flag = df["failed"] if "failed" in df else pd.Series(0, index=df.index)
    failure_dates = df.loc[failure_flag.eq(1)].groupby("device_id")["date"].min()
    failure_for_row = df["device_id"].map(failure_dates)
    known_end = df["date"] + pd.Timedelta(days=horizon) <= last_dates
    at_risk = failure_for_row.isna() | (df["date"] < failure_for_row)
    eligible = known_end & at_risk
    y = (failure_for_row.notna() & (failure_for_row > df["date"]) & (failure_for_row <= df["date"] + pd.Timedelta(days=horizon))).astype(int)
    return X.loc[eligible].reset_index(drop=True), y.loc[eligible].reset_index(drop=True), eligible.reset_index(drop=True)


def make_features(frame: pd.DataFrame, horizon: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Build point-in-time features and forward failure labels per device."""
    df = _prepare(frame)
    X = _build_features(df)
    failure_flag = df["failed"] if "failed" in df.columns else pd.Series(0, index=df.index)
    failure_dates = df.loc[failure_flag.eq(1)].groupby("device_id")["date"].min()
    y = pd.Series(0, index=df.index, dtype="int64")
    for device, failure_date in failure_dates.items():
        y.loc[(df["device_id"] == device) & (df["date"] < failure_date) & (df["date"] >= failure_date - pd.Timedelta(days=horizon))] = 1
    for col in BASE:
        X[f"{col}_roll7_mean"] = X[f"{col}_mean7"]
        X[f"{col}_delta7_legacy"] = X[f"{col}_delta7"]
        X[f"{col}_std7_legacy"] = X[f"{col}_std7"]
        X[f"{col}_missing_rate7_legacy"] = X[f"{col}_missing_rate7"]
    return X, y

