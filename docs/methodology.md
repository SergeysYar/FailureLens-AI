# FailureLens AI — methodology

This document describes the implemented prediction methodology. It intentionally separates executed behavior from planned extensions.

## 1. Problem definition

For device `d`, observation time `t`, and forecast horizon `H ∈ {7, 30}`, the target is:

```text
y_H(d, t) = 1{ t < T_failure(d) ≤ t + H }
```

where `T_failure(d)` is the first observation date at which the source marks the device as failed. The model estimates the conditional probability of a registered failure during the next `H` days, given that the device is still at risk and only observations through `t` are available. This is a forward-risk classification problem, not a claim that the model knows the physical cause of failure.

## 2. Input contract

The normalized telemetry frame requires:

| Column | Meaning | Rule |
|---|---|---|
| `date` | Observation timestamp | Parsed as a timestamp |
| `device_id` | Stable equipment identifier | Required grouping key |
| `model` | Equipment model label | Metadata; not currently used by numeric baseline |
| `smart_5` | Reallocated-sector-style signal | Non-negative numeric value |
| `smart_187` | Reported/unrecoverable-error-style signal | Non-negative numeric value |
| `temperature` | Device temperature | Numeric; domain range must be checked for real data |
| `read_errors` | Read-error count | Non-negative numeric value |
| `failed` | Registered failure indicator | Optional for inference, required for supervised training |

The demo generator creates these fields deterministically. It is synthetic and must never be mixed into claims about Backblaze or production performance.

## 3. Data preparation

`_prepare()` verifies required columns, parses `date`, sorts by `device_id` and `date`, and removes duplicate `(device_id, date)` pairs. A real ingestion pipeline should additionally emit a duplicate count and retain a source manifest so discarded rows can be audited.

## 4. Target construction and censoring

`make_supervised()` differs from the inference helper `make_features()` because training must exclude labels that cannot be known yet.

For each device:

- `last_date` is the last available observation;
- a row is right-censored when `date + H > last_date` and there is no complete forward observation window;
- a row is not at risk when it occurs on or after the first registered failure;
- an eligible row is at risk and has a complete forward window;
- `y=1` only when the first registered failure is inside `(date, date + H]`;
- an eligible no-failure row receives `y=0` only when the full horizon is observed.

This prevents treating a device that disappeared before the horizon as a confirmed negative. Disappearance without an explicit failure mark remains censored, not “healthy”. The function returns `(X, y, eligible_mask)`. Training and research use the eligible subset; the API uses the non-filtering feature builder for current-time inference.

## 5. Point-in-time feature engineering

Every feature is computed within a device history and uses the current row plus prior rows only. No future row is joined into the feature frame.

### Current state

Raw values: `smart_5`, `smart_187`, `temperature`, and `read_errors`.

### Historical statistics

For each base signal and each window in `{3, 7, 14, 30}`: rolling mean, median, standard deviation, minimum, maximum, interquartile range, and missing-observation rate. Short histories use the available prefix. Undefined one-observation dispersion values are imputed to `0.0` after computation.

### Temporal trends

The pipeline computes an exponential moving average, first difference, and relative change for each window:

```text
(x_t - x_{t-w}) / max(|x_{t-w}|, 1e-6)
```

The denominator guard prevents division by zero. The value is a change indicator, not a calibrated percentage near zero.

### Degradation indicators

The current degradation group contains 7-step trend acceleration (the first difference of the 7-step difference) and cumulative change from the first available device observation. These are hypotheses, not causal diagnoses, and are evaluated through ablation.

### Missing values

Non-finite and missing feature values are filled with `0.0` after temporal computation. For real data, missingness semantics must be reviewed per SMART attribute: “not reported” can mean unsupported, not healthy. The serving schema is labeled `temporal-v3`.

## 6. Validation protocol

The default research split is chronological 60/20/20: earliest observations for train, the next block for validation, and the latest block for test. It is never a random row split. Thresholds and model choices belong on validation; test is reserved for the final estimate.

The implementation uses a global chronological ordering of eligible observations. A production site should additionally run a device-group holdout to measure generalization to unseen equipment.

## 7. Models

Implemented: prior `DummyClassifier`, balanced Logistic Regression with standardized features, Random Forest, and CatBoost when the optional package is installed. Serving artifacts are horizon-specific: `baseline_7d.joblib` and `baseline_30d.joblib`; one horizon is never silently substituted for another.

LightGBM, GRU/TCN, Cox survival, Optuna, MLflow, SHAP, and plotting integrations are detected at runtime. Reports mark unavailable methods as `not_installed / not_executed`.

## 8. Metrics

Failures are rare, so PR-AUC is the primary ranking metric. Reports also include ROC-AUC, precision, recall, Brier score, recall at top 1/5/10% alert budgets, bootstrap intervals for test PR-AUC, and operational backtest counts. Kaplan–Meier survival output is separate and is not directly comparable to classification PR-AUC.

## 9. Explainability

The baseline is a `StandardScaler` + Logistic Regression pipeline. Local explanations are signed standardized coefficient contributions:

```text
contribution_j = coefficient_j × standardized_feature_j
```

Positive values increase the model score; negative values decrease it. They are model associations, not causal explanations or maintenance instructions.

## 10. Reproducibility

Research reports record seed `7` where applicable, dataset fingerprint, git commit, feature schema, split protocol, optional dependency status, and generated JSON/Markdown paths.

```bash
make demo
make data-validate
make research-full
```

## 11. Limitations

The demo is synthetic and small; Backblaze-style data is observational; device rows are correlated; calibration is measured but not yet tuned; alert economics require site-specific policy; and the local JSON registry is not a managed multi-user registry.
