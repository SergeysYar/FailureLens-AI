# Model card — FailureLens baseline

## Model

The serving baseline is a horizon-specific scikit-learn pipeline: `StandardScaler` followed by class-balanced `LogisticRegression`. Artifacts are stored as `baseline_7d.joblib` and `baseline_30d.joblib`.

## Intended use

Prioritizing maintenance review for equipment with SMART telemetry. The output is a risk signal for human decision support, not an autonomous shutdown command or a safety certification.

## Training data

The repository demo uses deterministic synthetic telemetry. Real-data training is designed for normalized Backblaze-style observations but has not been validated here as production performance.

## Evaluation

Training uses censoring-aware eligible rows and chronological validation. PR-AUC is the primary metric because positive events are rare. Metrics generated in this repository are demo-specific and must be regenerated and reviewed for every dataset version.

## Limitations and risks

- source failures may be incomplete or operationally defined;
- missing SMART values may reflect unsupported attributes;
- device rows are temporally correlated;
- probability calibration and alert thresholds require site-specific validation;
- coefficients explain model behavior, not physical causality;
- performance on unseen device models is not established by the default split.

## Monitoring

Monitor artifact readiness, request latency, errors, risk distribution, model version, input schema, PSI drift, and delayed-label quality separately. Drift alone is not proof of model degradation.
