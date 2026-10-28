# Production MLOps

## Lifecycle

`data-validate -> features -> train/retrain -> evaluate -> register-model -> candidate -> explicit promote-model -> serving -> monitoring -> drift review -> retrain`.

The local registry is a deterministic fallback for environments without MLflow. It stores model version, artifact path, horizon, feature schema, training metrics, and lifecycle state in `artifacts/registry.json`. Promotion is never automatic: `make promote-model VERSION=<version>` is an explicit operation and archives the previous production version. `make rollback VERSION=<version>` archives the selected version; deployment rollback should restore the last production artifact in the target environment.

## Serving and readiness

The API loads and validates horizon-specific joblib artifacts through `ModelStore`, caches them by modification time, and exposes `/health/live`, `/health/ready`, `/api/v1/model/status`, `/api/v1/model/versions`, `/api/v1/model/metadata`, `/api/v1/predict`, and `/api/v1/predict/batch`. A missing model returns `503`; it never produces a fabricated prediction.

## Monitoring

`/metrics` exposes request/error counters, p95 latency, risk distribution, and model-version usage. `/api/v1/monitoring/status` returns the structured snapshot. `make monitor-drift` computes configurable PSI for the supported telemetry columns. Drift is an input-distribution signal, not proof of model-quality degradation; quality needs delayed labels.

## Deployment

`docker compose up --build` starts API, static web, and PostgreSQL. MLflow is an explicit optional profile: `docker compose --profile tracking up mlflow`. The development compose file bootstraps the small deterministic demo only when `FAILURELENS_BOOTSTRAP_DEMO=1`; production should set it to `0` and mount approved artifacts.

