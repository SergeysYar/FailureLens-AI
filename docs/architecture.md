# Architecture

`data source -> validation/normalization -> point-in-time features -> chronological training -> artifact -> FastAPI -> dashboard`.

The API is stateless for the MVP. Model artifacts are horizon-specific (`baseline_7d.joblib` and `baseline_30d.joblib`) and are never silently substituted. PostgreSQL, MLflow, authentication, scheduled ingestion, model registry, and production monitoring remain deployment extensions.

