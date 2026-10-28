# Architecture

`data source -> validation/normalization -> point-in-time features -> chronological training -> artifact -> FastAPI -> dashboard`.

The API is stateless for the MVP. PostgreSQL, MLflow, authentication, scheduled ingestion, model registry, and production monitoring remain deployment extensions.

