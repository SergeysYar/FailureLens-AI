# Deployment

## Compose services

- `api`: FastAPI inference service;
- `web`: Nginx serving the static dashboard;
- `postgres`: provisioned persistence service for future application state;
- `mlflow`: optional `tracking` profile.

```bash
docker compose up --build
docker compose --profile tracking up mlflow
```

The API image runs as a non-root user and has a liveness healthcheck. Compose readiness also waits for PostgreSQL, although the current MVP does not yet use PostgreSQL for inference records.

## Production artifact policy

Build or mount approved horizon-specific artifacts before enabling production traffic. Set `FAILURELENS_BOOTSTRAP_DEMO=0`, configure allowed origins, and provide secrets through the deployment environment. Never use the demo bootstrap for real production data.

## Promotion

```bash
make register-model
make promote-model VERSION=<candidate-version>
make rollback VERSION=<previous-version>
```

Promotion is explicit and archives the previous production record for the selected horizon.
