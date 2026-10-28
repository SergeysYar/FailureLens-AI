# Security

- No credentials are committed; use `.env.example` as a template only.
- CORS is allowlisted through `FAILURELENS_ALLOWED_ORIGINS`.
- Pydantic validates prediction payloads and batch size (maximum 1000 records).
- No administrative registry mutation is exposed through the public API; promotion and rollback are CLI/deployment operations.
- Logs contain request metadata and timing, not telemetry payloads or secrets.
- Docker runs the API as a non-root user.
- Dependency and secret checks run in CI where the runner supports them.

