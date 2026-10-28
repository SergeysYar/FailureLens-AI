# Troubleshooting

## `503 Model is not trained`

Run `make train` or `make retrain`. Check `/health/ready` and `/api/v1/model/status` for the missing horizon artifact.

## Dashboard says API offline

Start `uvicorn failurelens.api:app --reload`, verify port `8000`, and open the dashboard through a local web server rather than a restrictive `file://` context if browser CORS blocks requests.

## Empty Model Lab

Run `make research-quick` or `make research-full`. Empty state is intentional when no persisted report exists.

## Drift report has no features

Run `make data-validate` first and ensure the reference/current frames contain the normalized numeric columns.

## Docker unavailable locally

Install Docker Desktop or use the Python quick start. Do not claim a Compose build passed unless it has actually run in the target environment.
