# FailureLens AI

Predictive equipment-failure intelligence for SMART telemetry. The repository contains a reproducible demo mode, a leakage-safe feature pipeline, a trainable baseline model, and a FastAPI service.

> **Demo / synthetic data:** the local demo is generated deterministically and is clearly separated from the official Backblaze Drive Stats dataset. No performance claims are made until real data has been downloaded and evaluated.

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
pip install -e ".[dev]"
python -m failurelens.cli demo
python -m failurelens.cli train
python -m failurelens.cli experiment --horizon 7
uvicorn failurelens.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API. A small static dashboard is available from `web/index.html`.

## Commands

```bash
make demo       # generate deterministic synthetic telemetry
make train      # train logistic regression on generated telemetry
make experiment # run the chronological baseline comparison
make test       # run unit tests
docker compose up --build
```

The real dataset is downloaded only by an explicit command and is not required for CI or demo mode. See [docs/dataset.md](docs/dataset.md) and [docs/methodology.md](docs/methodology.md).

Phase 3 research adds censoring-aware labels, advanced temporal features, model comparison, ablation, bootstrap intervals, backtesting, and survival analysis. See [docs/research.md](docs/research.md).

## Production MLOps

The production path includes horizon-specific model artifacts, explicit candidate/staging/production registry states, rollback, readiness/liveness checks, batch inference, request metrics, PSI drift reports, structured logs, CI smoke checks, and a non-root Docker image. See [docs/mlops.md](docs/mlops.md), [docs/security.md](docs/security.md), and [docs/performance.md](docs/performance.md).

```bash
make data-validate
make train
make register-model
make promote-model VERSION=<version>
make monitor-drift
docker compose up --build
```

## API

- `GET /health`
- `GET /api/v1/summary`
- `GET /api/v1/devices`
- `GET /api/v1/devices/{device_id}`
- `POST /api/v1/predict`
- `GET /api/v1/model`
- `GET /api/v1/experiments`
- `GET /api/v1/research`

## Limitations

The demo is not a safety-critical maintenance system. The Backblaze data has observational limitations, failure labels are sparse, and production use requires site-specific validation, calibration, alert-cost policy, monitoring, and human review.

The checked-in experiment result is generated from synthetic data and is illustrative only; it must not be presented as real-world model performance.

## Data source

Backblaze Drive Stats: <https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data>. Use subject to the dataset's published terms and attribution requirements.

## License

MIT. See [LICENSE](LICENSE).

