# Getting started

## Local Python path

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # PowerShell
pip install -e ".[dev]"
make demo
make train
uvicorn failurelens.api:app --reload
```

Use `python -m failurelens.cli --help` to inspect commands. API docs are at `/docs`; `/health/ready` reports whether 7- and 30-day artifacts are available.

## Research path

```bash
make data-validate
make features
make research-quick
make research-full
make benchmark
```

## Docker path

```bash
docker compose up --build
docker compose --profile tracking up mlflow
```

The default development Compose file generates a small demo only when `FAILURELENS_BOOTSTRAP_DEMO=1`. Set it to `0` when artifacts are supplied by a deployment process.
