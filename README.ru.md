# FailureLens AI

**Predict failures before they happen.**

FailureLens AI — исследовательская и production-oriented платформа для прогнозирования отказов оборудования по SMART-телеметрии. Проект включает point-in-time feature engineering, chronological ML validation, explainable inference API, model registry fallback, drift monitoring и responsive dashboard.

## Быстрый запуск

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
make demo
make train
uvicorn failurelens.api:app --reload
```

API: `http://127.0.0.1:8000/docs`.

Docker:

```bash
docker compose up --build
```

Dashboard: `http://127.0.0.1:3000`.

## Исследования

```bash
make research-quick
make research-full
make evaluate
make backtest
```

Исследования используют временное разбиение, censoring-aware labels, PR-AUC, bootstrap intervals, feature ablation, backtesting и Kaplan–Meier survival analysis.

## Важно

Демо использует детерминированные синтетические данные. Synthetic metrics не являются оценкой качества на реальных промышленных данных. Подробности: [README.md](README.md), [docs/research.md](docs/research.md), [docs/mlops.md](docs/mlops.md).

