# ML research execution guide

The formal methodology is in [methodology.md](methodology.md). This file explains how to run and interpret the experimental pipeline.

## Quick versus full

```bash
make research-quick  # Dummy + Logistic Regression; fast smoke experiment
make research-full   # adds Random Forest and CatBoost when installed
```

The commands read `data/demo/telemetry.csv`, generating it only if absent. They write JSON/Markdown into `artifacts/research/`; these outputs are ignored by Git and must be regenerated in a clean checkout.

## Report contents

Each report contains status, horizon, mode, protocol, dataset fingerprint, eligible row count after censoring, validation/test metrics, bootstrap 95% PR-AUC interval, optional dependency status, alert-budget recall, feature-group ablation, operational backtest, Kaplan–Meier survival output, seed, git commit, and feature metadata.

## Interpreting results

Use test PR-AUC only as an estimate for the exact demo generation process and split. Compare it with the dummy prevalence baseline. A higher ROC-AUC alone is not enough for an alerting system: inspect PR-AUC, Brier score, recall at the available alert budget, and lead-time behavior.

Do not compare a classification probability directly with a Kaplan–Meier survival probability. They answer related but different questions.

## Honest status policy

If a dependency is unavailable, the report says `not_installed / not_executed`. If a report is absent, the dashboard shows an empty state. No random charts, placeholder metrics, or claims of completed deep-learning/survival experiments are generated.
