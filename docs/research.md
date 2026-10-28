# ML Research Protocol

Phase 3 experiments are explicit, reproducible, and honest about unavailable methods.

## Executed by default

- point-in-time temporal features over 3/7/14/30 observations;
- censored-row exclusion for forward labels;
- DummyClassifier, Logistic Regression, Random Forest, and CatBoost when installed;
- chronological 60/20/20 validation;
- PR-AUC, ROC-AUC, Brier score, precision, recall, bootstrap PR-AUC intervals;
- feature-group ablation;
- top-1/5/10% alert-budget recall;
- operational backtest;
- dependency-free Kaplan–Meier survival curve with right censoring;
- JSON and Markdown research reports with dataset fingerprint and git commit.

## Explicitly optional

LightGBM, GRU/TCN, Cox survival, Optuna, MLflow, SHAP, and plotting packages are detected at runtime. If unavailable, the report marks them `not_installed / not_executed`; no placeholder metrics are generated.

Run `make research-quick` for a lightweight run or `make research-full` for the extended classical comparison. Reports are written under `artifacts/research/` and are ignored by Git.
