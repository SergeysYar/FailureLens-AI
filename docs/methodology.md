# Methodology

The prediction target is whether a registered device failure occurs within the next 7 or 30 days. Features are calculated using observations available at time `t`; labels look forward only after feature construction. Data is sorted per device, duplicates are removed, and evaluation uses a chronological 60/20/20 holdout. The current research baseline compares a prior DummyClassifier with balanced logistic regression. PR-AUC is the primary ranking metric because failures are rare; Brier score and recall are also reported. The baseline's signed standardized coefficients provide local explanations, but they are associations, not causal failure reasons. CatBoost, LightGBM, GRU/TCN, calibration tuning, alert-budget recall, lead time, and optional SHAP are planned extensions.

