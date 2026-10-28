# Methodology

The prediction target is whether a registered device failure occurs within the next 7 days. Features are calculated using observations available at time `t`; labels look forward only after feature construction. Data is sorted per device, duplicates are removed, and evaluation uses a chronological holdout. The current MVP uses a balanced logistic-regression baseline so the pipeline remains fast and inspectable. Production work should add calibrated tree models, group-aware validation, alert-budget metrics, lead time, and SHAP explanations.

