# Explainability

The API returns the highest absolute signed contributions for the Logistic Regression baseline. Each contribution is calculated in standardized model space:

```text
feature contribution = standardized input value × fitted coefficient
```

`increases_risk` means the contribution moves the linear score upward; `reduces_risk` means it moves it downward. The value is local to the request and model version.

## Correct interpretation

An explanation answers “which inputs moved this model score?” It does not answer “which component physically caused the failure?” Correlation, proxy behavior, sensor quality, and missingness can all influence a contribution.

## Product behavior

The API includes `model_version` and `feature_schema_version` with the prediction. If an artifact is missing, the API returns `503` rather than fabricating an explanation.
