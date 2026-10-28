import pandas as pd
from failurelens.monitoring import drift_report
from failurelens.registry import register, transition


def test_registry_requires_explicit_promotion(tmp_path):
    path = tmp_path / "registry.json"
    record = register(path, "model.joblib", {"horizon_days": 7}, version="v1")
    assert record["state"] == "candidate"
    promoted = transition(path, "v1", "production")
    assert promoted["state"] == "production"


def test_drift_report_detects_shift():
    reference = pd.DataFrame({"value": [0, 0, 0, 1, 1]})
    current = pd.DataFrame({"value": [10, 10, 10, 11, 11]})
    result = drift_report(reference, current, ["value"], threshold=0.01)
    assert result["features"]["value"]["status"] == "drift"
