import pandas as pd
from failurelens.features import make_supervised
from failurelens.survival import kaplan_meier


def test_censored_rows_are_excluded_from_supervised_labels():
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "device_id": ["a"] * 3,
                          "smart_5": [1, 2, 3], "smart_187": [1, 2, 3], "temperature": [30, 30, 30],
                          "read_errors": [0, 0, 0], "failed": [0, 0, 0]})
    X, y, eligible = make_supervised(frame, horizon=2)
    assert len(X) == 1
    assert len(y) == 1 and y.iloc[0] == 0
    assert eligible.tolist() == [True, False, False]


def test_survival_marks_failures_and_censoring():
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3).tolist() * 2,
                          "device_id": ["a"] * 3 + ["b"] * 3,
                          "smart_5": [1] * 6, "smart_187": [1] * 6, "temperature": [30] * 6,
                          "read_errors": [0] * 6, "failed": [0, 0, 1, 0, 0, 0]})
    result = kaplan_meier(frame)
    assert result["events"] == 1
    assert result["right_censored"] == 1
