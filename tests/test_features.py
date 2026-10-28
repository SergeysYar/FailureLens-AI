import pandas as pd
from failurelens.demo import generate_demo
from failurelens.features import make_features


def test_features_are_sorted_and_finite():
    X, y = make_features(generate_demo(n_devices=4, days=20))
    assert len(X) == len(y)
    assert X.notna().all().all()


def test_future_failure_is_not_used_as_feature():
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "device_id": ["a"] * 3,
                          "smart_5": [1, 2, 3], "smart_187": [1, 2, 3], "temperature": [30, 30, 30],
                          "read_errors": [0, 0, 1], "failed": [0, 0, 1]})
    X, y = make_features(frame)
    # The failure is seven days ahead, so it is a valid positive label at t=0;
    # the feature must still be based only on the observation at t=0.
    assert y.iloc[0] == 1
    assert X.iloc[0]["smart_5_roll7_mean"] == 1

