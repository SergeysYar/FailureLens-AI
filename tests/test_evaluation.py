import numpy as np
from failurelens.evaluation import chronological_split


def test_chronological_split_preserves_time_order():
    dates = np.array(["2024-01-03", "2024-01-01", "2024-01-04", "2024-01-02"], dtype="datetime64[D]")
    split = chronological_split(dates, train_ratio=.5, validation_ratio=.25)
    assert max(dates[split.train]) <= min(dates[split.validation])
    assert max(dates[split.validation]) <= min(dates[split.test])
