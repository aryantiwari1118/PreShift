import numpy as np

from preshift.ks import ks_statistic


def test_identical_distributions():
    result = ks_statistic(
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]
    )

    assert result == 0.0


def test_completely_separated_distributions():
    result = ks_statistic(
        [1, 2, 3, 4, 5],
        [10, 11, 12, 13, 14]
    )

    assert result == 1.0


def test_nan_values_are_ignored():
    result = ks_statistic(
        [1, 2, 3, np.nan, 5],
        [1, 2, 3, 4, 5]
    )

    assert 0.0 <= result <= 1.0