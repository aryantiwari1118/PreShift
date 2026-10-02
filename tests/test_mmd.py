import numpy as np

from preshift.mmd import mmd_score


def test_identical_distributions():
    result = mmd_score(
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]
    )

    assert result == 0.0


def test_shifted_distributions():
    result = mmd_score(
        [1, 2, 3, 4, 5],
        [10, 11, 12, 13, 14]
    )

    assert result > 0.0


def test_multivariate_data():
    reference = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
        [5, 50]
    ])

    incoming = np.array([
        [10, 100],
        [11, 110],
        [12, 120],
        [13, 130],
        [14, 140]
    ])

    result = mmd_score(
        reference,
        incoming
    )

    assert result > 0.0