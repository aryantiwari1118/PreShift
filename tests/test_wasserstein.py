from preshift.wasserstein import wasserstein_distance_score


def test_identical_distributions():
    result = wasserstein_distance_score(
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]
    )

    assert result == 0.0


def test_shifted_distributions():
    result = wasserstein_distance_score(
        [1, 2, 3, 4, 5],
        [10, 11, 12, 13, 14]
    )

    assert result == 9.0


def test_nan_values_are_ignored():
    result = wasserstein_distance_score(
        [1, 2, 3, float("nan"), 5],
        [1, 2, 3, 4, 5]
    )

    assert result >= 0.0