from preshift.psi import psi_score


def test_identical_distributions():
    result = psi_score(
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]
    )

    assert result == 0.0


def test_shifted_distributions():
    result = psi_score(
        [1, 2, 3, 4, 5],
        [10, 11, 12, 13, 14]
    )

    assert result > 0.0


def test_invalid_bins():
    try:
        psi_score(
            [1, 2, 3],
            [1, 2, 3],
            bins=1
        )
        assert False
    except ValueError:
        assert True