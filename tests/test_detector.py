import pytest

from preshift.detector import ShiftDetector


def test_no_shift():
    detector = ShiftDetector(
        [[1], [2], [3], [4], [5]]
    )

    result = detector.detect(
        [[1], [2], [3], [4], [5]]
    )

    assert result["ks_shift_score"] == 0.0
    assert result["wasserstein_shift_score"] == 0.0
    assert result["psi_shift_score"] == 0.0
    assert result["mmd_score"] == 0.0


def test_detect_shift():
    detector = ShiftDetector(
        [[1], [2], [3], [4], [5]]
    )

    result = detector.detect(
        [[10], [11], [12], [13], [14]]
    )

    assert result["ks_shift_score"] == 1.0
    assert result["wasserstein_shift_score"] == 9.0
    assert result["psi_shift_score"] > 0.0
    assert result["mmd_score"] > 0.0


def test_feature_dimension_mismatch():
    detector = ShiftDetector(
        [[1, 10], [2, 20], [3, 30]]
    )

    with pytest.raises(ValueError):
        detector.detect(
            [[1], [2], [3]]
        )