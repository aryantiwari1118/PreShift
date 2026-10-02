import pytest

from preshift.calibration import calibrate_thresholds


def test_calibrate_thresholds():
    accept, watch = calibrate_thresholds(
        [0.20, 0.30, 0.35, 0.40, 0.50, 0.60]
    )

    assert accept == 0.375
    assert watch == 0.475


def test_empty_risk_is_rejected():
    with pytest.raises(ValueError):
        calibrate_thresholds([])


def test_invalid_accept_quantile():
    with pytest.raises(ValueError):
        calibrate_thresholds(
            [0.1, 0.2, 0.3],
            accept_quantile=-0.1,
        )


def test_invalid_watch_quantile():
    with pytest.raises(ValueError):
        calibrate_thresholds(
            [0.1, 0.2, 0.3],
            watch_quantile=1.1,
        )


def test_invalid_quantile_order():
    with pytest.raises(ValueError):
        calibrate_thresholds(
            [0.1, 0.2, 0.3],
            accept_quantile=0.80,
            watch_quantile=0.70,
        )


def test_non_finite_risk_is_rejected():
    with pytest.raises(ValueError):
        calibrate_thresholds(
            [0.1, float("nan"), 0.3]
        )