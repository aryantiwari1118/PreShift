import numpy as np
import pytest

from preshift.risk import RiskPredictor


def test_risk_predictor_training_and_prediction():
    X = [
        [0.0, 0.0, 0.0, 0.0],
        [0.2, 0.3, 0.4, 0.1],
        [0.5, 0.6, 0.7, 0.4],
        [0.8, 0.9, 1.0, 0.7],
    ]

    y = [
        0.10,
        0.20,
        0.40,
        0.60
    ]

    model = RiskPredictor(
        n_estimators=50,
        random_state=42
    )

    model.fit(X, y)

    prediction = model.predict(
        [[0.5, 0.6, 0.7, 0.4]]
    )

    assert isinstance(
        prediction,
        np.ndarray
    )

    assert prediction.shape == (1,)

    assert np.isfinite(
        prediction[0]
    )


def test_prediction_before_training():
    model = RiskPredictor()

    with pytest.raises(RuntimeError):
        model.predict(
            [[0.1, 0.2, 0.3, 0.4]]
        )


def test_mismatched_training_data():
    model = RiskPredictor()

    with pytest.raises(ValueError):
        model.fit(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5]
            ],
            [0.1]
        )
def test_nan_shift_features_are_rejected():
    model = RiskPredictor()

    with pytest.raises(ValueError):
        model.fit(
            [
                [0.1, 0.2],
                [float("nan"), 0.4],
            ],
            [0.1, 0.2],
        )


def test_nan_observed_risk_is_rejected():
    model = RiskPredictor()

    with pytest.raises(ValueError):
        model.fit(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            [0.1, float("nan")],
        )

def test_feature_names_are_stored():
    feature_names = [
        "ks_shift_score",
        "psi_shift_score",
    ]

    model = RiskPredictor(
        feature_names=feature_names
    )

    assert model.feature_names() == feature_names


def test_feature_name_dimension_mismatch():
    model = RiskPredictor(
        feature_names=[
            "ks_shift_score",
        ]
    )

    with pytest.raises(ValueError):
        model.fit(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            [0.1, 0.2],
        )
def test_prediction_feature_dimension_mismatch():
    model = RiskPredictor(
        feature_names=[
            "ks_shift_score",
            "psi_shift_score",
        ]
    )

    model.fit(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        [0.1, 0.2, 0.3],
    )

    with pytest.raises(ValueError):
        model.predict(
            [[0.1]]
        )