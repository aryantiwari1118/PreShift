import numpy as np
import pytest
from preshift.result import PreShiftResult
from preshift import (
    PreShiftGate,
    PreShiftPipeline,
    RiskPredictor,
)


def create_pipeline():
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
        0.60,
    ]

    risk_model = RiskPredictor(
        n_estimators=50,
        random_state=42,
    )

    risk_model.fit(X, y)

    gate = PreShiftGate(
        accept_threshold=0.30,
        watch_threshold=0.50,
    )

    return PreShiftPipeline(
        reference_data=[
            [1],
            [2],
            [3],
            [4],
            [5],
        ],
        risk_predictor=risk_model,
        gate=gate,
    )


def test_pipeline_output_structure():
    pipeline = create_pipeline()

    result = pipeline.predict(
        [
            [10],
            [11],
            [12],
            [13],
            [14],
        ]
    )

    assert isinstance(
        result,
        PreShiftResult
    )

    assert result.shift_scores is not None
    assert result.predicted_risk is not None
    assert result.decision is not None


def test_pipeline_shift_scores():
    pipeline = create_pipeline()

    result = pipeline.predict(
        [
            [10],
            [11],
            [12],
            [13],
            [14],
        ]
    )

    scores = result.shift_scores

    assert scores["ks_shift_score"] == 1.0
    assert scores["wasserstein_shift_score"] == 9.0
    assert scores["psi_shift_score"] > 0.0
    assert scores["mmd_score"] > 0.0


def test_pipeline_risk_is_valid():
    pipeline = create_pipeline()

    result = pipeline.predict(
        [
            [10],
            [11],
            [12],
            [13],
            [14],
        ]
    )

    assert isinstance(
        result.predicted_risk,
        float,
    )

    assert np.isfinite(
        result.predicted_risk
    )


def test_pipeline_decision_is_valid():
    pipeline = create_pipeline()

    result = pipeline.predict(
        [
            [10],
            [11],
            [12],
            [13],
            [14],
        ]
    )

    assert result.decision in {
        "ACCEPT",
        "WATCH",
        "DEFER",
    }
def test_pipeline_with_selected_metrics():
    X = [
        [0.0, 0.0],
        [0.2, 0.3],
        [0.5, 0.6],
        [0.8, 0.9],
    ]

    y = [
        0.10,
        0.20,
        0.40,
        0.60,
    ]

    risk_model = RiskPredictor(
        n_estimators=50,
        random_state=42,
    )

    risk_model.fit(X, y)

    gate = PreShiftGate(
        accept_threshold=0.30,
        watch_threshold=0.50,
    )

    pipeline = PreShiftPipeline(
        reference_data=[
            [1],
            [2],
            [3],
            [4],
            [5],
        ],
        risk_predictor=risk_model,
        gate=gate,
        metrics=["ks", "psi"],
    )

    result = pipeline.predict(
        [
            [10],
            [11],
            [12],
            [13],
            [14],
        ]
    )

    assert set(result.shift_scores.keys()) == {
        "ks_shift_score",
        "psi_shift_score",
    }

    assert result.predicted_risk is not None
    assert result.decision is not None


def test_pipeline_selected_metric_order():
    detector_metrics = [
        "psi",
        "ks",
    ]

    pipeline = PreShiftPipeline(
        reference_data=[
            [1],
            [2],
            [3],
            [4],
            [5],
        ],
        risk_predictor=RiskPredictor(),
        gate=PreShiftGate(
            accept_threshold=0.30,
            watch_threshold=0.50,
        ),
        metrics=detector_metrics,
    )

    assert pipeline.detector.feature_names() == [
        "psi_shift_score",
        "ks_shift_score",
    ]
def test_pipeline_rejects_mismatched_risk_features():
    risk_model = RiskPredictor(
        n_estimators=50,
        random_state=42,
        feature_names=[
            "ks_shift_score",
            "wasserstein_shift_score",
        ],
    )

    risk_model.fit(
        [
            [0.0, 0.0],
            [0.2, 0.3],
            [0.5, 0.6],
            [0.8, 0.9],
        ],
        [0.10, 0.20, 0.40, 0.60],
    )

    gate = PreShiftGate(
        accept_threshold=0.30,
        watch_threshold=0.50,
    )

    try:
        PreShiftPipeline(
            reference_data=[
                [1],
                [2],
                [3],
                [4],
                [5],
            ],
            risk_predictor=risk_model,
            gate=gate,
            metrics=["ks", "psi"],
        )
        assert False
    except ValueError:
        assert True
def test_pipeline_accepts_matching_risk_features():
    feature_names = [
        "ks_shift_score",
        "psi_shift_score",
    ]

    risk_model = RiskPredictor(
        n_estimators=50,
        random_state=42,
        feature_names=feature_names,
    )

    risk_model.fit(
        [
            [0.0, 0.0],
            [0.2, 0.3],
            [0.5, 0.6],
            [0.8, 0.9],
        ],
        [0.10, 0.20, 0.40, 0.60],
    )

    pipeline = PreShiftPipeline(
        reference_data=[
            [1],
            [2],
            [3],
            [4],
            [5],
        ],
        risk_predictor=risk_model,
        gate=PreShiftGate(
            accept_threshold=0.30,
            watch_threshold=0.50,
        ),
        metrics=["ks", "psi"],
    )

    assert pipeline.detector.feature_names() == feature_names


def test_pipeline_requires_fitted_risk_predictor():
    pipeline = PreShiftPipeline(
        reference_data=[
            [1],
            [2],
            [3],
            [4],
            [5],
        ],
        risk_predictor=RiskPredictor(),
        gate=PreShiftGate(
            accept_threshold=0.30,
            watch_threshold=0.50,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="must be fitted",
    ):
        pipeline.predict(
            [
                [10],
                [11],
                [12],
                [13],
                [14],
            ]
        )


def test_pipeline_rejects_incoming_feature_dimension_mismatch():
    pipeline = create_pipeline()

    with pytest.raises(
        ValueError,
        match="same number of features",
    ):
        pipeline.predict(
            [
                [10, 20],
                [11, 21],
                [12, 22],
                [13, 23],
                [14, 24],
            ]
        )