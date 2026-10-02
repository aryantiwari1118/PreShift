"""Smoke test for the installed PreShift package."""

import numpy as np

from preshift import (
    PreShiftGate,
    PreShiftPipeline,
    RiskPredictor,
)


def main():
    rng = np.random.default_rng(42)

    reference = rng.normal(
        0.0,
        1.0,
        size=(500, 3)
    )

    X_train = rng.uniform(
        0.0,
        1.0,
        size=(100, 4)
    )

    y_train = (
        0.4 * X_train[:, 0]
        + 0.3 * X_train[:, 1]
        + 0.2 * X_train[:, 2]
        + 0.1 * X_train[:, 3]
    )

    feature_names = [
        "ks_shift_score",
        "wasserstein_shift_score",
        "psi_shift_score",
        "mmd_score",
    ]

    predictor = RiskPredictor(
        random_state=42,
        feature_names=feature_names
    )

    predictor.fit(X_train, y_train)

    gate = PreShiftGate(
        accept_threshold=0.30,
        watch_threshold=0.50
    )

    pipeline = PreShiftPipeline(
        reference_data=reference,
        risk_predictor=predictor,
        gate=gate
    )

    incoming = rng.normal(
        0.8,
        1.2,
        size=(500, 3)
    )

    result = pipeline.predict(incoming)

    assert len(result.shift_scores) == 4
    assert np.isfinite(result.predicted_risk)
    assert result.decision in {"ACCEPT", "WATCH", "DEFER"}

    print("PreShift wheel smoke test")
    print("-------------------------")
    print("Status: PASS")
    print(f"Predicted risk: {result.predicted_risk:.6f}")
    print(f"Decision: {result.decision}")


if __name__ == "__main__":
    main()