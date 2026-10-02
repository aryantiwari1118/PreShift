"""End-to-end PreShift pipeline example."""

import numpy as np

from preshift import (
    PreShiftGate,
    PreShiftPipeline,
    RiskPredictor,
)


def main():
    rng = np.random.default_rng(42)

    reference = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(1000, 3)
    )

    training_shift_features = rng.uniform(
        0.0,
        1.0,
        size=(200, 4)
    )

    training_risk = (
        0.35 * training_shift_features[:, 0]
        + 0.30 * training_shift_features[:, 1]
        + 0.25 * training_shift_features[:, 2]
        + 0.10 * training_shift_features[:, 3]
    )

    feature_names = [
        "ks_shift_score",
        "wasserstein_shift_score",
        "psi_shift_score",
        "mmd_score",
    ]

    risk_predictor = RiskPredictor(
        random_state=42,
        feature_names=feature_names
    )

    risk_predictor.fit(
        training_shift_features,
        training_risk
    )

    gate = PreShiftGate(
        accept_threshold=0.30,
        watch_threshold=0.50
    )

    pipeline = PreShiftPipeline(
        reference_data=reference,
        risk_predictor=risk_predictor,
        gate=gate
    )

    incoming = rng.normal(
        loc=0.8,
        scale=1.2,
        size=(1000, 3)
    )

    result = pipeline.predict(incoming)

    print(result.summary())


if __name__ == "__main__":
    main()