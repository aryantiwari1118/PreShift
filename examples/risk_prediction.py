"""Basic PreShift risk-prediction example."""

import numpy as np

from preshift import RiskPredictor


def main():
    rng = np.random.default_rng(42)

    X = rng.uniform(0.0, 1.0, size=(100, 4))
    y = (
        0.4 * X[:, 0]
        + 0.3 * X[:, 1]
        + 0.2 * X[:, 2]
        + 0.1 * X[:, 3]
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

    predictor.fit(X, y)

    new_shift = np.array([
        [0.33, 0.84, 0.57, 0.14]
    ])

    predicted_risk = predictor.predict(new_shift)

    print("PreShift Risk Prediction")
    print("------------------------")
    print(f"Predicted risk: {predicted_risk[0]:.6f}")
    print(f"Feature names: {predictor.feature_names()}")


if __name__ == "__main__":
    main()