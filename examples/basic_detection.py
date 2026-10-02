"""Basic PreShift shift-detection example."""

import numpy as np

from preshift import ShiftDetector


def main():
    rng = np.random.default_rng(42)

    reference = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(1000, 3)
    )

    incoming = rng.normal(
        loc=0.8,
        scale=1.2,
        size=(1000, 3)
    )

    detector = ShiftDetector(reference)

    result = detector.detect(incoming)

    print("PreShift Basic Detection")
    print("------------------------")

    for name, value in result.items():
        print(f"{name}: {value:.6f}")

    print("\nFeature array:")
    print(detector.to_feature_array(result))


if __name__ == "__main__":
    main()