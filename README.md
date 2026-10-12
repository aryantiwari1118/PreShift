# PreShift

**Prediction-aware dataset shift detection and risk estimation for machine learning monitoring.**

PreShift is a Python library for measuring changes between reference and incoming data, estimating downstream prediction risk from historical observations, and turning risk estimates into monitoring decisions.

Distribution shift does not automatically imply model failure. PreShift provides tools to measure shift and build a risk-estimation workflow; the usefulness of risk estimates must be validated for each application.

## Features

- Multiple distribution-shift metrics: KS, Wasserstein, PSI, and MMD.
- Risk prediction from historical shift metrics and observed downstream errors.
- Threshold-based decision gating.
- A pipeline API for combining detection, risk estimation, and decisions.
- A Python API designed for integration into machine-learning monitoring workflows.

## Installation

```bash
pip install preshift
```

If the package is not yet available on the public PyPI index, install from the repository instead:

```bash
pip install git+https://github.com/aryantiwari1118/PreShift.git
```

## Quick start

### 1. Detect dataset shift

```python
import numpy as np
from preshift import ShiftDetector

rng = np.random.default_rng(42)

reference_data = rng.normal(0.0, 1.0, size=(500, 3))
incoming_data = rng.normal(0.5, 1.2, size=(200, 3))

detector = ShiftDetector(
    reference_data,
    metrics=["ks", "wasserstein", "psi", "mmd"],
)

scores = detector.detect(incoming_data)

for name, value in scores.items():
    print(f"{name}: {value:.4f}")
```

The detector compares incoming observations with the reference data and returns scores for the selected metrics. Score scales and interpretations differ by metric; do not compare their raw magnitudes as if they were interchangeable.

### 2. Estimate downstream prediction risk

Risk prediction requires historical examples pairing shift metrics with observed downstream model errors. Synthetic targets are useful for testing the API, but do not establish real predictive performance.

```python
import numpy as np
from preshift import RiskPredictor

rng = np.random.default_rng(42)

feature_names = [
    "ks_shift_score",
    "wasserstein_shift_score",
    "psi_shift_score",
    "mmd_score",
]

X_history = rng.uniform(0.0, 1.0, size=(200, 4))

# Demonstration targets only.
# In production, use errors measured on historical batches.
y_history = (
    0.4 * X_history[:, 0]
    + 0.3 * X_history[:, 1]
    + 0.2 * X_history[:, 2]
    + 0.1 * X_history[:, 3]
)

predictor = RiskPredictor(feature_names=feature_names)
predictor.fit(X_history, y_history)

X_current = np.array([[0.33, 0.84, 0.57, 0.14]])
risk = predictor.predict(X_current)

print("Estimated risk:", risk)
```

For real use, calculate the same shift features on historical batches and measure the model's error on those batches. Keep the evaluation protocol separate from training to avoid overly optimistic results.

### 3. Apply decision gating

A gate converts a numeric risk estimate into a monitoring decision.

```python

from preshift import PreShiftGate

gate = PreShiftGate(
    accept_threshold=0.30,
    watch_threshold=0.40,
)

print(gate.decide(0.20))  # ACCEPT
print(gate.decide(0.35))  # WATCH
print(gate.decide(0.50))  # DEFER

```

The thresholds above are illustrative. Select thresholds using validation data and the cost of false alarms versus missed degradation. Verify the gate's exact method and return type against the installed version.

### 4. Use the integrated pipeline

PreShift also provides `PreShiftPipeline` to combine shift detection, risk estimation, and gating. The required constructor arguments and output structure depend on the installed API; consult the source and examples before relying on this section in production.

```python

import numpy as np
from preshift import (
    ShiftDetector,
    RiskPredictor,
    PreShiftGate,
    PreShiftPipeline,
)

rng = np.random.default_rng(42)

reference_data = rng.normal(0.0, 1.0, size=(500, 3))

# Demonstration data only; real use requires historical
# shift features paired with measured downstream model errors.
X_history = rng.uniform(0.0, 1.0, size=(200, 4))
y_history = (
    0.4 * X_history[:, 0]
    + 0.3 * X_history[:, 1]
    + 0.2 * X_history[:, 2]
    + 0.1 * X_history[:, 3]
)

detector = ShiftDetector(
    reference_data,
    metrics=["ks", "wasserstein", "psi", "mmd"],
)

predictor = RiskPredictor(
    feature_names=detector.feature_names()
)
predictor.fit(X_history, y_history)

gate = PreShiftGate(
    accept_threshold=0.30,
    watch_threshold=0.40,
)

pipeline = PreShiftPipeline(
    reference_data=reference_data,
    risk_predictor=predictor,
    gate=gate,
    metrics=["ks", "wasserstein", "psi", "mmd"],
)

incoming_data = rng.normal(0.2, 1.1, size=(200, 3))
result = pipeline.predict(incoming_data)

print(result.summary())


# Construct these components using the same feature order
# and thresholds validated for your application.
# See the repository examples for the supported pipeline API.
```

## Important considerations

- A shift score is not a direct measure of model error.
- Risk prediction requires representative historical shift features and observed errors.
- Validate performance on future or otherwise held-out batches.
- Monitor changes in data quality, feature types, seasonal patterns, and target relationships.
- Thresholds should be selected for the application's operational costs.
- Synthetic examples demonstrate usage only; they are not evidence of predictive accuracy.

## Development

Clone the repository:

```bash
git clone https://github.com/aryantiwari1118/PreShift.git
cd PreShift
```

Install development dependencies according to the project configuration, then run the test suite:

```bash
python -m pytest
```

## Contributing

Bug reports, reproducible examples, tests, and documentation improvements are welcome. Please include the Python version, PreShift version, and a minimal example when reporting an issue.

## License

PreShift is distributed under the MIT License. See [LICENSE](LICENSE).
