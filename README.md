# PreShift

**Prediction-aware dataset shift detection and risk estimation for machine learning.**

PreShift is a Python library for measuring distributional changes between reference and incoming data, estimating potential downstream prediction risk, and translating risk estimates into monitoring decisions.

Traditional drift monitoring asks:

> Has the incoming data distribution changed?

PreShift adds another question:

> Given historical evidence, what downstream prediction risk might this change indicate?

PreShift combines statistical shift metrics, a supervised risk predictor, and an optional decision gate in a single workflow.

**Status:** Alpha — research-oriented and under active development.

## Features

- **Multiple shift metrics:** Kolmogorov–Smirnov (KS), Wasserstein distance, Population Stability Index (PSI), and Maximum Mean Discrepancy (MMD).
- **Configurable detection:** Select one or more supported metrics.
- **Prediction-risk estimation:** Train a scikit-learn-based model using historical shift features and observed downstream prediction errors.
- **Decision gating:** Map estimated risk to `ACCEPT`, `WATCH`, or `DEFER`.
- **Threshold calibration:** Derive decision thresholds from historical risk observations.
- **Integrated pipeline:** Combine shift detection, risk prediction, and decision-making.
- **Structured results:** Access shift scores, estimated risk, and the final decision programmatically.

## How it works

```text
Reference data + Incoming data
              |
              v
       Shift detection
              |
              v
   KS / Wasserstein / PSI / MMD
              |
              v
     Historical risk model
              |
              v
      Estimated risk
              |
              v
       Decision gate
              |
      +-------+-------+
      |       |       |
    ACCEPT  WATCH   DEFER
```

The risk-estimation stage requires a predictor trained on historical examples linking shift metrics to observed downstream model errors. PreShift does not assume that every distributional change causes a decline in model performance.

## Installation

### Install from GitHub

Until the first PyPI release is available, install the current repository version:

```bash
pip install git+https://github.com/aryantiwari1118/PreShift.git
```

### Install from source

```bash
git clone https://github.com/aryantiwari1118/PreShift.git
cd PreShift
python -m pip install .
```

### Install from PyPI

After the package is published, install it with:

```bash
pip install preshift
```

### Requirements

- Python 3.10+
- NumPy
- pandas
- SciPy
- scikit-learn

## Quick start: detect dataset shift

Create a reference dataset and an incoming dataset, then compare them using the supported metrics.

```python
import numpy as np
from preshift import ShiftDetector

rng = np.random.default_rng(42)

reference_data = rng.normal(
    loc=0.0,
    scale=1.0,
    size=(500, 3),
)

incoming_data = rng.normal(
    loc=0.5,
    scale=1.2,
    size=(200, 3),
)

detector = ShiftDetector(
    reference_data,
    metrics=["ks", "wasserstein", "psi", "mmd"],
)

shift_scores = detector.detect(incoming_data)

for name, score in shift_scores.items():
    print(f"{name}: {score:.4f}")
```

The result is a dictionary containing the selected metrics. For example:

```python
{
    "ks_shift_score": 0.20,
    "wasserstein_shift_score": 0.36,
    "psi_shift_score": 0.21,
    "mmd_score": 0.04,
}
```

The numbers above are illustrative; actual values depend on the data and metric implementations.

### Supported metrics

| Metric | Purpose |
|---|---|
| KS | Compares empirical distributions using the maximum difference between their cumulative distributions. |
| Wasserstein | Measures distributional distance using optimal transport. |
| PSI | Quantifies changes in binned population distributions. |
| MMD | Measures distributional discrepancy using a kernel-based statistic. |

KS, Wasserstein, and PSI are aggregated across features by the current detector implementation. MMD is calculated on the multivariate datasets.

**Important:** Scores use different definitions and scales. Do not compare them as if they shared a universal threshold. They are shift indicators, not direct measurements of downstream model error.

## Estimate downstream prediction risk

`RiskPredictor` learns a relationship between historical shift features and observed prediction risk.

Historical training data should contain:

- `X_history`: one row of shift metrics per historical batch.
- `y_history`: the corresponding observed prediction error for the downstream model.

For example, `y_history` could contain RMSE or NRMSE values measured after ground-truth labels become available.

```python
import numpy as np
from preshift import RiskPredictor

feature_names = [
    "ks_shift_score",
    "wasserstein_shift_score",
    "psi_shift_score",
    "mmd_score",
]

# Replace these illustrative arrays with historical observations.
rng = np.random.default_rng(42)
X_history = rng.uniform(0.0, 1.0, size=(200, 4))

y_history = (
    0.4 * X_history[:, 0]
    + 0.3 * X_history[:, 1]
    + 0.2 * X_history[:, 2]
    + 0.1 * X_history[:, 3]
)

predictor = RiskPredictor(
    feature_names=feature_names,
    random_state=42,
)

predictor.fit(X_history, y_history)

current_shift = np.array([
    [0.33, 0.84, 0.57, 0.14]
])

estimated_risk = predictor.predict(current_shift)

print("Estimated risk:", estimated_risk[0])
```

**This example demonstrates API usage only.** The synthetic target above is deliberately illustrative; it does not establish that these shift metrics predict real model errors. For meaningful deployment, train and validate the predictor using actual historical downstream errors.

Keep the following consistent between training and inference:

- Metric definitions and feature order.
- The downstream model and preprocessing pipeline.
- The definition and scale of the risk target.
- The data collection and batch-construction procedure.

## Convert risk into a decision

`PreShiftGate` converts a numerical risk estimate into a monitoring decision.

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

The decision rules are:

| Condition | Decision |
|---|---|
| Risk below `accept_threshold` | `ACCEPT` |
| Risk at or above `accept_threshold`, but below `watch_threshold` | `WATCH` |
| Risk at or above `watch_threshold` | `DEFER` |

The thresholds above are examples, not recommended universal defaults. Choose thresholds using historical validation data and the operational cost of each decision. Thresholds must use the same scale as the risk model's target.

A decision is a monitoring signal; PreShift does not automatically block requests, retrain models, or guarantee safe predictions.

## End-to-end pipeline

`PreShiftPipeline` combines shift detection, risk estimation, and decision gating.

The following is a runnable API demonstration. Its risk-training data is synthetic and should be replaced with historical observations for real use.

```python
import numpy as np

from preshift import (
    ShiftDetector,
    RiskPredictor,
    PreShiftGate,
    PreShiftPipeline,
)

rng = np.random.default_rng(42)

# Reference dataset
reference_data = rng.normal(
    loc=0.0,
    scale=1.0,
    size=(500, 3),
)

metrics = ["ks", "wasserstein", "psi", "mmd"]

# Establish the exact feature order used by the detector.
detector = ShiftDetector(
    reference_data,
    metrics=metrics,
)

feature_names = detector.feature_names()

# Synthetic historical observations for API demonstration only.
X_history = rng.uniform(
    0.0,
    1.0,
    size=(200, len(feature_names)),
)

y_history = (
    0.4 * X_history[:, 0]
    + 0.3 * X_history[:, 1]
    + 0.2 * X_history[:, 2]
    + 0.1 * X_history[:, 3]
)

risk_predictor = RiskPredictor(
    feature_names=feature_names,
    random_state=42,
)

risk_predictor.fit(X_history, y_history)

gate = PreShiftGate(
    accept_threshold=0.30,
    watch_threshold=0.40,
)

pipeline = PreShiftPipeline(
    reference_data=reference_data,
    risk_predictor=risk_predictor,
    gate=gate,
    metrics=metrics,
)

incoming_data = rng.normal(
    loc=0.2,
    scale=1.1,
    size=(200, 3),
)

result = pipeline.predict(incoming_data)

print(result.summary())
print(result.to_dict())
```

The pipeline result provides:

- `result.shift_scores`: dictionary of calculated shift scores.
- `result.predicted_risk`: estimated downstream prediction risk.
- `result.decision`: `ACCEPT`, `WATCH`, or `DEFER`.
- `result.summary()`: human-readable summary.
- `result.to_dict()`: dictionary representation.

## Calibrate decision thresholds

PreShift can calculate thresholds from historical observed risk values.

```python
from preshift import calibrate_thresholds, PreShiftGate

historical_risk = [
    0.08, 0.10, 0.12, 0.15, 0.18,
    0.22, 0.27, 0.31, 0.38, 0.45,
]

accept_threshold, watch_threshold = calibrate_thresholds(
    historical_risk,
    accept_quantile=0.50,
    watch_quantile=0.75,
)

gate = PreShiftGate(
    accept_threshold=accept_threshold,
    watch_threshold=watch_threshold,
)

print("Thresholds:", accept_threshold, watch_threshold)
```

The default quantiles use the median and 75th percentile of the supplied observations. This is a calibration mechanism, not a guarantee that the resulting decisions meet any particular risk or safety target. Validate the thresholds against the intended deployment requirements.

## API reference

| Component | Responsibility |
|---|---|
| `ShiftDetector` | Calculates selected shift metrics between reference and incoming datasets. |
| `RiskPredictor` | Fits a supervised model to historical shift features and observed risk, then estimates risk for new observations. |
| `PreShiftGate` | Maps risk values to `ACCEPT`, `WATCH`, or `DEFER`. |
| `PreShiftPipeline` | Integrates detection, risk prediction, and decision gating. |
| `PreShiftResult` | Stores shift scores, predicted risk, and decision; supports summary and dictionary output. |
| `calibrate_thresholds` | Calculates decision thresholds from historical risk observations. |

Import the public API with:

```python
from preshift import (
    ShiftDetector,
    RiskPredictor,
    PreShiftGate,
    PreShiftPipeline,
    PreShiftResult,
    calibrate_thresholds,
)
```

## Input expectations

- Reference and incoming datasets must be numeric arrays with one or two dimensions.
- Reference and incoming data must contain the same number of features.
- Each dataset must contain at least one sample.
- For risk prediction, the number and order of input features must match the predictor's training data.
- Risk targets must represent the downstream error or risk definition you want to estimate.

The current detector converts inputs to floating-point arrays. Categorical variables and missing values should therefore be handled with a suitable preprocessing pipeline before passing data to PreShift.

## Limitations

PreShift is an alpha-stage library and should be evaluated before production use.

- **Shift is not the same as model failure.** A change in input distribution does not necessarily imply increased prediction error.
- **Risk prediction requires labels historically.** The risk model needs observed downstream errors for historical batches, even though inference can be performed before current-batch labels arrive.
- **Performance is application-dependent.** Results depend on the downstream model, data, shift mechanism, and risk target.
- **Thresholds are not universal.** Decision boundaries should be validated against the deployment context.
- **No automatic remediation.** PreShift reports signals and decisions; it does not independently retrain or deploy models.
- **No universal performance guarantee.** Successful execution or controlled experiments do not establish generalization to arbitrary datasets or production environments.

## Development

Clone the repository and install the package with development dependencies available in your environment:

```bash
git clone https://github.com/aryantiwari1118/PreShift.git
cd PreShift
python -m pip install -e .
python -m pytest -q
```

The test suite covers the implemented package behavior. Run the tests after modifying source code.

## Repository

- **GitHub:** https://github.com/aryantiwari1118/PreShift
- **Issues:** https://github.com/aryantiwari1118/PreShift/issues
- **License:** MIT

## Contributing

Bug reports, reproducible examples, documentation improvements, and suggestions are welcome through GitHub Issues and Pull Requests.

When reporting a problem, include the Python version, package version, a minimal reproducible example, and the observed error where possible.

## License

PreShift is distributed under the MIT License. See [LICENSE](LICENSE) for details.
