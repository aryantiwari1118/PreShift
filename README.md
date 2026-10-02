# PreShift

**Prediction-aware dataset shift detection and risk estimation for machine learning.**

PreShift is a Python library for characterizing distributional changes in incoming data and estimating their potential downstream impact on machine learning prediction performance.

Traditional dataset-shift monitoring asks:

> Has the incoming data distribution changed?

PreShift extends this workflow by asking:

> Can observed distributional changes be used to estimate downstream prediction risk before true outcomes become available?

The library combines multiple statistical shift signals with a prediction-risk model and an optional decision layer for operational monitoring.

---

## Key idea

A distributional change does not necessarily imply that a deployed model will perform worse.

PreShift therefore separates the monitoring problem into three stages:

```text
Incoming unlabeled data
          │
          ▼
   Shift characterization
          │
          ▼
 Multi-signal representation
          │
          ▼
    Risk prediction
          │
          ▼
 ACCEPT / WATCH / DEFER
```

The goal is to estimate potential downstream prediction degradation before delayed ground-truth labels become available.

---

## Features

PreShift currently provides:

* Kolmogorov-Smirnov (KS) shift detection
* Wasserstein distance
* Population Stability Index (PSI)
* Maximum Mean Discrepancy (MMD)
* Configurable multi-metric shift detection
* Multi-signal feature representation
* Prediction-risk estimation
* Risk-based decision gating
* ACCEPT / WATCH / DEFER decisions
* Quantile-based threshold calibration
* Structured pipeline results
* Input validation and error handling
* Research-oriented examples and experiments

---

## Installation

### Clone the repository

```bash
git clone https://github.com/aryantiwari1118/PreShift.git
cd PreShift
```

### Install in editable mode

```bash
pip install -e .
```

### Requirements

* Python 3.10+
* NumPy
* Pandas
* SciPy
* scikit-learn

---

## Quick start

### 1. Detect dataset shift

```python
from preshift import ShiftDetector

detector = ShiftDetector()

detector.fit(reference_data)

result = detector.detect(incoming_data)

print(result)
```

The detector can use multiple statistical signals to characterize the difference between reference and incoming data.

Supported metrics include:

```text
KS
Wasserstein
PSI
MMD
```

---

## 2. Estimate prediction risk

A risk predictor can be trained using historical shift representations and observed downstream model errors.

```python
from preshift import RiskPredictor

predictor = RiskPredictor()

predictor.fit(
    X_shift_features,
    y_observed_risk
)

predicted_risk = predictor.predict(
    X_current_shift
)

print(predicted_risk)
```

Here:

* `X_shift_features` represents observed distributional-shift signals.
* `y_observed_risk` represents historical downstream prediction error.
* `predicted_risk` estimates expected downstream error for the current incoming batch.

---

## 3. Decision gating

Predicted risk can be converted into an operational monitoring decision.

```python
from preshift import PreShiftGate

gate = PreShiftGate(
    accept_threshold=0.30,
    watch_threshold=0.40
)

decision = gate.decide(predicted_risk)

print(decision)
```

Possible decisions are:

```text
ACCEPT
WATCH
DEFER
```

Conceptually:

```text
Low estimated risk
       │
       ▼
    ACCEPT

Moderate estimated risk
       │
       ▼
     WATCH

High estimated risk
       │
       ▼
     DEFER
```

Thresholds should be calibrated for the deployment setting rather than treated as universal values.

---

## 4. End-to-end pipeline

PreShift also provides an integrated pipeline.

```python
from preshift import (
    ShiftDetector,
    RiskPredictor,
    PreShiftGate,
    PreShiftPipeline
)

detector = ShiftDetector()

risk_predictor = RiskPredictor(
    feature_names=detector.feature_names
)

gate = PreShiftGate(
    accept_threshold=0.30,
    watch_threshold=0.40
)

pipeline = PreShiftPipeline(
    reference_data=reference_data,
    risk_predictor=risk_predictor,
    gate=gate
)

result = pipeline.predict(
    incoming_data
)

print(result.summary())
```

The resulting analysis contains:

* shift scores
* predicted risk
* monitoring decision

---

# Research motivation

Machine learning systems deployed over time may encounter data whose distribution differs from the historical training environment.

A conventional drift detector can identify statistical differences between a reference distribution and incoming data. However, statistical shift alone does not directly quantify how much a downstream model's prediction performance will change.

PreShift investigates a prediction-aware alternative:

```text
P_train(X) ≠ P_incoming(X)
```

is treated as evidence of distributional change, while the downstream objective is to estimate the corresponding prediction risk.

The framework therefore studies the relationship between:

```text
Observed distributional shift
              ↓
      Shift representation
              ↓
      Downstream model error
```

---

# Shift detection

PreShift currently implements four primary statistical signals.

| Method      | Purpose                                                         |
| ----------- | --------------------------------------------------------------- |
| KS          | Measures the maximum difference between empirical distributions |
| Wasserstein | Measures distributional distance using transport cost           |
| PSI         | Measures population distribution changes across bins            |
| MMD         | Measures multivariate distributional discrepancy using a kernel |

Different metrics capture different aspects of distributional change. PreShift therefore supports combining multiple signals rather than relying on a single statistic.

---

# Prediction-aware risk estimation

The central research component is the risk prediction layer.

Historical observations can be represented as:

```text
Shift features → Observed downstream error
```

A supervised risk model learns this relationship.

For a new incoming batch:

```text
Reference data
      +
Incoming unlabeled data
      ↓
Shift metrics
      ↓
Risk predictor
      ↓
Estimated downstream error
```

The resulting estimate can be used before true labels for the incoming batch become available.

---

# Decision support

The predicted risk can optionally be mapped to operational actions:

```text
Estimated risk

      LOW
       │
       ▼
    ACCEPT

    MEDIUM
       │
       ▼
     WATCH

     HIGH
       │
       ▼
     DEFER
```

These thresholds are deployment-specific and should be calibrated using historical validation data.

PreShift does not assume that a particular threshold is universally appropriate.

---

# Research experiments

The repository contains experiments investigating prediction-aware dataset-shift monitoring.

## Bike Sharing

The controlled benchmark uses the UCI Bike Sharing dataset and investigates several shift mechanisms:

* temperature mean shift
* humidity mean shift
* humidity variance shift
* feature-dependency shift
* categorical weather shift

Controlled shift magnitudes range from 0.0 to 1.0.

The repository also contains a natural chronological evaluation using temporal batches.

---

## Air Quality

Air Quality data is used as an additional validation dataset.

The experiments evaluate whether relationships between shift signals and downstream prediction degradation can be observed beyond the Bike Sharing benchmark.

---

# Controlled LOMO evaluation

The controlled Bike Sharing benchmark evaluates risk prediction using **leave-one-magnitude-out (LOMO)** validation.

For each magnitude:

```text
Training:
all other shift magnitudes

Testing:
held-out magnitude
```

This evaluates whether the risk predictor can estimate downstream degradation at a magnitude that was not directly included in training.

Current controlled benchmark result:

| Metric               |     LOMO |
| -------------------- | -------: |
| MAE                  | 0.021856 |
| RMSE                 | 0.032586 |
| R²                   | 0.712263 |
| Spearman correlation | 0.644242 |

These results are specific to the controlled Bike Sharing experiment and should not be interpreted as universal performance guarantees.

---

# Ablation study

The controlled benchmark also evaluates different shift-signal combinations.

| Feature set            |          MAE |         RMSE |           R² |     Spearman |
| ---------------------- | -----------: | -----------: | -----------: | -----------: |
| KS                     |     0.044801 |     0.059580 |     0.038114 |     0.014329 |
| Wasserstein            |     0.040844 |     0.052224 |     0.260964 |     0.065676 |
| PSI                    |     0.030032 |     0.043827 |     0.479513 |     0.242487 |
| MMD                    |     0.041615 |     0.055730 |     0.158390 |     0.106137 |
| Categorical            |     0.044753 |     0.061428 |    -0.022488 |    -0.396737 |
| KS + Wasserstein       |     0.042102 |     0.054799 |     0.186272 |     0.054341 |
| KS + Wasserstein + PSI |     0.027817 |     0.036408 |     0.640817 |     0.261490 |
| Numerical signals      |     0.025936 |     0.035687 |     0.654891 |     0.361210 |
| All five signals       | **0.021856** | **0.032586** | **0.712263** | **0.644242** |

The controlled ablation indicates that combining heterogeneous shift signals can improve prediction of downstream degradation relative to individual signals in this benchmark.

---

# Important research caveats

PreShift is a research-oriented library and the current results have several limitations.

### Controlled experiments

Controlled shift experiments are performed on specific datasets and prediction models. Results should not be interpreted as evidence of universal performance across datasets or deployment environments.

### Risk-model dependence

The risk predictor requires historical examples connecting observed shift features with downstream model error.

### Threshold dependence

ACCEPT / WATCH / DEFER thresholds are application-dependent and should be calibrated for the intended deployment environment.

### Unlabeled incoming data

Prediction-aware risk estimation is designed for situations where incoming features are available before corresponding ground-truth labels. Once labels become available, actual model performance should be measured and used for monitoring and recalibration.

### Generalization

The current experiments do not establish universal generalization to arbitrary datasets, prediction models, or unseen shift mechanisms.

---

# Repository structure

```text
PreShift/
│
├── src/
│   └── preshift/
│       ├── __init__.py
│       ├── calibration.py
│       ├── detector.py
│       ├── gate.py
│       ├── ks.py
│       ├── mmd.py
│       ├── pipeline.py
│       ├── psi.py
│       ├── result.py
│       ├── risk.py
│       ├── utils.py
│       └── wasserstein.py
│
├── tests/
│
├── examples/
│
├── research/
│   ├── bike_sharing/
│   ├── air_quality/
│   └── data/
│
├── README.md
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── pyproject.toml
└── .gitignore
```

---

# Testing

The current test suite covers the core library components, including:

* shift metrics
* shift detector
* risk predictor
* decision gate
* calibration
* pipeline
* structured results

Run the test suite with:

```bash
python -m pytest -q
```

Current development status:

```text
51 tests passed
```

---

# Building the package

Install the build tool:

```bash
pip install build
```

Build the source distribution and wheel:

```bash
python -m build
```

The generated files will be placed in:

```text
dist/
```

---

# Examples

Example scripts are available under:

```text
examples/
```

including:

```text
basic_detection.py
full_pipeline.py
gate.py
risk_prediction.py
wheel_smoke_test.py
```

These examples demonstrate the main library functionality independently of the research experiments.

---

# Version

Current version:

```text
0.1.0
```

PreShift is currently an early research/software release.

See [`CHANGELOG.md`](CHANGELOG.md) for release information.

---

# License

PreShift is released under the MIT License.

See [`LICENSE`](LICENSE) for the complete license text.

---

# Citation

If you use PreShift in academic or research work, please cite the software using the metadata provided in:

```text
CITATION.cff
```

Repository:

https://github.com/aryantiwari1118/PreShift

---

# Disclaimer

PreShift is intended for research and experimental machine-learning monitoring.

Predicted risk values are estimates produced from historical relationships between shift characteristics and observed downstream model performance. They should not automatically be interpreted as guarantees of future model behavior.

Deployment decisions should consider application-specific validation, calibration, monitoring requirements, and the availability of ground-truth outcomes.

---

# Author

**Aryan Tiwari**

GitHub:
https://github.com/aryantiwari1118
