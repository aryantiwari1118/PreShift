# PreShift

Prediction-aware dataset shift detection and risk estimation for machine learning.

## Overview

PreShift is a Python library for analyzing distributional changes in incoming data and estimating their potential impact on downstream model prediction risk.

Traditional dataset-shift detection answers:

> Has the incoming data distribution changed?

PreShift additionally investigates:

> Can observed distributional changes be used to estimate downstream prediction risk before true outcomes become available?

The library currently provides:

- Kolmogorov-Smirnov (KS) shift detection
- Wasserstein distance
- Population Stability Index (PSI)
- Maximum Mean Discrepancy (MMD)
- Multi-metric shift detection
- Prediction-risk estimation
- Risk-based ACCEPT / WATCH / DEFER decisions
- Historical risk-based threshold calibration
- Structured pipeline results

## Installation

Clone the repository and install it in editable mode:

```bash
git clone <repository-url>
cd PreShift
pip install -e .