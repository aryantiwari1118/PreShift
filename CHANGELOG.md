# Changelog

All notable changes to PreShift are documented in this file.

The format follows a simplified version of Keep a Changelog.

## [0.1.0] - 2026-09-29

### Added

* Statistical dataset-shift detection using Kolmogorov-Smirnov statistic.
* Wasserstein distance-based shift detection.
* Population Stability Index (PSI).
* Multivariate Maximum Mean Discrepancy (MMD).
* Configurable shift detection through `ShiftDetector`.
* Multi-signal shift representation.
* Random Forest-based prediction risk estimation through `RiskPredictor`.
* Risk-based decision gating through `PreShiftGate`.
* Accept, watch, and defer monitoring decisions.
* Quantile-based threshold calibration.
* End-to-end monitoring through `PreShiftPipeline`.
* Structured analysis results through `PreShiftResult`.
* Input validation and error handling.
* Unit test suite covering the core library components.
* Basic usage examples.
* Research-oriented documentation.
* MIT license.
* Citation metadata through `CITATION.cff`.

### Research

* Controlled dataset-shift benchmarking.
* Natural temporal dataset-shift experiments.
* Prediction-aware risk estimation experiments.
* Leave-one-magnitude-out validation.
* Multi-signal ablation experiments.
* Bike Sharing benchmark.
* Air Quality benchmark.

### Packaging

* Initial Python package release.
* Source distribution (`sdist`).
* Python wheel distribution.
* Python 3.10+ support.
