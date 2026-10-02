"""Unified multi-metric dataset shift detector."""

import numpy as np

from .ks import ks_statistic
from .wasserstein import wasserstein_distance_score
from .psi import psi_score
from .mmd import mmd_score
from .utils import DEFAULT_METRICS, METRIC_OUTPUT_NAMES


class ShiftDetector:
    """
    Calculate selected dataset-shift metrics.

    Parameters
    ----------
    reference : array-like
        Reference dataset.

    metrics : list of str, optional
        Metrics to calculate.

        Supported metrics:
        - "ks"
        - "wasserstein"
        - "psi"
        - "mmd"

        If None, all supported metrics are used.
    """

    SUPPORTED_METRICS = set(DEFAULT_METRICS)

    def __init__(
        self,
        reference,
        metrics=None
    ):
        self.reference = self._prepare_data(
            reference
        )

        if metrics is None:
            metrics = DEFAULT_METRICS.copy()

        if not metrics:
            raise ValueError(
                "At least one metric must be selected."
            )

        invalid_metrics = (
            set(metrics)
            - self.SUPPORTED_METRICS
        )

        if invalid_metrics:
            raise ValueError(
                f"Unsupported metrics: "
                f"{sorted(invalid_metrics)}. "
                f"Supported metrics: "
                f"{sorted(self.SUPPORTED_METRICS)}."
            )

        self.metrics = list(metrics)

    @staticmethod
    def _prepare_data(data):
        data = np.asarray(
            data,
            dtype=float
        )

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if data.ndim != 2:
            raise ValueError(
                "Data must be a 1D or 2D array."
            )

        if data.shape[0] == 0:
            raise ValueError(
                "Data must contain at least one sample."
            )

        return data

    def feature_names(self):
        """
        Return shift-feature names in deterministic order.

        Returns
        -------
        list[str]
            Output feature names corresponding to the
            selected metrics.
        """

        return [
            METRIC_OUTPUT_NAMES[metric]
            for metric in self.metrics
        ]

    def to_feature_array(self, shift_scores):
        """
        Convert shift-score dictionary into a
        deterministic feature array.

        Parameters
        ----------
        shift_scores : dict
            Result returned by ``detect()``.

        Returns
        -------
        numpy.ndarray
            One-dimensional feature array.
        """

        expected_features = self.feature_names()

        missing_features = [
            name
            for name in expected_features
            if name not in shift_scores
        ]

        if missing_features:
            raise ValueError(
                "Missing shift features: "
                f"{missing_features}"
            )

        return np.asarray(
            [
                shift_scores[name]
                for name in expected_features
            ],
            dtype=float
        )

    def detect(self, incoming):
        """
        Calculate selected shift metrics.

        Parameters
        ----------
        incoming : array-like
            Incoming dataset.

        Returns
        -------
        dict
            Dictionary containing requested shift scores.
        """

        incoming = self._prepare_data(
            incoming
        )

        if (
            self.reference.shape[1]
            != incoming.shape[1]
        ):
            raise ValueError(
                "Reference and incoming data must "
                "have the same number of features."
            )

        results = {}

        if "ks" in self.metrics:
            scores = []

            for feature_index in range(
                self.reference.shape[1]
            ):
                scores.append(
                    ks_statistic(
                        self.reference[:, feature_index],
                        incoming[:, feature_index]
                    )
                )

            results[
                METRIC_OUTPUT_NAMES["ks"]
            ] = float(np.mean(scores))

        if "wasserstein" in self.metrics:
            scores = []

            for feature_index in range(
                self.reference.shape[1]
            ):
                scores.append(
                    wasserstein_distance_score(
                        self.reference[:, feature_index],
                        incoming[:, feature_index]
                    )
                )

            results[
                METRIC_OUTPUT_NAMES["wasserstein"]
            ] = float(np.mean(scores))

        if "psi" in self.metrics:
            scores = []

            for feature_index in range(
                self.reference.shape[1]
            ):
                scores.append(
                    psi_score(
                        self.reference[:, feature_index],
                        incoming[:, feature_index]
                    )
                )

            results[
                METRIC_OUTPUT_NAMES["psi"]
            ] = float(np.mean(scores))

        if "mmd" in self.metrics:
            results[
                METRIC_OUTPUT_NAMES["mmd"]
            ] = float(
                mmd_score(
                    self.reference,
                    incoming
                )
            )

        return results