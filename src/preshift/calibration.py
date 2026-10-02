"""Risk-threshold calibration utilities."""

import numpy as np


def calibrate_thresholds(
    observed_risk,
    accept_quantile=0.50,
    watch_quantile=0.75
):
    """
    Calculate ACCEPT and WATCH thresholds from
    observed historical prediction risk.

    Parameters
    ----------
    observed_risk : array-like
        Historical observed prediction-risk values.

    accept_quantile : float, default=0.50
        Quantile used as the ACCEPT/WATCH boundary.

    watch_quantile : float, default=0.75
        Quantile used as the WATCH/DEFER boundary.

    Returns
    -------
    tuple[float, float]
        Accept threshold and watch threshold.
    """

    risk = np.asarray(
        observed_risk,
        dtype=float
    )

    if risk.ndim != 1:
        raise ValueError(
            "observed_risk must be a 1D array."
        )

    if len(risk) == 0:
        raise ValueError(
            "observed_risk cannot be empty."
        )

    if not np.isfinite(risk).all():
        raise ValueError(
            "observed_risk must contain only finite values."
        )

    if not 0.0 <= accept_quantile <= 1.0:
        raise ValueError(
            "accept_quantile must be between 0 and 1."
        )

    if not 0.0 <= watch_quantile <= 1.0:
        raise ValueError(
            "watch_quantile must be between 0 and 1."
        )

    if accept_quantile >= watch_quantile:
        raise ValueError(
            "accept_quantile must be smaller "
            "than watch_quantile."
        )

    accept_threshold = float(
        np.quantile(
            risk,
            accept_quantile
        )
    )

    watch_threshold = float(
        np.quantile(
            risk,
            watch_quantile
        )
    )

    return (
        accept_threshold,
        watch_threshold
    )