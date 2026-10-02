"""Kolmogorov-Smirnov based dataset shift detection."""

import numpy as np
from scipy.stats import ks_2samp


def ks_statistic(reference, incoming):
    """
    Calculate the two-sample Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    reference : array-like
        Reference distribution.
    incoming : array-like
        Incoming distribution.

    Returns
    -------
    float
        KS statistic in the range [0, 1].
    """

    reference = np.asarray(reference, dtype=float)
    incoming = np.asarray(incoming, dtype=float)

    reference = reference[~np.isnan(reference)]
    incoming = incoming[~np.isnan(incoming)]

    if len(reference) == 0:
        raise ValueError("Reference data contains no valid values.")

    if len(incoming) == 0:
        raise ValueError("Incoming data contains no valid values.")

    return float(
        ks_2samp(reference, incoming).statistic
    )