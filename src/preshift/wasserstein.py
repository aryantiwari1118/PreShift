"""Wasserstein distance based dataset shift detection."""

import numpy as np
from scipy.stats import wasserstein_distance


def wasserstein_distance_score(reference, incoming):
    """
    Calculate the first Wasserstein distance between
    reference and incoming distributions.

    Parameters
    ----------
    reference : array-like
        Reference distribution.
    incoming : array-like
        Incoming distribution.

    Returns
    -------
    float
        Wasserstein distance.
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
        wasserstein_distance(
            reference,
            incoming
        )
    )