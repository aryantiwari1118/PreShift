"""Population Stability Index based dataset shift detection."""

import numpy as np


def psi_score(
    reference,
    incoming,
    bins=10,
    epsilon=1e-6
):
    """
    Calculate the Population Stability Index (PSI).

    Parameters
    ----------
    reference : array-like
        Reference distribution.
    incoming : array-like
        Incoming distribution.
    bins : int, default=10
        Number of percentile-based bins.
    epsilon : float, default=1e-6
        Small value used to avoid division by zero or log(0).

    Returns
    -------
    float
        PSI score.
    """

    reference = np.asarray(reference, dtype=float)
    incoming = np.asarray(incoming, dtype=float)

    reference = reference[~np.isnan(reference)]
    incoming = incoming[~np.isnan(incoming)]

    if len(reference) == 0:
        raise ValueError("Reference data contains no valid values.")

    if len(incoming) == 0:
        raise ValueError("Incoming data contains no valid values.")

    if bins < 2:
        raise ValueError("bins must be at least 2.")

    # Create percentile-based bin edges from reference data.
    percentiles = np.linspace(0, 100, bins + 1)

    edges = np.percentile(
        reference,
        percentiles
    )

    # Remove duplicate edges caused by repeated values.
    edges = np.unique(edges)

    # A constant reference distribution cannot form meaningful bins.
    if len(edges) < 2:
        return 0.0

    # Make sure the extreme incoming values are included.
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference,
        bins=edges
    )

    incoming_counts, _ = np.histogram(
        incoming,
        bins=edges
    )

    reference_proportions = (
        reference_counts / len(reference)
    )

    incoming_proportions = (
        incoming_counts / len(incoming)
    )

    reference_proportions = np.clip(
        reference_proportions,
        epsilon,
        None
    )

    incoming_proportions = np.clip(
        incoming_proportions,
        epsilon,
        None
    )

    psi = np.sum(
        (
            incoming_proportions
            - reference_proportions
        )
        *
        np.log(
            incoming_proportions
            / reference_proportions
        )
    )

    return float(psi)