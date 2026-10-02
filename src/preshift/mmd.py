"""Maximum Mean Discrepancy based dataset shift detection."""

import numpy as np
from sklearn.metrics import pairwise_distances


def mmd_score(
    reference,
    incoming
):
    """
    Calculate the squared Maximum Mean Discrepancy (MMD)
    using an RBF kernel with median-distance bandwidth.

    Parameters
    ----------
    reference : array-like
        Reference samples.
        Shape: (n_samples,) or (n_samples, n_features).

    incoming : array-like
        Incoming samples.
        Shape: (n_samples,) or (n_samples, n_features).

    Returns
    -------
    float
        Squared MMD estimate.
    """

    reference = np.asarray(
        reference,
        dtype=float
    )

    incoming = np.asarray(
        incoming,
        dtype=float
    )

    # Convert one-dimensional input into
    # a two-dimensional feature matrix.
    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)

    if incoming.ndim == 1:
        incoming = incoming.reshape(-1, 1)

    if reference.ndim != 2:
        raise ValueError(
            "Reference data must be 1D or 2D."
        )

    if incoming.ndim != 2:
        raise ValueError(
            "Incoming data must be 1D or 2D."
        )

    if reference.shape[1] != incoming.shape[1]:
        raise ValueError(
            "Reference and incoming data must "
            "have the same number of features."
        )

    # Remove rows containing NaN values.
    reference = reference[
        ~np.isnan(reference).any(axis=1)
    ]

    incoming = incoming[
        ~np.isnan(incoming).any(axis=1)
    ]

    if len(reference) == 0:
        raise ValueError(
            "Reference data contains no valid samples."
        )

    if len(incoming) == 0:
        raise ValueError(
            "Incoming data contains no valid samples."
        )

    # Combine both datasets to estimate the
    # median pairwise distance.
    combined = np.vstack(
        [reference, incoming]
    )

    distances = pairwise_distances(
        combined,
        metric="euclidean"
    )

    non_zero_distances = distances[
        distances > 0
    ]

    # If every sample is identical, there is no
    # detectable distributional difference.
    if len(non_zero_distances) == 0:
        return 0.0

    median_distance = np.median(
        non_zero_distances
    )

    # RBF kernel bandwidth.
    gamma = 1 / (
        2 * median_distance ** 2
    )

    K_xx = np.exp(
        -gamma
        * pairwise_distances(
            reference,
            reference,
            metric="sqeuclidean"
        )
    )

    K_yy = np.exp(
        -gamma
        * pairwise_distances(
            incoming,
            incoming,
            metric="sqeuclidean"
        )
    )

    K_xy = np.exp(
        -gamma
        * pairwise_distances(
            reference,
            incoming,
            metric="sqeuclidean"
        )
    )

    mmd_squared = (
        K_xx.mean()
        + K_yy.mean()
        - 2 * K_xy.mean()
    )

    # Numerical precision can occasionally produce
    # a tiny negative value.
    return float(
        max(mmd_squared, 0.0)
    )