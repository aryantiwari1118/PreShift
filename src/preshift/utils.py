"""Utility functions for PreShift."""


DEFAULT_METRICS = [
    "ks",
    "wasserstein",
    "psi",
    "mmd",
]


METRIC_OUTPUT_NAMES = {
    "ks": "ks_shift_score",
    "wasserstein": "wasserstein_shift_score",
    "psi": "psi_shift_score",
    "mmd": "mmd_score",
}


def metric_output_name(metric):
    """
    Return the output feature name for a metric.

    Parameters
    ----------
    metric : str
        Metric name.

    Returns
    -------
    str
        Corresponding output feature name.

    Raises
    ------
    ValueError
        If the metric is unsupported.
    """

    if metric not in METRIC_OUTPUT_NAMES:
        raise ValueError(
            f"Unsupported metric: {metric}"
        )

    return METRIC_OUTPUT_NAMES[metric]