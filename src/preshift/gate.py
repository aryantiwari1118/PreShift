"""Risk-based decision gate for PreShift."""
import math

class PreShiftGate:
    """
    Convert estimated prediction risk into
    ACCEPT, WATCH, or DEFER decisions.

    Parameters
    ----------
    accept_threshold : float
        Risk values below this threshold are ACCEPT.

    watch_threshold : float
        Risk values below this threshold are WATCH.
        Values equal to or above it are DEFER.
    """

    def __init__(
        self,
        accept_threshold,
        watch_threshold
    ):
        if accept_threshold >= watch_threshold:
            raise ValueError(
                "accept_threshold must be smaller "
                "than watch_threshold."
            )

        self.accept_threshold = float(
            accept_threshold
        )

        self.watch_threshold = float(
            watch_threshold
        )
    @classmethod
    def from_observed_risk(
        cls,
        observed_risk,
        accept_quantile=0.50,
        watch_quantile=0.75
    ):
        """
        Create a gate by calibrating thresholds
        from observed historical risk.

        Parameters
        ----------
        observed_risk : array-like
            Historical prediction-risk values.

        accept_quantile : float, default=0.50
            Quantile used for the ACCEPT/WATCH boundary.

        watch_quantile : float, default=0.75
            Quantile used for the WATCH/DEFER boundary.

        Returns
        -------
        PreShiftGate
            Gate with calibrated thresholds.
        """

        from .calibration import calibrate_thresholds

        accept_threshold, watch_threshold = (
            calibrate_thresholds(
                observed_risk,
                accept_quantile=accept_quantile,
                watch_quantile=watch_quantile
            )
        )

        return cls(
            accept_threshold=accept_threshold,
            watch_threshold=watch_threshold
        )
    def decide(self, risk):
        """
        Convert a risk value into a decision.

        Parameters
        ----------
        risk : float
            Estimated prediction risk.

        Returns
        -------
        str
            ACCEPT, WATCH, or DEFER.
        """

        risk = float(risk)
        if not math.isfinite(risk):
            raise ValueError(
                "risk must be a finite number."
            )
        if risk < self.accept_threshold:
            return "ACCEPT"

        if risk < self.watch_threshold:
            return "WATCH"

        return "DEFER"

    def decide_batch(self, risks):
        """
        Convert multiple risk values into decisions.

        Parameters
        ----------
        risks : array-like
            Estimated prediction risks.

        Returns
        -------
        list[str]
            Decisions for each risk value.
        """

        return [
            self.decide(risk)
            for risk in risks
        ]