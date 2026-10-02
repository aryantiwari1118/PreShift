"""Result object returned by the PreShift pipeline."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class PreShiftResult:
    """
    Store the result of a PreShift analysis.

    Parameters
    ----------
    shift_scores : dict
        Calculated dataset-shift scores.

    predicted_risk : float
        Estimated downstream prediction risk.

    decision : str
        ACCEPT, WATCH, or DEFER.
    """

    shift_scores: Dict[str, float]
    predicted_risk: float
    decision: str

    def to_dict(self):
        """
        Convert the result into a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the result.
        """

        return {
            "shift_scores": self.shift_scores,
            "predicted_risk": self.predicted_risk,
            "decision": self.decision,
        }
    def summary(self):
        """
        Return a human-readable summary of the analysis.

        Returns
        -------
        str
            Formatted PreShift analysis summary.
        """

        lines = [
            "PreShift Analysis",
            "-----------------",
            f"Predicted risk: {self.predicted_risk:.4f}",
            f"Decision: {self.decision}",
            "",
            "Shift scores:",
        ]

        for name, value in self.shift_scores.items():
            lines.append(
                f"  {name}: {value:.4f}"
            )

        return "\n".join(lines)