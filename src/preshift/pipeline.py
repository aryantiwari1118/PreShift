"""High-level PreShift prediction-risk pipeline."""
from .detector import ShiftDetector
from .risk import RiskPredictor
from .gate import PreShiftGate
from .result import PreShiftResult


class PreShiftPipeline:
    """
    End-to-end PreShift pipeline.

    Flow
    ----
    Reference data
        ↓
    Shift detection
        ↓
    Risk prediction
        ↓
    ACCEPT / WATCH / DEFER
    """

    def __init__(
        self,
        reference_data,
        risk_predictor,
        gate,
        metrics=None
    ):
        self.risk_predictor = risk_predictor
        self.detector = ShiftDetector(
            reference_data,
            metrics=metrics
        )

        detector_features = self.detector.feature_names()

        if self.risk_predictor.feature_names() is not None:
            if (
                self.risk_predictor.feature_names()
                != detector_features
            ):
                raise ValueError(
                    "RiskPredictor feature names do not match "
                    "ShiftDetector feature names."
                )

        self.gate = gate

    def predict(self, incoming_data):
        """
        Analyze incoming data and produce
        shift scores, estimated risk, and decision.

        Parameters
        ----------
        incoming_data : array-like
            Incoming unlabeled data.

        Returns
        -------
        dict
            PreShift analysis result.
        """

        shift_scores = self.detector.detect(
            incoming_data
        )

        shift_features = (
            self.detector.to_feature_array(
                shift_scores
            )
        )

        predicted_risk = float(
            self.risk_predictor.predict(
                shift_features
            )[0]
        )

        decision = self.gate.decide(
            predicted_risk
        )

        return PreShiftResult(
            shift_scores=shift_scores,
            predicted_risk=predicted_risk,
            decision=decision
        )