"""PreShift: Prediction-aware dataset shift detection
and risk estimation.
"""

from .detector import ShiftDetector
from .risk import RiskPredictor
from .gate import PreShiftGate
from .pipeline import PreShiftPipeline
from .result import PreShiftResult
from .calibration import calibrate_thresholds

__version__ = "0.1.0"

__all__ = [
    "ShiftDetector",
    "RiskPredictor",
    "PreShiftGate",
    "PreShiftPipeline",
    "PreShiftResult",
    "calibrate_thresholds",
]