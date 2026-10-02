import pytest

from preshift.gate import PreShiftGate


def test_accept_decision():
    gate = PreShiftGate(
        accept_threshold=0.38,
        watch_threshold=0.41
    )

    assert gate.decide(0.35) == "ACCEPT"


def test_watch_decision():
    gate = PreShiftGate(
        accept_threshold=0.38,
        watch_threshold=0.41
    )

    assert gate.decide(0.40) == "WATCH"


def test_defer_decision():
    gate = PreShiftGate(
        accept_threshold=0.38,
        watch_threshold=0.41
    )

    assert gate.decide(0.45) == "DEFER"


def test_batch_decisions():
    gate = PreShiftGate(
        accept_threshold=0.38,
        watch_threshold=0.41
    )

    result = gate.decide_batch(
        [0.35, 0.40, 0.45]
    )

    assert result == [
        "ACCEPT",
        "WATCH",
        "DEFER"
    ]


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        PreShiftGate(
            accept_threshold=0.50,
            watch_threshold=0.40
        )

def test_gate_from_observed_risk():
    gate = PreShiftGate.from_observed_risk(
        [
            0.20,
            0.30,
            0.35,
            0.40,
            0.50,
            0.60,
        ]
    )

    assert gate.accept_threshold == 0.375
    assert gate.watch_threshold == 0.475

    assert gate.decide(0.30) == "ACCEPT"
    assert gate.decide(0.42) == "WATCH"
    assert gate.decide(0.50) == "DEFER"
def test_gate_from_observed_risk_rejects_invalid_data():
    with pytest.raises(ValueError):
        PreShiftGate.from_observed_risk([])
def test_nan_risk_is_rejected():
    gate = PreShiftGate(
        accept_threshold=0.38,
        watch_threshold=0.41
    )

    with pytest.raises(ValueError):
        gate.decide(float("nan"))


def test_infinite_risk_is_rejected():
    gate = PreShiftGate(
        accept_threshold=0.38,
        watch_threshold=0.41
    )

    with pytest.raises(ValueError):
        gate.decide(float("inf"))