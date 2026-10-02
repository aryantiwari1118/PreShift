from preshift.result import PreShiftResult


def test_result_creation():
    result = PreShiftResult(
        shift_scores={
            "ks_shift_score": 0.5,
            "psi_shift_score": 1.2,
        },
        predicted_risk=0.35,
        decision="ACCEPT",
    )

    assert result.predicted_risk == 0.35
    assert result.decision == "ACCEPT"
    assert result.shift_scores["ks_shift_score"] == 0.5


def test_result_to_dict():
    result = PreShiftResult(
        shift_scores={
            "ks_shift_score": 0.5,
        },
        predicted_risk=0.35,
        decision="ACCEPT",
    )

    output = result.to_dict()

    assert output == {
        "shift_scores": {
            "ks_shift_score": 0.5,
        },
        "predicted_risk": 0.35,
        "decision": "ACCEPT",
    }
def test_result_summary():
    result = PreShiftResult(
        shift_scores={
            "ks_shift_score": 0.5,
            "psi_shift_score": 1.2,
        },
        predicted_risk=0.35,
        decision="ACCEPT",
    )

    summary = result.summary()

    assert "PreShift Analysis" in summary
    assert "Predicted risk: 0.3500" in summary
    assert "Decision: ACCEPT" in summary
    assert "ks_shift_score: 0.5000" in summary
    assert "psi_shift_score: 1.2000" in summary