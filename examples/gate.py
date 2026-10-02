"""Basic PreShift decision-gate example."""

from preshift import PreShiftGate


def main():
    gate = PreShiftGate(
        accept_threshold=0.30,
        watch_threshold=0.50
    )

    risks = [
        0.15,
        0.30,
        0.40,
        0.50,
        0.75
    ]

    print("PreShift Decision Gate")
    print("----------------------")

    for risk in risks:
        decision = gate.decide(risk)
        print(f"Risk: {risk:.2f} -> {decision}")


if __name__ == "__main__":
    main()