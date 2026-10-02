import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from preshift import ShiftDetector


FEATURES = [
    "season",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

NUMERICAL_FEATURES = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

TARGET = "cnt"


def nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / np.std(y_true)


df = pd.read_csv("../../research/data/hour.csv")
df["dteday"] = pd.to_datetime(df["dteday"])

train = df[df["dteday"] <= "2011-08-31"]
reference = df[
    (df["dteday"] >= "2011-09-01")
    & (df["dteday"] <= "2011-12-31")
]

incoming = df[df["dteday"] >= "2012-01-01"]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(
    train[FEATURES],
    train[TARGET]
)

X_reference = reference[FEATURES]
y_reference = reference[TARGET]

reference_pred = model.predict(X_reference)

baseline_nrmse = nrmse(
    y_reference,
    reference_pred
)

reference_numeric = reference[NUMERICAL_FEATURES].to_numpy(dtype=float)

detector = ShiftDetector(reference_numeric)

X_incoming = incoming[FEATURES].copy()
y_incoming = incoming[TARGET].to_numpy()

results = []

magnitudes = np.linspace(0.0, 1.0, 11)

for magnitude in magnitudes:

    shifted = X_incoming.copy()

    shifted["temp"] = shifted["temp"] + magnitude

    shifted_numeric = shifted[
        NUMERICAL_FEATURES
    ].to_numpy(dtype=float)

    shift_scores = detector.detect(
        shifted_numeric
    )

    predictions = model.predict(
        shifted
    )

    risk = nrmse(
        y_incoming,
        predictions
    )

    results.append({
        "shift_type": "temperature_mean",
        "magnitude": magnitude,
        "nrmse": risk,
        **shift_scores
    })


results_df = pd.DataFrame(results)

print("Controlled Temperature Mean Shift")
print("---------------------------------")
print(f"Baseline reference NRMSE: {baseline_nrmse:.6f}")
print()

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    "../../research/data/bike_controlled_temperature.csv",
    index=False
)

print()
print("Saved:")
print("../../research/data/bike_controlled_temperature.csv")