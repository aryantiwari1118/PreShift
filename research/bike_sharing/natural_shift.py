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

reference_pred = model.predict(reference[FEATURES])
reference_nrmse = nrmse(reference[TARGET], reference_pred)

reference_numeric = reference[NUMERICAL_FEATURES].to_numpy(dtype=float)

detector = ShiftDetector(reference_numeric)

incoming_days = sorted(incoming["dteday"].dt.normalize().unique())

results = []

for start in range(0, len(incoming_days), 7):
    batch_days = incoming_days[start:start + 7]

    if len(batch_days) < 7:
        continue

    batch = incoming[
        incoming["dteday"].dt.normalize().isin(batch_days)
    ]

    if len(batch) == 0:
        continue

    X_batch = batch[NUMERICAL_FEATURES].to_numpy(dtype=float)
    predictions = model.predict(batch[FEATURES])

    batch_nrmse = nrmse(
        batch[TARGET],
        predictions
    )

    shift_scores = detector.detect(X_batch)

    results.append({
        "batch": len(results) + 1,
        "start": batch_days[0],
        "end": batch_days[-1],
        "nrmse": batch_nrmse,
        **shift_scores
    })


results_df = pd.DataFrame(results)

print("Bike Sharing Natural Shift Experiment")
print("--------------------------------------")
print(f"Reference NRMSE: {reference_nrmse:.6f}")
print(f"Number of 7-day batches: {len(results_df)}")
print()

print(
    results_df[
        [
            "batch",
            "start",
            "end",
            "nrmse",
            "ks_shift_score",
            "wasserstein_shift_score",
            "psi_shift_score",
            "mmd_score",
        ]
    ].to_string(index=False)
)

print()
print("Correlations with NRMSE")
print("-----------------------")

for metric in [
    "ks_shift_score",
    "wasserstein_shift_score",
    "psi_shift_score",
    "mmd_score",
]:
    pearson = results_df[metric].corr(results_df["nrmse"])
    spearman = results_df[metric].corr(
        results_df["nrmse"],
        method="spearman"
    )

    print(
        f"{metric}: "
        f"Pearson={pearson:.4f}, "
        f"Spearman={spearman:.4f}"
    )
results_df.to_csv(
    "../../research/data/bike_natural_results.csv",
    index=False
)

print()
print("Saved:")
print("../../research/data/bike_natural_results.csv")