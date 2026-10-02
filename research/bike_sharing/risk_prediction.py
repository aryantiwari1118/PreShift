import pandas as pd
import numpy as np

from preshift import RiskPredictor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = [
    "ks_shift_score",
    "wasserstein_shift_score",
    "psi_shift_score",
    "mmd_score",
]

TARGET = "nrmse"


df = pd.read_csv("../../research/data/bike_natural_results.csv")

X = df[FEATURES].to_numpy(dtype=float)
y = df[TARGET].to_numpy(dtype=float)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

predictions = np.zeros(len(y))

for train_idx, test_idx in kf.split(X):
    predictor = RiskPredictor(
        n_estimators=300,
        random_state=42,
        max_depth=5,
        min_samples_leaf=2,
        feature_names=FEATURES
    )

    predictor.fit(
        X[train_idx],
        y[train_idx]
    )

    predictions[test_idx] = predictor.predict(
        X[test_idx]
    )

mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

pearson = np.corrcoef(
    y,
    predictions
)[0, 1]

spearman = pd.Series(y).corr(
    pd.Series(predictions),
    method="spearman"
)

print("PreShift Natural Risk Prediction")
print("--------------------------------")
print(f"Samples:  {len(y)}")
print(f"MAE:      {mae:.6f}")
print(f"RMSE:     {rmse:.6f}")
print(f"R2:       {r2:.6f}")
print(f"Pearson:  {pearson:.6f}")
print(f"Spearman: {spearman:.6f}")

df["predicted_nrmse"] = predictions

df.to_csv(
    "../../research/data/bike_natural_risk_predictions.csv",
    index=False
)

print()
print("Saved:")
print("../../research/data/bike_natural_risk_predictions.csv")