import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import numpy as np


DATA_PATH = "../../research/data/bike_controlled_clean.csv"

FEATURES = [
    "ks_shift_score",
    "wasserstein_shift_score",
    "psi_shift_score",
    "mmd_score",
    "categorical_shift",
]

TARGET = "nrmse"


df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df[TARGET]


model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=5,
    min_samples_leaf=2
)


cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv
)


mae = mean_absolute_error(
    y,
    predictions
)

rmse = mean_squared_error(
    y,
    predictions
) ** 0.5

r2 = r2_score(
    y,
    predictions
)

spearman = spearmanr(
    y,
    predictions
).statistic


print("Controlled Risk Prediction")
print("--------------------------")
print(f"Samples: {len(df)}")
print(f"Features: {len(FEATURES)}")
print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R2: {r2:.6f}")
print(f"Spearman: {spearman:.6f}")