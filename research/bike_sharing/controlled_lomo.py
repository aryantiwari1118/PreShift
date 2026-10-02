import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr


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

all_predictions = []
all_actual = []


for magnitude in sorted(df["magnitude"].unique()):

    train = df[df["magnitude"] != magnitude]
    test = df[df["magnitude"] == magnitude]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=5,
        min_samples_leaf=2
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    all_predictions.extend(
        predictions
    )

    all_actual.extend(
        y_test.to_numpy()
    )


all_predictions = np.asarray(
    all_predictions
)

all_actual = np.asarray(
    all_actual
)


mae = mean_absolute_error(
    all_actual,
    all_predictions
)

rmse = mean_squared_error(
    all_actual,
    all_predictions
) ** 0.5

r2 = r2_score(
    all_actual,
    all_predictions
)

spearman = spearmanr(
    all_actual,
    all_predictions
).statistic


print("Controlled LOMO Risk Prediction")
print("-------------------------------")
print(f"Samples: {len(df)}")
print(f"Magnitudes: {df['magnitude'].nunique()}")
print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R2: {r2:.6f}")
print(f"Spearman: {spearman:.6f}")