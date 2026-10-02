import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

DATA_PATH = "../../research/data/bike_controlled_clean.csv"

TARGET = "nrmse"

FEATURE_SETS = {
    "KS": [
        "ks_shift_score"
    ],
    "Wasserstein": [
        "wasserstein_shift_score"
    ],
    "PSI": [
        "psi_shift_score"
    ],
    "MMD": [
        "mmd_score"
    ],
    "Categorical": [
        "categorical_shift"
    ],
    "KS_Wasserstein": [
        "ks_shift_score",
        "wasserstein_shift_score"
    ],
    "KS_Wasserstein_PSI": [
        "ks_shift_score",
        "wasserstein_shift_score",
        "psi_shift_score"
    ],
    "Numerical_All": [
        "ks_shift_score",
        "wasserstein_shift_score",
        "psi_shift_score",
        "mmd_score"
    ],
    "All_5_Signals": [
        "ks_shift_score",
        "wasserstein_shift_score",
        "psi_shift_score",
        "mmd_score",
        "categorical_shift"
    ]
}

df = pd.read_csv(DATA_PATH)

results = []

for feature_set_name, features in FEATURE_SETS.items():

    all_predictions = []
    all_actual = []

    for magnitude in sorted(df["magnitude"].unique()):

        train = df[df["magnitude"] != magnitude]
        test = df[df["magnitude"] == magnitude]

        X_train = train[features]
        y_train = train[TARGET]

        X_test = test[features]
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

        all_predictions.extend(predictions)
        all_actual.extend(y_test.to_numpy())

    all_predictions = np.asarray(all_predictions)
    all_actual = np.asarray(all_actual)

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

    results.append({
        "feature_set": feature_set_name,
        "features": len(features),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman
    })

results_df = pd.DataFrame(results)

print("Controlled LOMO Ablation")
print("------------------------")
print(f"Samples: {len(df)}")
print(f"Magnitudes: {df['magnitude'].nunique()}")
print()

for _, row in results_df.iterrows():

    print(
        f"{row['feature_set']:<25}"
        f"MAE: {row['mae']:.6f}  "
        f"RMSE: {row['rmse']:.6f}  "
        f"R2: {row['r2']:.6f}  "
        f"Spearman: {row['spearman']:.6f}"
    )

output_path = "../../research/data/bike_lomo_ablation.csv"

results_df.to_csv(
    output_path,
    index=False
)

print()
print(f"Saved results to: {output_path}")