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


def apply_temperature_mean(data, magnitude):
    shifted = data.copy()
    shifted["temp"] = shifted["temp"] + magnitude
    return shifted


def apply_humidity_mean(data, magnitude):
    shifted = data.copy()
    shifted["hum"] = shifted["hum"] + magnitude
    return shifted


def apply_humidity_variance(data, magnitude):
    shifted = data.copy()
    mean = shifted["hum"].mean()
    shifted["hum"] = mean + (shifted["hum"] - mean) * (1.0 + magnitude)
    return shifted


def apply_dependency_shift(data, magnitude):
    shifted = data.copy()

    if magnitude == 0:
        return shifted

    temp = shifted["temp"].to_numpy()
    hum = shifted["hum"].to_numpy()

    order = np.argsort(temp)

    hum_by_temp = hum[order].copy()

    n = len(hum_by_temp)
    pairs = n // 2
    swaps = round(magnitude * pairs)

    for i in range(swaps):
        j = n - 1 - i
        hum_by_temp[i], hum_by_temp[j] = (
            hum_by_temp[j],
            hum_by_temp[i]
        )

    new_hum = hum.copy()
    new_hum[order] = hum_by_temp

    shifted["hum"] = new_hum

    return shifted

def apply_weather_categorical(data, magnitude):
    shifted = data.copy()

    if magnitude == 0:
        return shifted

    categories = sorted(shifted["weathersit"].unique())
    current = shifted["weathersit"].to_numpy().copy()

    n = len(current)
    changes = round(magnitude * n)

    for i in range(changes):
        index = i % n
        current_category = current[index]

        category_index = categories.index(current_category)
        next_category = categories[
            (category_index + 1) % len(categories)
        ]

        current[index] = next_category

    shifted["weathersit"] = current

    return shifted
def categorical_shift_score(reference, incoming, column):
    categories = sorted(
        set(reference[column].unique()) |
        set(incoming[column].unique())
    )

    reference_distribution = (
        reference[column]
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0.0)
    )

    incoming_distribution = (
        incoming[column]
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0.0)
    )

    return float(
        0.5 * np.abs(
            reference_distribution.to_numpy()
            - incoming_distribution.to_numpy()
        ).sum()
    )
df = pd.read_csv(
    "../../research/data/hour.csv"
)

df["dteday"] = pd.to_datetime(
    df["dteday"]
)

train = df[
    df["dteday"] <= "2011-08-31"
]

reference = df[
    (df["dteday"] >= "2011-09-01")
    & (df["dteday"] <= "2011-12-31")
].copy()

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(
    train[FEATURES],
    train[TARGET]
)

reference_numeric = reference[
    NUMERICAL_FEATURES
].to_numpy(dtype=float)

detector = ShiftDetector(
    reference_numeric
)

baseline_prediction = model.predict(
    reference[FEATURES]
)

baseline_nrmse = nrmse(
    reference[TARGET],
    baseline_prediction
)

shift_functions = {
    "temperature_mean": apply_temperature_mean,
    "humidity_mean": apply_humidity_mean,
    "humidity_variance": apply_humidity_variance,
    "dependency": apply_dependency_shift,
    "weather_categorical": apply_weather_categorical,
}

magnitudes = np.linspace(
    0.0,
    1.0,
    11
)

results = []

for shift_type, shift_function in shift_functions.items():

    for magnitude in magnitudes:

        shifted = shift_function(
            reference[FEATURES],
            magnitude
        )

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
            reference[TARGET],
            predictions
        )

        categorical_shift = categorical_shift_score(
            reference,
            shifted,
            "weathersit"
        )

        results.append({
            "shift_type": shift_type,
            "magnitude": magnitude,
            "nrmse": risk,
            "categorical_shift": categorical_shift,
            **shift_scores
        })

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "../../research/data/bike_controlled_clean.csv",
    index=False
)

print("Complete Controlled Benchmark")
print("-----------------------------")
print(
    f"Baseline NRMSE: "
    f"{baseline_nrmse:.6f}"
)

print(
    f"Observations: "
    f"{len(results_df)}"
)

print()

print(
    results_df.to_string(
        index=False
    )
)

print()
print("Saved:")
print(
    "../../research/data/bike_controlled_clean.csv"
)