import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


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

X_train = train[FEATURES]
y_train = train[TARGET]

X_reference = reference[FEATURES]
y_reference = reference[TARGET]

X_incoming = incoming[FEATURES]
y_incoming = incoming[TARGET]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

reference_pred = model.predict(X_reference)
incoming_pred = model.predict(X_incoming)

reference_nrmse = nrmse(y_reference, reference_pred)
incoming_nrmse = nrmse(y_incoming, incoming_pred)

print("Bike Sharing Baseline")
print("---------------------")
print(f"Training rows:   {len(train)}")
print(f"Reference rows:  {len(reference)}")
print(f"Incoming rows:   {len(incoming)}")
print()
print(f"Reference NRMSE: {reference_nrmse:.6f}")
print(f"Incoming NRMSE:  {incoming_nrmse:.6f}")
print(
    f"Relative change: "
    f"{((incoming_nrmse - reference_nrmse) / reference_nrmse) * 100:.2f}%"
)