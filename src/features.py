import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

TENURE_BINS = [-1, 12, 24, 48, 60, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-60", "61+"]
CATEGORICAL_FEATURES = ["tenure_group"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["services_count"] = df[SERVICE_COLUMNS].sum(axis=1).astype(int)
    df["avg_monthly_charge_per_tenure"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    return df
