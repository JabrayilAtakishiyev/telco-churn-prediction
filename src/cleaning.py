import pandas as pd

from src.config import TARGET_COLUMN

REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NO_INTERNET_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

BINARY_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]

CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
INTERNET_SERVICE_DUMMIES = ["Fiber optic", "No"]
PAYMENT_METHOD_DUMMIES = [
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]
FLOAT_COLUMNS = ["MonthlyCharges", "TotalCharges"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "customerID" in df.columns:
        df = df.drop(columns="customerID")

    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    new_customers = (df["tenure"] == 0) & df["TotalCharges"].isna()
    df.loc[new_customers, "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    for col in NO_INTERNET_COLUMNS:
        df[col] = df[col].replace({"No internet service": "No"})
    df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    for col in BINARY_COLUMNS:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["Contract"] = df["Contract"].map(CONTRACT_MAP)

    for value in INTERNET_SERVICE_DUMMIES:
        df[f"InternetService_{value}"] = (df["InternetService"] == value).astype("int8")
    for value in PAYMENT_METHOD_DUMMIES:
        df[f"PaymentMethod_{value}"] = (df["PaymentMethod"] == value).astype("int8")
    df = df.drop(columns=["InternetService", "PaymentMethod"])

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    columns_with_nulls = df[feature_columns].columns[df[feature_columns].isna().any()].tolist()
    if columns_with_nulls:
        raise ValueError(f"Unexpected or missing values found in columns: {columns_with_nulls}")
    if TARGET_COLUMN in df.columns and df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Unexpected values found in target column: {TARGET_COLUMN}")

    for col in feature_columns:
        if col in FLOAT_COLUMNS:
            df[col] = df[col].astype(float)
        elif col == "tenure":
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype("int8")
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int8")

    return df.reset_index(drop=True)
