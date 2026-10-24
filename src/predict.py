import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.cleaning import clean_data
from src.config import MODEL_PATH, PREDICTIONS_PATH, TARGET_COLUMN
from src.features import add_features


def load_package(model_path=MODEL_PATH):
    return joblib.load(model_path)


def predict_dataframe(raw_df: pd.DataFrame, package: dict) -> pd.DataFrame:
    customer_ids = None
    if "customerID" in raw_df.columns:
        customer_ids = raw_df["customerID"].reset_index(drop=True)

    features_df = raw_df.drop(columns=[TARGET_COLUMN], errors="ignore")
    features_df = add_features(clean_data(features_df))

    probabilities = package["pipeline"].predict_proba(features_df)[:, 1]
    predictions = (probabilities >= package["threshold"]).astype(int)

    result = pd.DataFrame(
        {
            "churn_probability": probabilities.round(4),
            "predicted_churn": predictions,
        }
    )
    if customer_ids is not None:
        result.insert(0, "customerID", customer_ids)
    return result


def predict_churn(input_path, output_path=PREDICTIONS_PATH, model_path=MODEL_PATH):
    print("Loading model...")
    package = load_package(model_path)

    print(f"Reading input data: {input_path}")
    raw_df = pd.read_csv(input_path)

    print("Computing churn probabilities...")
    result = predict_dataframe(raw_df, package)

    print("\nPrediction preview:")
    print(result.head())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Predict customer churn from a raw Telco CSV file.")
    parser.add_argument("--input", required=True, help="Path to the raw input CSV file.")
    parser.add_argument("--output", default=str(PREDICTIONS_PATH), help="Path for the predictions CSV file.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the trained model package.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_churn(args.input, args.output, args.model)
