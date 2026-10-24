from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "Churn_Cleaned.csv"
PREDICTIONS_PATH = DATA_DIR / "processed" / "Predictions_Output.csv"

MODEL_PATH = ROOT_DIR / "models" / "best_churn_model.pkl"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_SEARCH_ITERATIONS = 40

FALSE_POSITIVE_COST = 50.0
FALSE_NEGATIVE_COST = 300.0
