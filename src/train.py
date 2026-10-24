import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import clean_data
from src.config import (
    CV_FOLDS,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    N_SEARCH_ITERATIONS,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.features import CATEGORICAL_FEATURES, add_features


def load_and_prepare(raw_path=RAW_DATA_PATH, processed_path=PROCESSED_DATA_PATH):
    raw_df = pd.read_csv(raw_path)
    cleaned_df = clean_data(raw_df)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)
    return add_features(cleaned_df)


def build_pipeline(numeric_columns):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("sampler", "passthrough"),
            ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )


def tune_pipeline(pipeline, X_train, y_train):
    param_distributions = {
        "sampler": [SMOTE(random_state=RANDOM_STATE), "passthrough"],
        "classifier__n_estimators": randint(100, 600),
        "classifier__learning_rate": loguniform(0.01, 0.2),
        "classifier__max_depth": randint(2, 6),
        "classifier__subsample": uniform(0.5, 0.5),
        "classifier__min_samples_split": randint(2, 20),
        "classifier__min_samples_leaf": randint(1, 10),
        "classifier__max_features": [None, "sqrt"],
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=N_SEARCH_ITERATIONS,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def total_cost(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negatives = int(((y_pred == 0) & (y_true == 1)).sum())
    return false_positives * FALSE_POSITIVE_COST + false_negatives * FALSE_NEGATIVE_COST


def select_threshold(pipeline, X_train, y_train):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probabilities = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = [total_cost(y_train, oof_probabilities >= t) for t in thresholds]
    return float(thresholds[int(np.argmin(costs))])


def evaluate(pipeline, threshold, X_test, y_test):
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    model_cost = total_cost(y_test, predictions)
    no_model_cost = float(int(np.asarray(y_test).sum()) * FALSE_NEGATIVE_COST)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "precision_churn": float(report["1"]["precision"]),
        "recall_churn": float(report["1"]["recall"]),
        "f1_churn": float(report["1"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "model_cost": float(model_cost),
        "no_model_cost": no_model_cost,
        "cost_saving": float(no_model_cost - model_cost),
        "false_positive_cost": FALSE_POSITIVE_COST,
        "false_negative_cost": FALSE_NEGATIVE_COST,
    }
    text_report = classification_report(y_test, predictions, zero_division=0)
    return metrics, probabilities, predictions, text_report


def save_figures(y_test, probabilities, predictions):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax)
    ax.set_title("ROC Curve (Test Set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=ax)
    ax.set_title("Precision-Recall Curve (Test Set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fraction_positive, mean_predicted = calibration_curve(
        y_test, probabilities, n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_predicted, fraction_positive, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (Test Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_curve.png", dpi=150)
    plt.close(fig)


def train_model():
    print("Loading and preparing data...")
    df = load_and_prepare()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    numeric_columns = [col for col in X.columns if col not in CATEGORICAL_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    print("Running randomized hyperparameter search with cross-validation...")
    search = tune_pipeline(build_pipeline(numeric_columns), X_train, y_train)
    best_pipeline = search.best_estimator_
    print(f"Best CV PR-AUC: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")

    print("Selecting the decision threshold with out-of-fold predictions...")
    threshold = select_threshold(best_pipeline, X_train, y_train)
    print(f"Selected threshold: {threshold:.2f}")

    print("Evaluating once on the held-out test set...")
    metrics, probabilities, predictions, text_report = evaluate(
        best_pipeline, threshold, X_test, y_test
    )
    print(text_report)
    for name, value in metrics.items():
        print(f"{name}: {value}")

    metrics["best_cv_pr_auc"] = float(search.best_score_)
    metrics["best_params"] = {key: str(value) for key, value in search.best_params_.items()}

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    save_figures(y_test, probabilities, predictions)

    package = {
        "pipeline": best_pipeline,
        "threshold": threshold,
        "numeric_columns": numeric_columns,
        "categorical_columns": CATEGORICAL_FEATURES,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    train_model()
