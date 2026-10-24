import re

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.cleaning import clean_data
from src.config import (
    FIGURES_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.features import add_features

SHAP_FIGURES_DIR = FIGURES_DIR / "shap"
SHAP_IMPORTANCE_PATH = REPORTS_DIR / "shap_importance.csv"
SHAP_FINDINGS_PATH = REPORTS_DIR / "shap_findings.md"
TOP_FEATURES_FOR_DEPENDENCE = 3
MAX_DISPLAY = 15


def load_test_data():
    df = add_features(clean_data(pd.read_csv(RAW_DATA_PATH)))
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def build_display_frame(pipeline, X_test, numeric_columns):
    preprocessor = pipeline.named_steps["preprocessor"]
    matrix = preprocessor.transform(X_test)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    feature_names = [name.split("__", 1)[1] for name in preprocessor.get_feature_names_out()]
    model_frame = pd.DataFrame(matrix, columns=feature_names)
    display_frame = model_frame.copy()
    for column in numeric_columns:
        display_frame[column] = X_test[column].to_numpy()
    return model_frame, display_frame


def compute_shap_values(classifier, model_frame):
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(model_frame)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    base_value = float(np.ravel(explainer.expected_value)[-1])
    return values, base_value


def build_importance_table(values, display_frame):
    rows = []
    for position, column in enumerate(display_frame.columns):
        feature_values = display_frame[column].to_numpy(dtype=float)
        shap_column = values[:, position]
        if np.std(feature_values) == 0 or np.std(shap_column) == 0:
            direction = 0.0
        else:
            direction = float(np.corrcoef(feature_values, shap_column)[0, 1])
        rows.append(
            {
                "feature": column,
                "mean_abs_shap": float(np.abs(shap_column).mean()),
                "value_shap_correlation": direction,
            }
        )
    table = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    SHAP_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SHAP_IMPORTANCE_PATH, index=False)
    return table


def save_summary_plots(values, display_frame):
    SHAP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(values, display_frame, show=False, max_display=MAX_DISPLAY)
    plt.title("SHAP Summary (Test Set)")
    plt.savefig(SHAP_FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    plt.figure()
    shap.summary_plot(values, display_frame, plot_type="bar", show=False, max_display=MAX_DISPLAY)
    plt.title("Mean |SHAP| Feature Importance (Test Set)")
    plt.savefig(SHAP_FIGURES_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def save_dependence_plots(values, display_frame, importance_table):
    for feature in importance_table["feature"].head(TOP_FEATURES_FOR_DEPENDENCE):
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", feature).strip("_")
        plt.figure()
        shap.dependence_plot(
            feature,
            values,
            display_frame,
            interaction_index=None,
            show=False,
        )
        plt.title(f"SHAP Dependence: {feature}")
        plt.savefig(SHAP_FIGURES_DIR / f"shap_dependence_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close("all")


def save_waterfall_plots(pipeline, threshold, X_test, values, base_value, display_frame):
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    selections = {
        "high_risk": int(np.argmax(probabilities)),
        "low_risk": int(np.argmin(probabilities)),
        "near_threshold": int(np.argmin(np.abs(probabilities - threshold))),
    }
    for label, index in selections.items():
        explanation = shap.Explanation(
            values=values[index],
            base_values=base_value,
            data=display_frame.iloc[index].to_numpy(),
            feature_names=list(display_frame.columns),
        )
        plt.figure()
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.savefig(SHAP_FIGURES_DIR / f"shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"Waterfall saved for {label}: customer index {index}, churn probability {probabilities[index]:.3f}")


def write_findings(importance_table):
    lines = ["# SHAP Findings", ""]
    for _, row in importance_table.head(10).iterrows():
        if row["value_shap_correlation"] > 0:
            effect = "higher values or presence increase churn risk"
        elif row["value_shap_correlation"] < 0:
            effect = "higher values or presence decrease churn risk"
        else:
            effect = "no clear direction"
        lines.append(
            f"- {row['feature']}: mean |SHAP| {row['mean_abs_shap']:.4f}; {effect} "
            f"(correlation {row['value_shap_correlation']:.2f})."
        )
    SHAP_FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHAP_FINDINGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def explain_model():
    print("Loading model package...")
    package = joblib.load(MODEL_PATH)
    pipeline = package["pipeline"]
    threshold = package["threshold"]
    numeric_columns = package["numeric_columns"]
    classifier = pipeline.named_steps["classifier"]

    print("Loading held-out test data...")
    X_test, _ = load_test_data()
    model_frame, display_frame = build_display_frame(pipeline, X_test, numeric_columns)

    print("Computing SHAP values...")
    values, base_value = compute_shap_values(classifier, model_frame)
    importance_table = build_importance_table(values, display_frame)

    print("Creating figures...")
    save_summary_plots(values, display_frame)
    save_dependence_plots(values, display_frame, importance_table)
    save_waterfall_plots(pipeline, threshold, X_test, values, base_value, display_frame)

    findings = write_findings(importance_table)
    print(findings)
    print(f"Figures saved to: {SHAP_FIGURES_DIR}")
    print(f"Importance table saved to: {SHAP_IMPORTANCE_PATH}")
    print(f"Findings saved to: {SHAP_FINDINGS_PATH}")


if __name__ == "__main__":
    explain_model()
