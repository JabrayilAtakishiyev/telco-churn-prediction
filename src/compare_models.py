import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import clean_data
from src.config import (
    CV_FOLDS,
    FIGURES_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.features import CATEGORICAL_FEATURES, add_features

COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
COMPARISON_FIGURE_PATH = FIGURES_DIR / "model_comparison.png"

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": make_scorer(f1_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "brier": "neg_brier_score",
}


def load_training_data():
    df = add_features(clean_data(pd.read_csv(RAW_DATA_PATH)))
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, y_train


def build_preprocessor(numeric_columns):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_models():
    models = {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    except ImportError:
        print("XGBoost is not installed, skipping.")
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=1,
        )
    except ImportError:
        print("LightGBM is not installed, skipping.")
    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=4,
            random_state=RANDOM_STATE,
            verbose=0,
            thread_count=1,
            allow_writing_files=False,
        )
    except ImportError:
        print("CatBoost is not installed, skipping.")
    return models


def evaluate_model(name, model, X_train, y_train, numeric_columns, cv):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_columns)),
            ("classifier", model),
        ]
    )
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=-1,
    )
    return {
        "Model": name,
        "ROC-AUC": scores["test_roc_auc"].mean(),
        "ROC-AUC Std": scores["test_roc_auc"].std(),
        "PR-AUC": scores["test_pr_auc"].mean(),
        "PR-AUC Std": scores["test_pr_auc"].std(),
        "Train PR-AUC": scores["train_pr_auc"].mean(),
        "PR-AUC Gap": scores["train_pr_auc"].mean() - scores["test_pr_auc"].mean(),
        "F1": scores["test_f1"].mean(),
        "Recall": scores["test_recall"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Brier Score": -scores["test_brier"].mean(),
        "Fit Time (s)": scores["fit_time"].mean(),
    }


def save_comparison_figure(results):
    ordered = results.sort_values("PR-AUC")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        ordered["Model"],
        ordered["PR-AUC"],
        xerr=ordered["PR-AUC Std"],
        color="#5B8DB8",
        capsize=3,
    )
    for position, value in enumerate(ordered["PR-AUC"]):
        ax.text(value + 0.005, position, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Cross-validated PR-AUC (mean, error bar = std)")
    ax.set_title("Model Comparison on the Training Set")
    fig.tight_layout()
    COMPARISON_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARISON_FIGURE_PATH, dpi=150)
    plt.close(fig)


def compare_models():
    print("Loading training data...")
    X_train, y_train = load_training_data()
    numeric_columns = [col for col in X_train.columns if col not in CATEGORICAL_FEATURES]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, model in build_models().items():
        print(f"Evaluating: {name}")
        rows.append(evaluate_model(name, model, X_train, y_train, numeric_columns, cv))

    results = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(COMPARISON_PATH, index=False)
    save_comparison_figure(results)

    print()
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResults saved to: {COMPARISON_PATH}")
    print(f"Figure saved to: {COMPARISON_FIGURE_PATH}")


if __name__ == "__main__":
    compare_models()
