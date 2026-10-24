import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from src.config import FIGURES_DIR, RAW_DATA_PATH, REPORTS_DIR
from src.features import TENURE_BINS, TENURE_LABELS

EDA_FIGURES_DIR = FIGURES_DIR / "eda"
FINDINGS_PATH = REPORTS_DIR / "eda_findings.md"
CRAMERS_V_PATH = REPORTS_DIR / "cramers_v.csv"

NO_INTERNET_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

KEY_CATEGORICAL_COLUMNS = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
    "SeniorCitizen",
    "PaperlessBilling",
    "Partner",
    "Dependents",
]

ALL_CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
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
    "tenure_group",
]

CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
CHURN_ORDER = ["No", "Yes"]
CHURN_COLORS = ["#4C9F8E", "#E8825B"]


def load_eda_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.drop(columns="customerID")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    new_customers = (df["tenure"] == 0) & df["TotalCharges"].isna()
    df.loc[new_customers, "TotalCharges"] = 0.0
    for col in NO_INTERNET_COLUMNS:
        df[col] = df[col].replace({"No internet service": "No"})
    df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["churn_flag"] = (df["Churn"] == "Yes").astype(int)
    df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)
    return df


def save_figure(fig, name):
    EDA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(EDA_FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def group_rates(df, column):
    return df.groupby(column, observed=True)["churn_flag"].mean().mul(100)


def plot_churn_distribution(df):
    counts = df["Churn"].value_counts().reindex(CHURN_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(counts.index, counts.values, color=CHURN_COLORS)
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value / len(df):.1%}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    ax.set_ylim(0, counts.max() * 1.12)
    ax.set_xlabel("Churn")
    ax.set_ylabel("Customers")
    ax.set_title("Churn Distribution")
    save_figure(fig, "churn_distribution.png")


def plot_categorical_churn_rates(df):
    overall_rate = df["churn_flag"].mean() * 100
    columns = KEY_CATEGORICAL_COLUMNS
    n_cols = 3
    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, columns):
        rates = group_rates(df, column).sort_values(ascending=False)
        bars = ax.bar(rates.index.astype(str), rates.values, color="#5B8DB8")
        for bar, value in zip(bars, rates.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.axhline(
            overall_rate,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Overall average ({overall_rate:.1f}%)",
        )
        ax.set_ylim(0, max(rates.max(), overall_rate) * 1.35)
        ax.set_title(f"Churn Rate by {column}")
        ax.set_ylabel("Churn (%)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(loc="upper right", frameon=False)
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    save_figure(fig, "categorical_churn_rates.png")


def plot_tenure_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(
        data=df,
        x="tenure",
        hue="Churn",
        hue_order=CHURN_ORDER,
        multiple="stack",
        bins=24,
        palette=CHURN_COLORS,
        ax=axes[0],
    )
    axes[0].set_title("Tenure Distribution by Churn (Counts)")
    axes[0].set_xlabel("Tenure (months)")
    axes[0].set_ylabel("Customers")
    sns.histplot(
        data=df,
        x="tenure",
        hue="Churn",
        hue_order=CHURN_ORDER,
        multiple="fill",
        bins=24,
        palette=CHURN_COLORS,
        ax=axes[1],
    )
    axes[1].set_title("Tenure Distribution by Churn (Share)")
    axes[1].set_xlabel("Tenure (months)")
    axes[1].set_ylabel("Share of customers")
    save_figure(fig, "tenure_distribution.png")


def plot_tenure_group_rates(df):
    rates = group_rates(df, "tenure_group")
    overall_rate = df["churn_flag"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(rates.index.astype(str), rates.values, color="#5B8DB8")
    for bar, value in zip(bars, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.axhline(
        overall_rate,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Overall average ({overall_rate:.1f}%)",
    )
    ax.set_ylim(0, rates.max() * 1.3)
    ax.set_xlabel("Tenure group (months)")
    ax.set_ylabel("Churn (%)")
    ax.set_title("Churn Rate by Tenure Group")
    ax.legend(loc="upper right", frameon=False)
    save_figure(fig, "tenure_group_churn_rate.png")


def plot_contract_tenure_heatmap(df):
    pivot = (
        df.pivot_table(
            values="churn_flag",
            index="Contract",
            columns="tenure_group",
            aggfunc="mean",
            observed=False,
        )
        .mul(100)
        .reindex(CONTRACT_ORDER)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": "Churn (%)"}, ax=ax)
    ax.set_xlabel("Tenure group (months)")
    ax.set_ylabel("Contract")
    ax.set_title("Churn Rate by Contract and Tenure Group")
    save_figure(fig, "contract_tenure_heatmap.png")


def plot_monthly_charges(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.kdeplot(
        data=df,
        x="MonthlyCharges",
        hue="Churn",
        hue_order=CHURN_ORDER,
        palette=CHURN_COLORS,
        common_norm=False,
        fill=True,
        alpha=0.4,
        ax=axes[0],
    )
    axes[0].set_title("Monthly Charges Distribution by Churn")
    axes[0].set_xlabel("Monthly charges")
    sns.boxplot(
        data=df,
        x="InternetService",
        y="MonthlyCharges",
        hue="Churn",
        hue_order=CHURN_ORDER,
        palette=CHURN_COLORS,
        ax=axes[1],
    )
    axes[1].set_title("Monthly Charges by Internet Service and Churn")
    axes[1].set_xlabel("Internet service")
    axes[1].set_ylabel("Monthly charges")
    save_figure(fig, "monthly_charges.png")


def plot_numeric_correlation(df):
    columns = ["tenure", "MonthlyCharges", "TotalCharges", "churn_flag"]
    corr = df[columns].corr().rename(index={"churn_flag": "Churn"}, columns={"churn_flag": "Churn"})
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Numeric Features Correlation")
    save_figure(fig, "numeric_correlation.png")
    return corr


def cramers_v(a, b):
    table = pd.crosstab(a, b)
    k = min(table.shape) - 1
    if k <= 0:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * k)))


def compute_cramers_v(df):
    values = {column: cramers_v(df[column], df["Churn"]) for column in ALL_CATEGORICAL_COLUMNS}
    series = pd.Series(values, name="cramers_v").sort_values(ascending=False)
    CRAMERS_V_PATH.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(CRAMERS_V_PATH, header=True, index_label="feature")

    fig, ax = plt.subplots(figsize=(8, 7))
    ordered = series.sort_values()
    ax.barh(ordered.index, ordered.values, color="#5B8DB8")
    for position, value in enumerate(ordered.values):
        ax.text(value, position, f" {value:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, ordered.max() * 1.15)
    ax.set_xlabel("Cramer's V with Churn")
    ax.set_title("Association of Categorical Features with Churn")
    save_figure(fig, "cramers_v.png")
    return series


def build_findings(df, corr, cramers):
    overall_rate = df["churn_flag"].mean() * 100
    lines = [
        "# EDA Findings",
        "",
        f"- Overall churn rate is {overall_rate:.1f}% ({int(df['churn_flag'].sum())} of {len(df)} customers).",
    ]
    for column in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"]:
        rates = group_rates(df, column).sort_values(ascending=False)
        lines.append(
            f"- {column}: churn is highest for '{rates.index[0]}' ({rates.iloc[0]:.1f}%) "
            f"and lowest for '{rates.index[-1]}' ({rates.iloc[-1]:.1f}%)."
        )
    tenure_rates = group_rates(df, "tenure_group")
    lines.append(
        f"- Customers in the {TENURE_LABELS[0]} month tenure group churn at {tenure_rates.iloc[0]:.1f}%, "
        f"compared with {tenure_rates.iloc[-1]:.1f}% in the {TENURE_LABELS[-1]} month group."
    )
    monthly_means = df.groupby("Churn")["MonthlyCharges"].mean()
    lines.append(
        f"- Average monthly charges are {monthly_means['Yes']:.2f} for churned customers "
        f"versus {monthly_means['No']:.2f} for retained customers."
    )
    lines.append(
        f"- Tenure and TotalCharges are strongly correlated ({corr.loc['tenure', 'TotalCharges']:.2f}), "
        "which indicates multicollinearity for linear models."
    )
    top_features = ", ".join(f"{name} ({value:.2f})" for name, value in cramers.head(5).items())
    lines.append(f"- Strongest categorical associations with churn (Cramer's V): {top_features}.")
    return "\n".join(lines) + "\n"


def run_eda():
    sns.set_theme(style="whitegrid")
    print("Loading data...")
    df = load_eda_data()

    print("Creating figures...")
    plot_churn_distribution(df)
    plot_categorical_churn_rates(df)
    plot_tenure_distribution(df)
    plot_tenure_group_rates(df)
    plot_contract_tenure_heatmap(df)
    plot_monthly_charges(df)
    corr = plot_numeric_correlation(df)
    cramers = compute_cramers_v(df)

    findings = build_findings(df, corr, cramers)
    FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_PATH.write_text(findings, encoding="utf-8")

    print(findings)
    print(f"Figures saved to: {EDA_FIGURES_DIR}")
    print(f"Findings saved to: {FINDINGS_PATH}")
    print(f"Cramer's V table saved to: {CRAMERS_V_PATH}")


if __name__ == "__main__":
    run_eda()
