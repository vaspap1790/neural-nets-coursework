"""Question 1: Dataset exploration, cleaning, encoding and scaling."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import ID_COL, NUMERIC_COLS, TARGET_COL, load_raw_dataset
from src.preprocessing import build_feature_target, clean_total_charges, split_and_scale

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
TABLE_DIR = Path(__file__).resolve().parents[1] / "outputs" / "tables"
sns.set_theme(style="whitegrid")


def describe_dataset(df: pd.DataFrame) -> None:
    print(f"Shape: {df.shape}")
    print(f"Duplicate customerID rows: {df[ID_COL].duplicated().sum()}")
    print(f"Blank TotalCharges: {(df['TotalCharges'].str.strip() == '').sum()}")

    churn_counts = df[TARGET_COL].value_counts()
    churn_pct = df[TARGET_COL].value_counts(normalize=True) * 100
    print("\nChurn distribution:")
    print(pd.DataFrame({"count": churn_counts, "pct": churn_pct.round(2)}))

    df_clean = clean_total_charges(df)
    numeric_summary = df_clean[NUMERIC_COLS].describe().T.round(2)
    numeric_summary.to_csv(TABLE_DIR / "numeric_summary.csv")
    print("\nNumeric feature summary:")
    print(numeric_summary)


def plot_churn_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=TARGET_COL, data=df, ax=ax, hue=TARGET_COL, palette="Blues", legend=False)
    ax.set_title("Class distribution of the target variable (Churn)")
    for container in ax.containers:
        ax.bar_label(container)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_churn_distribution.png", dpi=150)
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    df = clean_total_charges(df)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.histplot(data=df, x=col, hue=TARGET_COL, kde=True, ax=ax, bins=30, palette="Blues")
        ax.set_title(col)
    fig.suptitle("Distribution of numeric features by churn status")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_numeric_distributions.png", dpi=150)
    plt.close(fig)


def plot_churn_rate_by_category(df: pd.DataFrame, col: str, fname: str) -> None:
    rate = df.groupby(col)[TARGET_COL].apply(lambda s: (s == "Yes").mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    rate.plot(kind="barh", ax=ax, color="#2b6cb0")
    ax.set_xlabel("Churn rate (%)")
    ax.set_title(f"Churn rate by {col}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset()
    describe_dataset(df)

    plot_churn_distribution(df)
    plot_numeric_distributions(df)
    plot_churn_rate_by_category(df, "Contract", "fig3_churn_rate_by_contract.png")
    plot_churn_rate_by_category(df, "InternetService", "fig4_churn_rate_by_internet_service.png")

    X, y = build_feature_target(df)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)

    print(f"\nFeature matrix after encoding: {X.shape}")
    print(f"Train/test split: {X_train.shape[0]} / {X_test.shape[0]}")
    print(f"Train churn rate: {y_train.mean():.4f}, Test churn rate: {y_test.mean():.4f}")


if __name__ == "__main__":
    main()
