"""Cleaning, encoding and scaling for the Telco Customer Churn dataset."""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_utils import (
    BINARY_COLS,
    ID_COL,
    MULTI_CATEGORY_COLS,
    NUMERIC_COLS,
    TARGET_COL,
)

BINARY_MAPS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges is stored as text and is blank for 11 customers with
    tenure == 0 (brand-new accounts that have not been billed yet). Those
    blanks are coerced to 0.0 rather than dropped or imputed with a mean,
    since 0 is the factually correct total charge for a customer who has
    not completed a billing cycle."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BINARY_COLS:
        df[col] = df[col].map(BINARY_MAPS[col])
    return df


def one_hot_encode_multi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Nominal, no natural order -> one-hot rather than label/ordinal
    encoding, so neither the logistic regression nor the MLP infer a false
    ranking between e.g. contract types or payment methods."""
    return pd.get_dummies(df, columns=MULTI_CATEGORY_COLS, drop_first=True)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].map({"No": 0, "Yes": 1})
    return df


def build_feature_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = clean_total_charges(df)
    df = encode_binary_columns(df)
    df = encode_target(df)
    df = one_hot_encode_multi_category(df)
    df = df.drop(columns=[ID_COL])

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL]).astype(np.float64)
    return X, y


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified split (preserves the 26.5%/73.5% churn ratio in both
    folds) followed by standardisation. The scaler is fit on the training
    fold only and merely applied to the test fold, so no information about
    the held-out data leaks into training."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    return X_train, X_test, y_train, y_test, scaler
