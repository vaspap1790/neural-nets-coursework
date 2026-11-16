"""Loading and schema constants for the IBM Telco Customer Churn dataset."""
from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"

ID_COL = "customerID"
TARGET_COL = "Churn"

BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

MULTI_CATEGORY_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

PASSTHROUGH_COLS = ["SeniorCitizen"]  # already binary 0/1 in the source file


def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
