"""Question 4: benchmark Random Forest / XGBoost against the LR baseline
and the neural network, to support the business-implications discussion.

The Logistic Regression and Neural Network metrics are the same ones
already reported in Questions 2-3 (LR is deterministic given a fixed
random_state and reproduces exactly; the NN has run-to-run variance from
weight initialisation and is therefore reported once, not re-derived here,
so the report does not state two different numbers for the same model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline_model import train_logistic_regression
from src.benchmarking import train_random_forest, train_xgboost
from src.data_utils import load_raw_dataset
from src.evaluation import compute_metrics, metrics_table, plot_roc_curves
from src.preprocessing import build_feature_target, split_and_scale

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
TABLE_DIR = Path(__file__).resolve().parents[1] / "outputs" / "tables"

# as reported in Questions 2 and 3
REPORTED_METRICS = {
    "Logistic Regression": {
        "accuracy": 0.7381, "precision": 0.5043, "recall": 0.7834,
        "f1_score": 0.6136, "roc_auc": 0.8417,
    },
    "Neural Network": {
        "accuracy": 0.7189, "precision": 0.4833, "recall": 0.8503,
        "f1_score": 0.6163, "roc_auc": 0.8355,
    },
}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset()
    X, y = build_feature_target(df)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)

    results = dict(REPORTED_METRICS)
    probas = {}

    lr = train_logistic_regression(X_train, y_train)
    probas["Logistic Regression"] = lr.predict_proba(X_test)[:, 1]

    rf = train_random_forest(X_train, y_train)
    p_rf = rf.predict_proba(X_test)[:, 1]
    results["Random Forest"] = compute_metrics(y_test, (p_rf >= 0.5).astype(int), p_rf)
    probas["Random Forest"] = p_rf

    xgb = train_xgboost(X_train, y_train)
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    results["XGBoost"] = compute_metrics(y_test, (p_xgb >= 0.5).astype(int), p_xgb)
    probas["XGBoost"] = p_xgb

    table = metrics_table(results)
    table.to_csv(TABLE_DIR / "all_models_comparison.csv")
    print(table)

    plot_roc_curves(
        {name: (y_test, proba) for name, proba in probas.items()},
        FIG_DIR / "fig11_roc_all_models.png",
    )


if __name__ == "__main__":
    main()
