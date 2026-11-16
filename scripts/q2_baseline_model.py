"""Question 2: Logistic Regression baseline and its evaluation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline_model import top_coefficients, train_logistic_regression
from src.data_utils import load_raw_dataset
from src.evaluation import compute_metrics, plot_confusion_matrix, plot_roc_curves
from src.preprocessing import build_feature_target, split_and_scale

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
TABLE_DIR = Path(__file__).resolve().parents[1] / "outputs" / "tables"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset()
    X, y = build_feature_target(df)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)

    model = train_logistic_regression(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print("Logistic Regression baseline metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    plot_confusion_matrix(
        y_test, y_pred, "Logistic Regression - Confusion Matrix",
        FIG_DIR / "fig5_lr_confusion_matrix.png",
    )
    plot_roc_curves({"Logistic Regression": (y_test, y_proba)}, FIG_DIR / "fig6_lr_roc_curve.png")

    coefs = top_coefficients(model, X.columns, n=8)
    coefs.to_csv(TABLE_DIR / "lr_top_coefficients.csv")
    print("\nTop |coefficient| features:")
    print(coefs)


if __name__ == "__main__":
    main()
