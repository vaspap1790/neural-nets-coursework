"""Question 3: MLP with TensorFlow/Keras - tuning, training and SHAP."""
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline_model import train_logistic_regression
from src.data_utils import load_raw_dataset
from src.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
)
from src.explainability import compute_shap_values, plot_shap_summary, top_shap_features
from src.nn_model import build_mlp, class_weights
from src.preprocessing import build_feature_target, split_and_scale
from src.tuning import run_search

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
TABLE_DIR = Path(__file__).resolve().parents[1] / "outputs" / "tables"

tf.random.set_seed(42)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset()
    X, y = build_feature_target(df)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)
    input_dim = X_train.shape[1]

    print("Running KerasTuner random search (10 trials)...")
    _, best_hp = run_search(X_train.values, y_train.values, input_dim, max_trials=10, epochs=20)
    print("Best hyperparameters:", best_hp.values)

    model = build_mlp(
        input_dim=input_dim,
        units_1=best_hp.get("units_1"),
        units_2=best_hp.get("units_2"),
        dropout=best_hp.get("dropout"),
        l2=best_hp.get("l2"),
        learning_rate=best_hp.get("learning_rate"),
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=8, restore_best_weights=True
    )
    history = model.fit(
        X_train.values,
        y_train.values,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        class_weight=class_weights(y_train),
        callbacks=[early_stop],
        verbose=0,
    )
    print(f"Stopped after {len(history.history['loss'])} epochs (early stopping).")
    plot_training_history(history, FIG_DIR / "fig7_nn_training_history.png")

    y_proba_nn = model.predict(X_test.values, verbose=0).ravel()
    y_pred_nn = (y_proba_nn >= 0.5).astype(int)
    metrics_nn = compute_metrics(y_test, y_pred_nn, y_proba_nn)
    print("\nNeural network metrics:")
    for k, v in metrics_nn.items():
        print(f"  {k}: {v:.4f}")

    plot_confusion_matrix(
        y_test, y_pred_nn, "Neural Network - Confusion Matrix",
        FIG_DIR / "fig8_nn_confusion_matrix.png",
    )

    # re-fit the LR baseline here too, purely to overlay both ROC curves
    lr_model = train_logistic_regression(X_train, y_train)
    y_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    plot_roc_curves(
        {"Logistic Regression": (y_test, y_proba_lr), "Neural Network": (y_test, y_proba_nn)},
        FIG_DIR / "fig9_roc_comparison.png",
    )

    print("\nComputing SHAP values (this can take a minute)...")
    X_explain = X_test.sample(min(150, len(X_test)), random_state=42)
    shap_values = compute_shap_values(model, X_train, X_explain)
    plot_shap_summary(shap_values, X.columns.tolist(), FIG_DIR / "fig10_shap_summary.png")
    top_feats = top_shap_features(shap_values, X.columns.tolist(), n=8)
    top_feats.to_csv(TABLE_DIR / "nn_top_shap_features.csv")
    print("\nTop |SHAP| features:")
    print(top_feats)


if __name__ == "__main__":
    main()
