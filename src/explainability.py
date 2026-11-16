"""SHAP-based feature importance for the Keras MLP."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap


def compute_shap_values(model, X_background, X_explain):
    """Independent masker + permutation-based Explainer: model-agnostic,
    so it works directly on a Keras model's predict function without
    needing a TensorFlow-specific backend, at the cost of being slower
    than DeepExplainer. Sample sizes are kept small (tens-hundreds of
    rows) to keep runtime reasonable for a coursework pipeline."""
    background = shap.sample(X_background, min(100, len(X_background)), random_state=42)
    masker = shap.maskers.Independent(background)
    explainer = shap.Explainer(lambda x: model.predict(x, verbose=0).ravel(), masker)
    return explainer(X_explain)


def plot_shap_summary(shap_values, feature_names, path: Path, max_display: int = 12) -> None:
    plt.figure(figsize=(8, 5.5))
    shap.summary_plot(
        shap_values.values,
        shap_values.data,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    plt.xlabel("mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_shap_features(shap_values, feature_names, n: int = 8):
    import pandas as pd

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    ranked = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    return ranked.head(n).to_frame("mean_|SHAP|")
