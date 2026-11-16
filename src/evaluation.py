"""Shared evaluation metrics and plots, reused by every model in the project."""
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(models: Dict[str, tuple], path: Path) -> None:
    """models: name -> (y_true, y_proba)"""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, (y_true, y_proba) in models.items():
        RocCurveDisplay.from_predictions(y_true, y_proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def metrics_table(models: Dict[str, Dict[str, float]]):
    import pandas as pd

    return pd.DataFrame(models).T.round(4)


def plot_training_history(history, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["auc"], label="train")
    axes[1].plot(history.history["val_auc"], label="validation")
    axes[1].set_title("AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
