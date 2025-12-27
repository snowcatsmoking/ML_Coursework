"""Notebook-friendly evaluation with inline visualizations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

try:
    from joblib import load
except Exception:  # pragma: no cover
    load = None

# ---------------------------------------------------------------------
# 1) Paths (keep original .npz/model paths)
# ---------------------------------------------------------------------
DEFAULT_TEST = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/splits/test.npz")
DEFAULT_MODEL = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/models/final_model.pkl")
DEFAULT_OUT = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results")


# ---------------------------------------------------------------------
# 2) Config
# ---------------------------------------------------------------------
@dataclass
class EvalConfig:
    out_dir: Path = DEFAULT_OUT


# ---------------------------------------------------------------------
# 3) Loading + preprocessing
# ---------------------------------------------------------------------
def load_features(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["labels"].astype(str)
    return X, y


def sanitize_features(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    low = np.percentile(X, 1, axis=0)
    high = np.percentile(X, 99, axis=0)
    return np.clip(X, low, high)


def load_model(path: Path):
    if load is None:
        raise RuntimeError("joblib is required to load the model.")
    payload = load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("label_encoder")
    return payload, None


# ---------------------------------------------------------------------
# 4) Metrics + plots (show inline + optional save)
# ---------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    labels = sorted(np.unique(y_true).tolist())
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        metrics["macro_auc"] = float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
        )
    except Exception:
        metrics["macro_auc"] = float("nan")
    return metrics


def _finish_plot(fig: plt.Figure, out_path: Path | None, show: bool) -> None:
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path | None = None,
    show: bool = True,
) -> None:
    labels = sorted(np.unique(y_true).tolist())
    conf = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix (Test)")
    _finish_plot(fig, out_path, show)


def plot_confusion_normalized(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path | None = None,
    show: bool = True,
) -> None:
    labels = sorted(np.unique(y_true).tolist())
    conf = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = conf.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_norm = conf / row_sums
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        conf_norm,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Confusion Matrix (Row-Normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    _finish_plot(fig, out_path, show)


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path | None = None,
    show: bool = True,
) -> None:
    labels = sorted(np.unique(y_true).tolist())
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(labels):
        y_bin = (y_true == label).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{label} (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right")
    _finish_plot(fig, out_path, show)


def save_metrics(metrics: Dict[str, float], out_dir: Path | None = None) -> None:
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# 5) Example "notebook flow" (copy into cells if desired)
# ---------------------------------------------------------------------
# X_test, y_test = load_features(DEFAULT_TEST)
# X_test = sanitize_features(X_test)
# model, label_encoder = load_model(DEFAULT_MODEL)
# y_true = label_encoder.transform(y_test) if label_encoder is not None else y_test
# y_proba = model.predict_proba(X_test)
# y_pred = np.argmax(y_proba, axis=1)
# metrics = compute_metrics(y_true, y_pred, y_proba)
# metrics
# plot_confusion_matrix(y_true, y_pred, out_path=None, show=True)
# plot_confusion_normalized(y_true, y_pred, out_path=None, show=True)
# plot_roc_curves(y_true, y_proba, out_path=None, show=True)
# save_metrics(metrics, DEFAULT_OUT)
