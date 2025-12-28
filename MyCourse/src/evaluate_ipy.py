"""Notebook-friendly evaluation with inline visualizations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    Dataset = object

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

# ---------------------------------------------------------------------
# 1) Paths (keep original .npz/model paths)
# ---------------------------------------------------------------------
DEFAULT_TEST = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/splits/test.npz")
DEFAULT_MODEL = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/models/final_model.pkl")
DEFAULT_OUT = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results")
DEFAULT_MODELS = {"final": DEFAULT_MODEL}
METRIC_ORDER = ("accuracy", "macro_f1", "macro_auc")


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


def load_features_with_paths(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["labels"].astype(str)
    paths = data["paths"].astype(str)
    return X, y, paths


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
    title_suffix: str | None = None,
) -> None:
    labels = sorted(np.unique(y_true).tolist())
    conf = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    title = "Confusion Matrix (Test)"
    if title_suffix:
        title = f"{title} - {title_suffix}"
    ax.set_title(title)
    _finish_plot(fig, out_path, show)


def plot_confusion_normalized(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path | None = None,
    show: bool = True,
    title_suffix: str | None = None,
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
    title = "Confusion Matrix (Row-Normalized)"
    if title_suffix:
        title = f"{title} - {title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    _finish_plot(fig, out_path, show)


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path | None = None,
    show: bool = True,
    title_suffix: str | None = None,
) -> None:
    labels = sorted(np.unique(y_true).tolist())
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(labels):
        y_bin = (y_true == label).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{label} (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    title = "ROC Curves (One-vs-Rest)"
    if title_suffix:
        title = f"{title} - {title_suffix}"
    ax.set_title(title)
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


def save_metrics_report(
    metrics_by_model: Dict[str, Dict[str, float]], out_dir: Path | None = None
) -> None:
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "test_metrics_all.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_by_model, f, ensure_ascii=False, indent=2)


def print_metrics_table(metrics_by_model: Dict[str, Dict[str, float]]) -> None:
    if not metrics_by_model:
        return
    keys = [k for k in METRIC_ORDER if any(k in m for m in metrics_by_model.values())]
    header = ["model", *keys]
    rows = []
    for model_name, metrics in metrics_by_model.items():
        row = [model_name]
        for key in keys:
            value = metrics.get(key, float("nan"))
            if np.isfinite(value):
                row.append(f"{value:.4f}")
            else:
                row.append("nan")
        rows.append(row)
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    line = " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(header))
    sep = "-+-".join("-" * w for w in widths)
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


class MFCCDataset(Dataset):
    def __init__(self, paths: List[str], labels: np.ndarray, sr: int, n_mfcc: int):
        self.paths = paths
        self.labels = labels
        self.sr = sr
        self.n_mfcc = n_mfcc

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        path = self.paths[idx]
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        return mfcc.astype(np.float32), int(self.labels[idx])


def collate_pad(batch: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = [b[0].shape[1] for b in batch]
    max_len = max(lengths)
    n_mfcc = batch[0][0].shape[0]
    padded = np.zeros((len(batch), n_mfcc, max_len), dtype=np.float32)
    labels = np.zeros(len(batch), dtype=np.int64)
    for i, (mfcc, label) in enumerate(batch):
        padded[i, :, : mfcc.shape[1]] = mfcc
        labels[i] = label
    return torch.from_numpy(padded), torch.from_numpy(labels)


def _filter_existing_audio(paths: np.ndarray, labels: np.ndarray) -> Tuple[List[str], np.ndarray]:
    out_paths: List[str] = []
    out_labels: List[str] = []
    for p, y in zip(paths, labels):
        if "::" in p:
            continue
        if Path(p).is_file():
            out_paths.append(p)
            out_labels.append(y)
    return out_paths, np.asarray(out_labels)


def is_torch_model(model: object) -> bool:
    if torch is None or nn is None:
        return False
    return isinstance(model, nn.Module)


def predict_proba_cnn(
    model: nn.Module,
    paths: np.ndarray,
    y_true: np.ndarray,
    batch_size: int = 16,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    if torch is None or librosa is None:
        raise RuntimeError("PyTorch and librosa are required for CNN evaluation.")
    audio_paths, labels = _filter_existing_audio(paths, y_true)
    if len(audio_paths) == 0:
        raise RuntimeError("No valid audio paths found for CNN evaluation.")
    dataset = MFCCDataset(audio_paths, labels, sr=22050, n_mfcc=13)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)
    model = model.to(device)
    model.eval()
    all_probs: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_true.append(yb.numpy())
    return np.concatenate(all_true), np.vstack(all_probs)


def evaluate_models(
    model_paths: Dict[str, Path],
    test_path: Path = DEFAULT_TEST,
    out_dir: Path | None = DEFAULT_OUT,
    show: bool = True,
) -> Dict[str, Dict[str, float]]:
    X_test, y_test, test_paths = load_features_with_paths(test_path)
    X_test = sanitize_features(X_test)
    metrics_by_model: Dict[str, Dict[str, float]] = {}
    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            print(f"[warn] skip {model_name}: missing {model_path}")
            continue
        model, label_encoder = load_model(model_path)
        y_true = label_encoder.transform(y_test) if label_encoder is not None else y_test
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            y_pred = np.argmax(y_proba, axis=1)
        elif is_torch_model(model):
            y_true, y_proba = predict_proba_cnn(model, test_paths, y_true)
            y_pred = np.argmax(y_proba, axis=1)
        else:
            print(f"[warn] skip {model_name}: unsupported model type")
            continue
        metrics = compute_metrics(y_true, y_pred, y_proba)
        metrics_by_model[model_name] = metrics
        fig_dir = out_dir / "figures" / model_name if out_dir is not None else None
        if fig_dir is not None:
            fig_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(
            y_true,
            y_pred,
            out_path=(fig_dir / "test_confusion.png") if fig_dir is not None else None,
            show=show,
            title_suffix=model_name,
        )
        plot_confusion_normalized(
            y_true,
            y_pred,
            out_path=(fig_dir / "test_confusion_norm.png") if fig_dir is not None else None,
            show=show,
            title_suffix=model_name,
        )
        plot_roc_curves(
            y_true,
            y_proba,
            out_path=(fig_dir / "test_roc_curves.png") if fig_dir is not None else None,
            show=show,
            title_suffix=model_name,
        )
    print_metrics_table(metrics_by_model)
    if out_dir is not None:
        save_metrics_report(metrics_by_model, out_dir)
    return metrics_by_model


# ---------------------------------------------------------------------
# 5) Example "notebook flow" (copy into cells if desired)
# ---------------------------------------------------------------------
# models = {
#     "knn": Path("/Users/panmingh/Code/ML_Coursework/MyCourse/models/knn_model.pkl"),
#     "rf": Path("/Users/panmingh/Code/ML_Coursework/MyCourse/models/rf_model.pkl"),
#     "mlp": Path("/Users/panmingh/Code/ML_Coursework/MyCourse/models/mlp_model.pkl"),
#     "cnn": Path("/Users/panmingh/Code/ML_Coursework/MyCourse/models/cnn_model.pkl"),
# }
# evaluate_models(models, test_path=DEFAULT_TEST, out_dir=DEFAULT_OUT, show=True)
