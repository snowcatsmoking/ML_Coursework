"""Notebook-friendly unsupervised analysis with K-Means and clustering metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)

# ---------------------------------------------------------------------
# 1) Paths (keep original .npz path)
# ---------------------------------------------------------------------
DEFAULT_FEATURES = Path(
    "/Users/panmingh/Code/ML_Coursework/MyCourse/results/splits/train_full.npz"
)
DEFAULT_OUT_DIR = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results")


# ---------------------------------------------------------------------
# 2) Config
# ---------------------------------------------------------------------
@dataclass
class UnsupervisedConfig:
    n_clusters: int = 8
    random_seed: int = 42
    max_points: int = 2000


# ---------------------------------------------------------------------
# 3) Data helpers
# ---------------------------------------------------------------------
def load_features_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    labels = data["labels"].astype(str)
    feature_names = data["feature_names"].astype(str)
    return X, labels, feature_names


def subset_data(
    X: np.ndarray, y: np.ndarray, cfg: UnsupervisedConfig
) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= cfg.max_points:
        return X, y
    rng = np.random.default_rng(cfg.random_seed)
    idx = rng.choice(X.shape[0], size=cfg.max_points, replace=False)
    return X[idx], y[idx]


# ---------------------------------------------------------------------
# 4) Metrics + plots (show inline + optional save)
# ---------------------------------------------------------------------
def compute_metrics(X: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
    nmi = normalized_mutual_info_score(y, clusters)
    ari = adjusted_rand_score(y, clusters)
    sil = silhouette_score(X, clusters) if len(np.unique(clusters)) > 1 else 0.0
    return {"nmi": float(nmi), "ari": float(ari), "silhouette": float(sil)}


def _finish_plot(fig: plt.Figure, out_path: Path | None, show: bool) -> None:
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion(
    y: np.ndarray,
    clusters: np.ndarray,
    out_path: Path | None = None,
    show: bool = True,
) -> None:
    labels = np.unique(y)
    conf = confusion_matrix(y, clusters, labels=labels)
    row_sums = conf.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_norm = conf / row_sums
    df = pd.DataFrame(
        conf_norm, index=labels, columns=[f"C{i}" for i in range(conf.shape[1])]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_title("Cluster vs Label Confusion (Row-Normalized)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Label")
    _finish_plot(fig, out_path, show)


def save_metrics(metrics: Dict[str, float], out_dir: Path | None = None) -> None:
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "unsupervised_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# 5) Example "notebook flow" (copy into cells if desired)
# ---------------------------------------------------------------------
# cfg = UnsupervisedConfig(n_clusters=8)
# X, y, _ = load_features_npz(DEFAULT_FEATURES)
# X, y = subset_data(X, y, cfg)
# kmeans = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.random_seed, n_init=10)
# clusters = kmeans.fit_predict(X)
# metrics = compute_metrics(X, y, clusters)
# metrics
# plot_confusion(y, clusters, out_path=None, show=True)
# save_metrics(metrics, DEFAULT_OUT_DIR)
