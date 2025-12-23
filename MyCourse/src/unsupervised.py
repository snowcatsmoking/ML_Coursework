"""Unsupervised analysis with K-Means and clustering metrics."""

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
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score, silhouette_score

DEFAULT_FEATURES = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/splits/train_full.npz")
DEFAULT_OUT_DIR = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results")


@dataclass
class UnsupervisedConfig:
    n_clusters: int = 8
    random_seed: int = 42
    max_points: int = 2000


def load_features_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    labels = data["labels"].astype(str)
    feature_names = data["feature_names"].astype(str)
    return X, labels, feature_names


def subset_data(X: np.ndarray, y: np.ndarray, cfg: UnsupervisedConfig) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= cfg.max_points:
        return X, y
    rng = np.random.default_rng(cfg.random_seed)
    idx = rng.choice(X.shape[0], size=cfg.max_points, replace=False)
    return X[idx], y[idx]


def compute_metrics(X: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
    nmi = normalized_mutual_info_score(y, clusters)
    ari = adjusted_rand_score(y, clusters)
    sil = silhouette_score(X, clusters) if len(np.unique(clusters)) > 1 else 0.0
    return {"nmi": float(nmi), "ari": float(ari), "silhouette": float(sil)}


def plot_confusion(y: np.ndarray, clusters: np.ndarray, out_path: Path) -> None:
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
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Unsupervised clustering analysis (K-Means).")
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Path to features.npz (from features.py).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to save metrics and figures.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of clusters.")
    args = parser.parse_args()

    cfg = UnsupervisedConfig(n_clusters=args.k)
    X, y, _ = load_features_npz(args.features)
    X, y = subset_data(X, y, cfg)

    kmeans = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.random_seed, n_init=10)
    clusters = kmeans.fit_predict(X)
    metrics = compute_metrics(X, y, clusters)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "unsupervised_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(y, clusters, fig_dir / "cluster_label_confusion.png")


if __name__ == "__main__":
    main()
