"""Notebook-friendly feature visualization utilities.

This mirrors `visualize.py` but shows plots inline in a Jupyter notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------
# 1) Paths (keep original .npz path)
# ---------------------------------------------------------------------
DEFAULT_FEATURES = Path(
    "/Users/panmingh/Code/ML_Coursework/MyCourse/results/splits/train_full.npz"
)
DEFAULT_FIG_DIR = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/figures")


# ---------------------------------------------------------------------
# 2) Config
# ---------------------------------------------------------------------
@dataclass
class VizConfig:
    max_points: int = 2000
    random_seed: int = 42
    tsne_perplexity: float = 30.0
    tsne_iter: int = 1000


# ---------------------------------------------------------------------
# 3) Data loading
# ---------------------------------------------------------------------
def load_features_npz(path: Path) -> pd.DataFrame:
    data = np.load(path, allow_pickle=True)
    features = data["X"]
    labels = data["labels"].astype(str)
    names = data["feature_names"].astype(str).tolist()
    df = pd.DataFrame(features, columns=names)
    df["label"] = labels
    return df


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _subset_df(df: pd.DataFrame, cfg: VizConfig) -> pd.DataFrame:
    if len(df) <= cfg.max_points:
        return df
    return df.sample(n=cfg.max_points, random_state=cfg.random_seed)


# ---------------------------------------------------------------------
# 4) Plotting helpers (show inline + optional save)
# ---------------------------------------------------------------------
def _finish_plot(fig: plt.Figure, out_path: Path | None, show: bool) -> None:
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_box_violin_kde(
    df: pd.DataFrame,
    features: Iterable[str],
    out_dir: Path | None = None,
    show: bool = True,
) -> None:
    if out_dir is not None:
        ensure_dir(out_dir)
    for feat in features:
        if feat not in df.columns:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        sns.boxplot(data=df, x="label", y=feat, ax=axes[0])
        axes[0].set_title(f"Box Plot - {feat}")
        axes[0].tick_params(axis="x", rotation=45)

        sns.violinplot(data=df, x="label", y=feat, ax=axes[1], cut=0)
        axes[1].set_title(f"Violin Plot - {feat}")
        axes[1].tick_params(axis="x", rotation=45)

        sns.kdeplot(
            data=df,
            x=feat,
            hue="label",
            ax=axes[2],
            fill=False,
            common_norm=False,
            warn_singular=False,
        )
        axes[2].set_title(f"KDE - {feat}")

        out_path = out_dir / f"{feat}_dist.png" if out_dir is not None else None
        _finish_plot(fig, out_path, show)


def plot_corr_heatmap(
    df: pd.DataFrame, out_dir: Path | None = None, show: bool = True
) -> None:
    if out_dir is not None:
        ensure_dir(out_dir)
    corr = df.drop(columns=["label"]).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    out_path = out_dir / "correlation_heatmap.png" if out_dir is not None else None
    _finish_plot(fig, out_path, show)


def plot_pca_tsne(
    df: pd.DataFrame,
    out_dir: Path | None = None,
    cfg: VizConfig | None = None,
    show: bool = True,
) -> None:
    if out_dir is not None:
        ensure_dir(out_dir)
    cfg = cfg or VizConfig()
    df_sub = _subset_df(df, cfg)
    X = df_sub.drop(columns=["label"]).to_numpy()
    y = df_sub["label"].to_numpy()

    pca = PCA(n_components=2, random_state=cfg.random_seed)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, s=30, ax=ax)
    ax.set_title("PCA (2D)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    out_path = out_dir / "pca_2d.png" if out_dir is not None else None
    _finish_plot(fig, out_path, show)

    tsne_kwargs = dict(
        n_components=2,
        perplexity=cfg.tsne_perplexity,
        random_state=cfg.random_seed,
        init="pca",
        learning_rate="auto",
    )
    try:
        tsne = TSNE(max_iter=cfg.tsne_iter, **tsne_kwargs)
    except TypeError:
        tsne = TSNE(n_iter=cfg.tsne_iter, **tsne_kwargs)
    X_tsne = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, s=30, ax=ax)
    ax.set_title("t-SNE (2D)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    out_path = out_dir / "tsne_2d.png" if out_dir is not None else None
    _finish_plot(fig, out_path, show)


def default_feature_list() -> List[str]:
    return [
        "interval_mean",
        "interval_std",
        "tempo",
        "ioi_std",
        "f0n_std",
        "mfcc_mean_1",
    ]


# ---------------------------------------------------------------------
# 5) Example "notebook flow" (copy into cells if desired)
# ---------------------------------------------------------------------
# df = load_features_npz(DEFAULT_FEATURES)
# feats = default_feature_list()
# plot_box_violin_kde(df, feats, out_dir=None, show=True)
# plot_corr_heatmap(df, out_dir=None, show=True)
# plot_pca_tsne(df, out_dir=None, show=True)
