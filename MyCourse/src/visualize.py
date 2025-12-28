"""Feature visualization utilities."""

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

DEFAULT_FEATURES = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/splits/train_full.npz")
DEFAULT_FIG_DIR = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/figures")


@dataclass
class VizConfig:
    max_points: int = 2000
    random_seed: int = 42
    tsne_perplexity: float = 30.0
    tsne_iter: int = 1000


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


def plot_box_violin_kde(
    df: pd.DataFrame, features: Iterable[str], out_dir: Path
) -> None:
    for feat in features:
        if feat not in df.columns:
            continue
        values = df[feat].to_numpy()
        if np.nanstd(values) == 0.0:
            print(f"[warn] skip {feat}: constant values")
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

        fig.tight_layout()
        fig.savefig(out_dir / f"{feat}_dist.png", dpi=150)
        plt.close(fig)


def plot_corr_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    corr = df.drop(columns=["label"]).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_pca_tsne(df: pd.DataFrame, out_dir: Path, cfg: VizConfig) -> None:
    df_sub = _subset_df(df, cfg)
    X = df_sub.drop(columns=["label"]).to_numpy()
    y = df_sub["label"].to_numpy()
    if X.shape[0] < 2:
        print("[warn] skip PCA/t-SNE: not enough samples")
        return
    std = np.nanstd(X, axis=0)
    keep = std > 1e-8
    if not np.any(keep):
        print("[warn] skip PCA/t-SNE: all features are constant")
        return
    X = X[:, keep]
    if np.unique(X, axis=0).shape[0] < 2:
        print("[warn] skip PCA/t-SNE: only one unique sample")
        return

    pca = PCA(n_components=2, random_state=cfg.random_seed)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, s=30, ax=ax)
    ax.set_title("PCA (2D)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_2d.png", dpi=150)
    plt.close(fig)

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
    fig.tight_layout()
    fig.savefig(out_dir / "tsne_2d.png", dpi=150)
    plt.close(fig)


def default_feature_list() -> List[str]:
    return [
        "interval_mean",
        "interval_std",
        "tempo",
        "ioi_std",
        "f0n_std",
        "mfcc_mean_1",
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Visualize extracted audio features.")
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Path to features.npz saved by features.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--features-list",
        type=str,
        default="",
        help="Comma-separated feature names for box/violin/kde.",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    df = load_features_npz(args.features)
    if args.features_list:
        feats = [f.strip() for f in args.features_list.split(",") if f.strip()]
    else:
        feats = default_feature_list()

    plot_box_violin_kde(df, feats, args.out_dir)
    plot_corr_heatmap(df, args.out_dir)
    plot_pca_tsne(df, args.out_dir, VizConfig())


if __name__ == "__main__":
    main()
