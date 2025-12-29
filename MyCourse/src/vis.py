"""Notebook-friendly feature visualization utilities with EDA analysis.

This mirrors `visualize.py` but shows plots inline in a Jupyter notebook.
增强版：添加完整的EDA分析功能，输出数值结果和统计表格。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats

# ---------------------------------------------------------------------
# 1) Paths (keep original .npz path)
# ---------------------------------------------------------------------
DEFAULT_FEATURES = Path(
    "C:\\Users\\ASUS\\Desktop\\ML_Course\\ML_Coursework\\MyCourse\\results\\splits\\train_full.npz"
)
DEFAULT_FIG_DIR = Path("C:\\Users\\ASUS\\Desktop\\ML_Course\\ML_Coursework\\MyCourse\\results\\figures")


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
# 4) EDA Analysis Functions (新增)
# ---------------------------------------------------------------------

def compute_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有特征的基础统计量，返回汇总表格。
    
    Returns:
        DataFrame with columns: feature, count, missing, missing_pct, 
        mean, std, min, 25%, 50%, 75%, max, skewness, kurtosis
    """
    # 获取数值列（排除label）
    num_cols = [c for c in df.columns if c != "label"]
    
    records = []
    for col in num_cols:
        vals = df[col]
        missing = vals.isna().sum()
        valid = vals.dropna()
        
        records.append({
            "feature": col,
            "count": len(valid),
            "missing": missing,
            "missing_pct": f"{100 * missing / len(df):.2f}%",
            "mean": valid.mean(),
            "std": valid.std(),
            "min": valid.min(),
            "25%": valid.quantile(0.25),
            "50%": valid.quantile(0.50),
            "75%": valid.quantile(0.75),
            "max": valid.max(),
            "skewness": stats.skew(valid) if len(valid) > 2 else np.nan,
            "kurtosis": stats.kurtosis(valid) if len(valid) > 2 else np.nan,
        })
    
    return pd.DataFrame(records)


def compute_class_stats(df: pd.DataFrame) -> pd.DataFrame:
    """按类别计算每个特征的统计量。
    
    Returns:
        DataFrame with multi-level columns: (feature, statistic) for each class
    """
    num_cols = [c for c in df.columns if c != "label"]
    labels = df["label"].unique()
    
    records = []
    for col in num_cols:
        row = {"feature": col}
        for lab in sorted(labels):
            vals = df[df["label"] == lab][col].dropna()
            row[f"{lab}_mean"] = vals.mean()
            row[f"{lab}_std"] = vals.std()
            row[f"{lab}_count"] = len(vals)
        records.append(row)
    
    return pd.DataFrame(records)


def compute_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """计算特征的区分能力指标。
    
    包含:
    - variance: 整体方差
    - between_class_var: 类间方差（越大越有区分力）
    - within_class_var: 类内方差
    - fisher_ratio: Fisher判别比 = 类间方差/类内方差
    - anova_f: ANOVA F统计量
    - anova_p: ANOVA p值
    """
    num_cols = [c for c in df.columns if c != "label"]
    labels = df["label"].unique()
    
    records = []
    for col in num_cols:
        vals = df[col].dropna()
        overall_mean = vals.mean()
        overall_var = vals.var()
        
        # 计算类间方差和类内方差
        between_var = 0.0
        within_var = 0.0
        groups = []
        
        for lab in labels:
            group_vals = df[df["label"] == lab][col].dropna()
            if len(group_vals) == 0:
                continue
            groups.append(group_vals.values)
            group_mean = group_vals.mean()
            group_var = group_vals.var()
            n = len(group_vals)
            
            between_var += n * (group_mean - overall_mean) ** 2
            within_var += (n - 1) * group_var if n > 1 else 0
        
        n_total = len(vals)
        k = len(labels)
        between_var /= (k - 1) if k > 1 else 1
        within_var /= (n_total - k) if n_total > k else 1
        
        fisher = between_var / within_var if within_var > 1e-10 else np.inf
        
        # ANOVA检验
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            try:
                f_stat, p_val = stats.f_oneway(*groups)
            except:
                f_stat, p_val = np.nan, np.nan
        else:
            f_stat, p_val = np.nan, np.nan
        
        records.append({
            "feature": col,
            "variance": overall_var,
            "between_class_var": between_var,
            "within_class_var": within_var,
            "fisher_ratio": fisher,
            "anova_F": f_stat,
            "anova_p": p_val,
            "significant": "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        })
    
    result = pd.DataFrame(records)
    # 按Fisher ratio排序，区分能力强的在前
    return result.sort_values("fisher_ratio", ascending=False).reset_index(drop=True)


def compute_correlation_pairs(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """找出高相关的特征对。
    
    Args:
        threshold: 相关系数阈值，默认0.7
    
    Returns:
        DataFrame with columns: feature1, feature2, correlation, abs_corr
    """
    num_df = df.drop(columns=["label"])
    corr = num_df.corr()
    
    pairs = []
    cols = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            r = corr.loc[c1, c2]
            if abs(r) >= threshold:
                pairs.append({
                    "feature1": c1,
                    "feature2": c2,
                    "correlation": r,
                    "abs_corr": abs(r)
                })
    
    result = pd.DataFrame(pairs)
    if len(result) > 0:
        result = result.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return result


def detect_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
    """检测每个特征的异常值数量。
    
    Args:
        method: 'iqr' (四分位距) 或 'zscore' (标准差)
        threshold: IQR倍数或Z-score阈值
    
    Returns:
        DataFrame with columns: feature, n_outliers, outlier_pct, range_info
    """
    num_cols = [c for c in df.columns if c != "label"]
    
    records = []
    for col in num_cols:
        vals = df[col].dropna()
        
        if method == "iqr":
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
            outliers = ((vals < lower) | (vals > upper)).sum()
            range_info = f"[{lower:.3g}, {upper:.3g}]"
        else:  # zscore
            z = np.abs(stats.zscore(vals))
            outliers = (z > threshold).sum()
            range_info = f"|z| > {threshold}"
        
        records.append({
            "feature": col,
            "n_outliers": outliers,
            "outlier_pct": f"{100 * outliers / len(vals):.2f}%",
            "valid_range": range_info
        })
    
    return pd.DataFrame(records).sort_values("n_outliers", ascending=False).reset_index(drop=True)


def compute_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """计算标签分布。"""
    counts = df["label"].value_counts()
    total = len(df)
    
    records = []
    for lab, cnt in counts.items():
        records.append({
            "label": lab,
            "count": cnt,
            "percentage": f"{100 * cnt / total:.2f}%"
        })
    
    return pd.DataFrame(records)


def run_full_eda(df: pd.DataFrame, verbose: int = 1) -> Dict[str, pd.DataFrame]:
    """运行完整的EDA分析，返回所有结果表格。
    
    Args:
        df: 输入数据
        verbose: 输出详细程度 (0=静默, 1=精简, 2=详细)
    
    Returns:
        Dict containing all analysis results
    """
    results = {}
    
    # 计算所有结果
    results["label_dist"] = compute_label_distribution(df)
    results["basic_stats"] = compute_basic_stats(df)
    results["feature_importance"] = compute_feature_importance(df)
    results["class_stats"] = compute_class_stats(df)
    results["high_correlations"] = compute_correlation_pairs(df, threshold=0.7)
    results["outliers"] = detect_outliers(df, method="iqr", threshold=1.5)
    
    if verbose == 0:
        return results
    
    # === 核心输出 ===
    n_samples = len(df)
    n_features = len([c for c in df.columns if c != "label"])
    n_classes = df["label"].nunique()
    
    print("=" * 65)
    print("  EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 65)
    
    # 1. 数据概览（一行搞定）
    print(f"\n[Dataset] {n_samples} samples × {n_features} features × {n_classes} classes")
    print(f"[Labels]  {dict(zip(results['label_dist']['label'], results['label_dist']['count']))}")
    
    # 2. 特征区分能力（核心！按Fisher ratio排序）
    print("\n" + "-" * 65)
    print("  FEATURE RANKING (by discriminative power)")
    print("-" * 65)
    feat_imp = results["feature_importance"]
    # 精简显示：只显示关键列
    display_cols = ["feature", "fisher_ratio", "anova_p", "significant"]
    display_df = feat_imp[display_cols].copy()
    display_df["fisher_ratio"] = display_df["fisher_ratio"].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) and not np.isinf(x) else "inf")
    display_df["anova_p"] = display_df["anova_p"].apply(
        lambda x: f"{x:.2e}" if pd.notna(x) else "NaN")
    print(display_df.to_string(index=False))
    
    # 3. 高相关特征对（只在有结果时显示）
    high_corr = results["high_correlations"]
    if len(high_corr) > 0:
        print("\n" + "-" * 65)
        print(f"  REDUNDANT FEATURES (|correlation| >= 0.7): {len(high_corr)} pairs")
        print("-" * 65)
        display_corr = high_corr[["feature1", "feature2", "correlation"]].head(10)
        display_corr["correlation"] = display_corr["correlation"].apply(lambda x: f"{x:.3f}")
        print(display_corr.to_string(index=False))
        if len(high_corr) > 10:
            print(f"  ... and {len(high_corr) - 10} more pairs")
    
    # 4. 数据质量警告（只在有问题时显示）
    basic_stats = results["basic_stats"]
    total_missing = basic_stats["missing"].sum()
    outliers = results["outliers"]
    high_outlier_feats = outliers[outliers["n_outliers"] > n_samples * 0.1]  # >10%异常值
    
    if total_missing > 0 or len(high_outlier_feats) > 0:
        print("\n" + "-" * 65)
        print("  DATA QUALITY WARNINGS")
        print("-" * 65)
        if total_missing > 0:
            missing_feats = basic_stats[basic_stats["missing"] > 0][["feature", "missing", "missing_pct"]]
            print(f"[Missing] {total_missing} values in {len(missing_feats)} features")
            if verbose >= 2:
                print(missing_feats.to_string(index=False))
        if len(high_outlier_feats) > 0:
            print(f"[Outliers] {len(high_outlier_feats)} features with >10% outliers:")
            for _, row in high_outlier_feats.head(5).iterrows():
                print(f"    {row['feature']}: {row['n_outliers']} ({row['outlier_pct']})")
    
    # 5. 结论摘要
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    top3 = feat_imp.head(3)
    weak = feat_imp[feat_imp["anova_p"] >= 0.05]
    print(f"  ✓ Top discriminative: {', '.join(top3['feature'].tolist())}")
    if len(weak) > 0:
        print(f"  ✗ Weak features (p>=0.05): {len(weak)} features may be removed")
    if len(high_corr) > 0:
        print(f"  ✗ Redundant pairs: {len(high_corr)} (consider removing one from each pair)")
    print("=" * 65)
    
    return results


def eda_to_markdown(results: Dict[str, pd.DataFrame], top_n: int = 10) -> str:
    """将EDA核心结果转换为Markdown格式（精简版）。"""
    md = ["## Feature Analysis Results\n"]
    
    # 特征排名（核心）
    md.append("### Feature Discriminative Power\n")
    imp = results["feature_importance"][["feature", "fisher_ratio", "anova_p", "significant"]].head(top_n)
    md.append(imp.to_markdown(index=False))
    md.append("\n")
    
    # 高相关（如果有）
    if len(results["high_correlations"]) > 0:
        md.append("### Redundant Features (|r| >= 0.7)\n")
        md.append(results["high_correlations"][["feature1", "feature2", "correlation"]].head(10).to_markdown(index=False))
        md.append("\n")
    
    return "\n".join(md)


# ---------------------------------------------------------------------
# 5) Plotting helpers (show inline + optional save) - 保持原有功能
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

        out_path = out_dir / f"{feat}_dist.png" if out_dir is not None else None
        _finish_plot(fig, out_path, show)


def plot_corr_heatmap(
    df: pd.DataFrame, out_dir: Path | None = None, show: bool = True
) -> pd.DataFrame:
    """绘制相关性热力图，并返回相关矩阵。"""
    if out_dir is not None:
        ensure_dir(out_dir)
    corr = df.drop(columns=["label"]).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    out_path = out_dir / "correlation_heatmap.png" if out_dir is not None else None
    _finish_plot(fig, out_path, show)
    return corr  # 返回相关矩阵供后续使用


def plot_pca_tsne(
    df: pd.DataFrame,
    out_dir: Path | None = None,
    cfg: VizConfig | None = None,
    show: bool = True,
) -> Dict[str, Any]:
    """执行PCA和t-SNE，返回降维结果和解释信息。"""
    if out_dir is not None:
        ensure_dir(out_dir)
    cfg = cfg or VizConfig()
    df_sub = _subset_df(df, cfg)
    X = df_sub.drop(columns=["label"]).to_numpy()
    y = df_sub["label"].to_numpy()
    
    result = {"pca": None, "tsne": None, "pca_var_explained": None}
    
    if X.shape[0] < 2:
        print("[warn] skip PCA/t-SNE: not enough samples")
        return result
    std = np.nanstd(X, axis=0)
    keep = std > 1e-8
    if not np.any(keep):
        print("[warn] skip PCA/t-SNE: all features are constant")
        return result
    X = X[:, keep]
    if np.unique(X, axis=0).shape[0] < 2:
        print("[warn] skip PCA/t-SNE: only one unique sample")
        return result

    # PCA
    pca = PCA(n_components=2, random_state=cfg.random_seed)
    X_pca = pca.fit_transform(X)
    result["pca"] = X_pca
    result["pca_var_explained"] = pca.explained_variance_ratio_
    
    # 打印PCA方差解释
    print(f"PCA Variance Explained: PC1={pca.explained_variance_ratio_[0]:.2%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.2%}, "
          f"Total={sum(pca.explained_variance_ratio_):.2%}")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, s=30, ax=ax)
    ax.set_title(f"PCA (2D) - Var Explained: {sum(pca.explained_variance_ratio_):.1%}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    out_path = out_dir / "pca_2d.png" if out_dir is not None else None
    _finish_plot(fig, out_path, show)

    # t-SNE
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
    result["tsne"] = X_tsne
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, s=30, ax=ax)
    ax.set_title("t-SNE (2D)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    out_path = out_dir / "tsne_2d.png" if out_dir is not None else None
    _finish_plot(fig, out_path, show)
    
    return result


def plot_feature_importance_bar(df: pd.DataFrame, top_n: int = 15, 
                                 out_dir: Path | None = None, show: bool = True) -> pd.DataFrame:
    """绘制特征重要性条形图（按Fisher ratio排序）。"""
    if out_dir is not None:
        ensure_dir(out_dir)
    
    feat_imp = compute_feature_importance(df)
    top_features = feat_imp.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if p < 0.001 else ('orange' if p < 0.01 else ('yellow' if p < 0.05 else 'gray'))
              for p in top_features["anova_p"]]
    
    bars = ax.barh(range(len(top_features)), top_features["fisher_ratio"], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Fisher Ratio (Between/Within Class Variance)")
    ax.set_title(f"Top {top_n} Most Discriminative Features")
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='p < 0.001 ***'),
        Patch(facecolor='orange', label='p < 0.01 **'),
        Patch(facecolor='yellow', label='p < 0.05 *'),
        Patch(facecolor='gray', label='p >= 0.05'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    out_path = out_dir / "feature_importance.png" if out_dir is not None else None
    _finish_plot(fig, out_path, show)
    
    return feat_imp


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
# 6) Complete Analysis Pipeline (新增)
# ---------------------------------------------------------------------

def run_complete_analysis(
    df: pd.DataFrame,
    features: Iterable[str] | None = None,
    out_dir: Path | None = None,
    show: bool = True,
    cfg: VizConfig | None = None,
    verbose: int = 1,
) -> Dict[str, Any]:
    """运行完整的分析流程：EDA + 可视化。
    
    Args:
        df: 输入数据
        features: 要可视化的特征列表，None则自动选择top特征
        out_dir: 输出目录
        show: 是否显示图片
        cfg: 可视化配置
        verbose: 输出详细程度 (0=静默, 1=精简, 2=详细)
    
    Returns:
        Dict containing all analysis results
    """
    results = {}
    
    # 1. EDA分析（核心）
    eda_results = run_full_eda(df, verbose=verbose)
    results["eda_results"] = eda_results
    
    # 2. 可视化（如果需要）
    if show or out_dir is not None:
        if verbose >= 1:
            print("\n[Generating visualizations...]")
        
        # 自动选择top特征进行可视化
        if features is None:
            top_feats = eda_results["feature_importance"].head(6)["feature"].tolist()
            features = [f for f in top_feats if f in df.columns]
        
        # 特征重要性条形图
        results["feature_importance"] = plot_feature_importance_bar(
            df, top_n=15, out_dir=out_dir, show=show)
        
        # 分布图（只画top特征）
        plot_box_violin_kde(df, features, out_dir=out_dir, show=show)
        
        # 相关性热力图
        results["correlation_matrix"] = plot_corr_heatmap(df, out_dir=out_dir, show=show)
        
        # PCA/t-SNE
        results["pca_tsne_results"] = plot_pca_tsne(df, out_dir=out_dir, cfg=cfg, show=show)
    
    # 3. 保存结果
    if out_dir is not None:
        ensure_dir(out_dir)
        eda_results["basic_stats"].to_csv(out_dir / "eda_basic_stats.csv", index=False)
        eda_results["feature_importance"].to_csv(out_dir / "eda_feature_importance.csv", index=False)
        if len(eda_results["high_correlations"]) > 0:
            eda_results["high_correlations"].to_csv(out_dir / "eda_high_correlations.csv", index=False)
        if verbose >= 1:
            print(f"\n[Results saved to {out_dir}]")
    
    return results


# ---------------------------------------------------------------------
# 7) Example usage
# ---------------------------------------------------------------------
df = load_features_npz(DEFAULT_FEATURES)
eda_results = run_full_eda(df)
results = run_complete_analysis(df, out_dir=DEFAULT_FIG_DIR)
md = eda_to_markdown(eda_results)
print(md)
