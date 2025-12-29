"""评估脚本：全面的模型评估与可视化

功能：
1. 加载所有训练好的模型（KNN, RF, MLP）
2. 在Test集上评估
3. 生成完整的可视化报告
4. 输出详细的评估指标
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from joblib import load as joblib_load
except ImportError:
    joblib_load = None

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


# =====================================================================
# 配置
# =====================================================================
@dataclass
class EvalConfig:
    # 数据路径
    test_path: Path = Path("C:\\Users\\ASUS\\Desktop\\ML_Course\\ML_Coursework\\MyCourse\\results\\splits\\test_selected.npz")
    
    # 模型路径
    model_dir: Path = Path("C:\\Users\\ASUS\\Desktop\\ML_Course\\ML_Coursework\\MyCourse\\models")
    
    # 输出路径
    out_dir: Path = Path("C:\\Users\\ASUS\\Desktop\\ML_Course\\ML_Coursework\\MyCourse\\results\\evaluation")
    
    # 可视化设置
    fig_dpi: int = 150
    show_plots: bool = True


# =====================================================================
# MLP模型定义（需要与训练时一致）
# =====================================================================

class MLP(nn.Module):
    """多层感知机：与训练时结构一致"""
    
    def __init__(self, input_dim: int, hidden_sizes: List[int], num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# =====================================================================
# 数据加载
# =====================================================================

def load_test_data(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """加载测试数据"""
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["labels"].astype(str)
    names = data["feature_names"].tolist() if "feature_names" in data else []
    return X, y, names


def sanitize_features(X: np.ndarray) -> np.ndarray:
    """处理NaN和Inf"""
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def preprocess_for_nn(
    X: np.ndarray,
    scaler: StandardScaler,
    clip_bounds: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """神经网络预处理（使用训练时保存的scaler和clip_bounds）"""
    X = sanitize_features(X)
    clip_low, clip_high = clip_bounds
    X = np.clip(X, clip_low, clip_high)
    X = scaler.transform(X)
    X = np.clip(X, -5.0, 5.0)
    return X


# =====================================================================
# 模型加载
# =====================================================================

def load_sklearn_model(path: Path) -> Tuple[Any, LabelEncoder]:
    """加载sklearn模型（KNN/RF）"""
    if joblib_load is None:
        raise RuntimeError("joblib未安装")
    
    payload = joblib_load(path)
    model = payload["model"]
    le = payload["label_encoder"]
    return model, le


def load_sklearn_mlp_model(path: Path) -> Tuple[Any, LabelEncoder, StandardScaler, Tuple[np.ndarray, np.ndarray], Dict]:
    """加载sklearn MLP模型（含预处理参数）"""
    if joblib_load is None:
        raise RuntimeError("joblib未安装")
    
    artifact = joblib_load(path)
    model = artifact["model"]
    config = artifact["config"]
    
    # 重建scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(artifact["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(artifact["scaler_scale"], dtype=np.float64)
    
    # clip bounds
    clip_low = np.array(artifact["clip_low"], dtype=np.float64)
    clip_high = np.array(artifact["clip_high"], dtype=np.float64)
    clip_bounds = (clip_low, clip_high)
    
    # label encoder (从classes重建)
    le = LabelEncoder()
    le.classes_ = np.array(artifact["classes"])
    
    return model, le, scaler, clip_bounds, config


def load_mlp_model(path: Path) -> Tuple[Any, LabelEncoder, StandardScaler, Tuple[np.ndarray, np.ndarray], Dict]:
    """加载PyTorch MLP模型"""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch未安装")
    
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    # 重建模型
    config = checkpoint["config"]
    input_dim = checkpoint["input_dim"]
    num_classes = checkpoint["num_classes"]
    
    model = MLP(
        input_dim=input_dim,
        hidden_sizes=config["hidden_sizes"],
        num_classes=num_classes,
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    return (
        model,
        checkpoint["label_encoder"],
        checkpoint["scaler"],
        checkpoint["clip_bounds"],
        config,
    )


# =====================================================================
# 评估指标
# =====================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """计算全面的评估指标"""
    
    metrics = {
        # 整体指标
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    
    # Macro AUC
    try:
        if y_proba.shape[1] == 2:
            metrics["macro_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            metrics["macro_auc"] = float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            ))
    except ValueError:
        metrics["macro_auc"] = float("nan")
    
    # 每个类别的指标
    per_class = {}
    for i, cls in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        
        per_class[cls] = {
            "precision": float(precision_score(y_bin, y_pred_bin, zero_division=0)),
            "recall": float(recall_score(y_bin, y_pred_bin, zero_division=0)),
            "f1": float(f1_score(y_bin, y_pred_bin, zero_division=0)),
            "support": int(y_bin.sum()),
        }
        
        # 单类AUC
        try:
            fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
            per_class[cls]["auc"] = float(auc(fpr, tpr))
        except ValueError:
            per_class[cls]["auc"] = float("nan")
    
    metrics["per_class"] = per_class
    
    return metrics


# =====================================================================
# 可视化
# =====================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Optional[Path] = None,
    show: bool = True,
    normalize: bool = False,
) -> None:
    """绘制混淆矩阵"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm, nan=0.0)
        fmt = ".2f"
        vmin, vmax = 0, 1
    else:
        fmt = "d"
        vmin, vmax = None, None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """绘制多类ROC曲线"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        y_bin = (y_true == i).astype(int)
        
        try:
            fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")
        except ValueError:
            continue
    
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_per_class_metrics(
    metrics: Dict[str, Any],
    title: str,
    out_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """绘制每个类别的指标对比"""
    
    per_class = metrics["per_class"]
    class_names = list(per_class.keys())
    
    precision = [per_class[c]["precision"] for c in class_names]
    recall = [per_class[c]["recall"] for c in class_names]
    f1 = [per_class[c]["f1"] for c in class_names]
    auc_scores = [per_class[c]["auc"] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - 1.5*width, precision, width, label="Precision", color="#2ecc71")
    ax.bar(x - 0.5*width, recall, width, label="Recall", color="#3498db")
    ax.bar(x + 0.5*width, f1, width, label="F1", color="#e74c3c")
    ax.bar(x + 1.5*width, auc_scores, width, label="AUC", color="#9b59b6")
    
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_model_comparison(
    all_metrics: Dict[str, Dict[str, Any]],
    out_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """绘制模型对比图"""
    
    model_names = list(all_metrics.keys())
    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    
    data = []
    for model in model_names:
        for metric, label in zip(metric_names, metric_labels):
            val = all_metrics[model].get(metric, 0)
            if np.isnan(val):
                val = 0
            data.append({"Model": model.upper(), "Metric": label, "Score": val})
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metric_labels))
    width = 0.25
    
    colors = {"KNN": "#3498db", "RF": "#2ecc71", "SKLEARN_MLP": "#e74c3c", "DEEP_MLP": "#9b59b6"}
    
    for i, model in enumerate(model_names):
        scores = [all_metrics[model].get(m, 0) for m in metric_names]
        scores = [0 if np.isnan(s) else s for s in scores]
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=model.upper(), 
                     color=colors.get(model.upper(), f"C{i}"))
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"{score:.3f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison on Test Set", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_prediction_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    title: str,
    out_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """绘制预测置信度分布"""
    
    max_proba = np.max(y_proba, axis=1)
    correct = (y_true == y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：正确vs错误的置信度分布
    axes[0].hist(max_proba[correct], bins=20, alpha=0.7, label="Correct", color="#2ecc71")
    axes[0].hist(max_proba[~correct], bins=20, alpha=0.7, label="Wrong", color="#e74c3c")
    axes[0].set_xlabel("Prediction Confidence", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Confidence Distribution", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 右图：置信度vs准确率
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(max_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins)-2)
    
    bin_acc = []
    bin_count = []
    for i in range(len(bins)-1):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
            bin_count.append(mask.sum())
        else:
            bin_acc.append(0)
            bin_count.append(0)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax2 = axes[1]
    ax2.bar(bin_centers, bin_acc, width=0.08, alpha=0.7, color="#3498db", label="Accuracy")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect Calibration")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Calibration Plot", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# =====================================================================
# 主评估流程
# =====================================================================

def evaluate_all(cfg: EvalConfig) -> Dict[str, Any]:
    """评估所有模型"""
    
    results = {}
    all_metrics = {}
    
    # ========== 加载测试数据 ==========
    print("=" * 65)
    print("  EVALUATION REPORT")
    print("=" * 65)
    
    X_test, y_test, feat_names = load_test_data(cfg.test_path)
    X_test_raw = sanitize_features(X_test.copy())
    
    print(f"\n[Test Data] {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # 创建输出目录
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 评估KNN ==========
    knn_path = cfg.model_dir / "knn_model.pkl"
    if knn_path.exists():
        print("\n" + "-" * 65)
        print("  KNN")
        print("-" * 65)
        
        model, le = load_sklearn_model(knn_path)
        y_test_enc = le.transform(y_test)
        class_names = le.classes_.tolist()
        
        y_proba = model.predict_proba(X_test_raw)
        y_pred = np.argmax(y_proba, axis=1)
        
        metrics = compute_metrics(y_test_enc, y_pred, y_proba, class_names)
        all_metrics["knn"] = metrics
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
        print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
        
        # 可视化
        fig_dir = cfg.out_dir / "knn"
        fig_dir.mkdir(exist_ok=True)
        
        plot_confusion_matrix(y_test_enc, y_pred, class_names, "KNN - Confusion Matrix",
                            fig_dir / "confusion_matrix.png", cfg.show_plots)
        plot_confusion_matrix(y_test_enc, y_pred, class_names, "KNN - Normalized Confusion Matrix",
                            fig_dir / "confusion_matrix_norm.png", cfg.show_plots, normalize=True)
        plot_roc_curves(y_test_enc, y_proba, class_names, "KNN - ROC Curves",
                       fig_dir / "roc_curves.png", cfg.show_plots)
        plot_per_class_metrics(metrics, "KNN - Per-Class Metrics",
                              fig_dir / "per_class_metrics.png", cfg.show_plots)
        plot_prediction_confidence(y_test_enc, y_pred, y_proba, "KNN - Prediction Confidence",
                                  fig_dir / "confidence.png", cfg.show_plots)
    
    # ========== 评估RF ==========
    rf_path = cfg.model_dir / "rf_model.pkl"
    if rf_path.exists():
        print("\n" + "-" * 65)
        print("  Random Forest")
        print("-" * 65)
        
        model, le = load_sklearn_model(rf_path)
        y_test_enc = le.transform(y_test)
        class_names = le.classes_.tolist()
        
        y_proba = model.predict_proba(X_test_raw)
        y_pred = np.argmax(y_proba, axis=1)
        
        metrics = compute_metrics(y_test_enc, y_pred, y_proba, class_names)
        all_metrics["rf"] = metrics
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
        print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
        
        # 特征重要性
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            top_idx = np.argsort(importance)[-10:][::-1]
            print(f"\n  Top 10 Important Features:")
            for i, idx in enumerate(top_idx):
                name = feat_names[idx] if idx < len(feat_names) else f"feature_{idx}"
                print(f"    {i+1}. {name}: {importance[idx]:.4f}")
        
        # 可视化
        fig_dir = cfg.out_dir / "rf"
        fig_dir.mkdir(exist_ok=True)
        
        plot_confusion_matrix(y_test_enc, y_pred, class_names, "RF - Confusion Matrix",
                            fig_dir / "confusion_matrix.png", cfg.show_plots)
        plot_confusion_matrix(y_test_enc, y_pred, class_names, "RF - Normalized Confusion Matrix",
                            fig_dir / "confusion_matrix_norm.png", cfg.show_plots, normalize=True)
        plot_roc_curves(y_test_enc, y_proba, class_names, "RF - ROC Curves",
                       fig_dir / "roc_curves.png", cfg.show_plots)
        plot_per_class_metrics(metrics, "RF - Per-Class Metrics",
                              fig_dir / "per_class_metrics.png", cfg.show_plots)
        plot_prediction_confidence(y_test_enc, y_pred, y_proba, "RF - Prediction Confidence",
                                  fig_dir / "confidence.png", cfg.show_plots)
        
        # 特征重要性图
        if hasattr(model, "feature_importances_"):
            fig, ax = plt.subplots(figsize=(10, 6))
            top_n = 15
            top_idx = np.argsort(importance)[-top_n:]
            top_names = [feat_names[i] if i < len(feat_names) else f"f_{i}" for i in top_idx]
            ax.barh(range(top_n), importance[top_idx], color="#2ecc71")
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_names)
            ax.set_xlabel("Importance")
            ax.set_title("RF - Feature Importance (Top 15)")
            fig.tight_layout()
            fig.savefig(fig_dir / "feature_importance.png", dpi=150)
            if cfg.show_plots:
                plt.show()
            plt.close(fig)
    
    # ========== 评估sklearn MLP ==========
    sklearn_mlp_path = cfg.model_dir / "mlp.pkl"
    if sklearn_mlp_path.exists():
        print("\n" + "-" * 65)
        print("  MLP (sklearn)")
        print("-" * 65)
        
        model, le, scaler, clip_bounds, config = load_sklearn_mlp_model(sklearn_mlp_path)
        y_test_enc = le.transform(y_test)
        class_names = le.classes_.tolist()
        
        # 预处理
        X_test_nn = preprocess_for_nn(X_test, scaler, clip_bounds)
        
        # 推理
        y_proba = model.predict_proba(X_test_nn)
        y_pred = np.argmax(y_proba, axis=1)
        
        metrics = compute_metrics(y_test_enc, y_pred, y_proba, class_names)
        all_metrics["sklearn_mlp"] = metrics
        
        print(f"  Architecture: {config.get('hidden_layer_sizes', 'N/A')}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
        print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
        
        # 可视化
        fig_dir = cfg.out_dir / "sklearn_mlp"
        fig_dir.mkdir(exist_ok=True)
        
        plot_confusion_matrix(y_test_enc, y_pred, class_names, "sklearn MLP - Confusion Matrix",
                            fig_dir / "confusion_matrix.png", cfg.show_plots)
        plot_confusion_matrix(y_test_enc, y_pred, class_names, "sklearn MLP - Normalized Confusion Matrix",
                            fig_dir / "confusion_matrix_norm.png", cfg.show_plots, normalize=True)
        plot_roc_curves(y_test_enc, y_proba, class_names, "sklearn MLP - ROC Curves",
                       fig_dir / "roc_curves.png", cfg.show_plots)
        plot_per_class_metrics(metrics, "sklearn MLP - Per-Class Metrics",
                              fig_dir / "per_class_metrics.png", cfg.show_plots)
        plot_prediction_confidence(y_test_enc, y_pred, y_proba, "sklearn MLP - Prediction Confidence",
                                  fig_dir / "confidence.png", cfg.show_plots)
    
    # ========== 评估DeepMLP (PyTorch) ==========
    mlp_path = cfg.model_dir / "mlp_model.pt"
    if mlp_path.exists() and HAS_TORCH:
        print("\n" + "-" * 65)
        print("  DeepMLP (PyTorch)")
        print("-" * 65)
        
        model, le, scaler, clip_bounds, config = load_mlp_model(mlp_path)
        y_test_enc = le.transform(y_test)
        class_names = le.classes_.tolist()
        
        # 预处理
        X_test_nn = preprocess_for_nn(X_test, scaler, clip_bounds)
        
        # 推理
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test_nn)
            logits = model(X_t)
            y_proba = torch.softmax(logits, dim=1).numpy()
            y_pred = np.argmax(y_proba, axis=1)
        
        metrics = compute_metrics(y_test_enc, y_pred, y_proba, class_names)
        all_metrics["deep_mlp"] = metrics
        
        print(f"  Architecture: {config['hidden_sizes']}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
        print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
        
        # 可视化
        fig_dir = cfg.out_dir / "deep_mlp"
        fig_dir.mkdir(exist_ok=True)
        
        plot_confusion_matrix(y_test_enc, y_pred, class_names, "DeepMLP - Confusion Matrix",
                            fig_dir / "confusion_matrix.png", cfg.show_plots)
        plot_confusion_matrix(y_test_enc, y_pred, class_names, "DeepMLP - Normalized Confusion Matrix",
                            fig_dir / "confusion_matrix_norm.png", cfg.show_plots, normalize=True)
        plot_roc_curves(y_test_enc, y_proba, class_names, "DeepMLP - ROC Curves",
                       fig_dir / "roc_curves.png", cfg.show_plots)
        plot_per_class_metrics(metrics, "DeepMLP - Per-Class Metrics",
                              fig_dir / "per_class_metrics.png", cfg.show_plots)
        plot_prediction_confidence(y_test_enc, y_pred, y_proba, "DeepMLP - Prediction Confidence",
                                  fig_dir / "confidence.png", cfg.show_plots)
    
    # ========== 模型对比 ==========
    if len(all_metrics) > 1:
        print("\n" + "-" * 65)
        print("  Model Comparison")
        print("-" * 65)
        
        plot_model_comparison(all_metrics, cfg.out_dir / "model_comparison.png", cfg.show_plots)
    
    # ========== 总结 ==========
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    
    # 宏平均表格输出
    print(f"\n[Overall Metrics - Macro Average]")
    print(f"{'Model':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-" * 70)
    
    best_model = None
    best_auc = -1
    
    for name, m in all_metrics.items():
        acc = m["accuracy"]
        prec = m["macro_precision"]
        rec = m["macro_recall"]
        f1 = m["macro_f1"]
        auc_score = m["macro_auc"]
        
        print(f"{name.upper():<10} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {auc_score:<12.4f}")
        
        if auc_score > best_auc:
            best_auc = auc_score
            best_model = name
    
    if best_model:
        print(f"\n  ✓ Best Model: {best_model.upper()} (AUC = {best_auc:.4f})")
    
    # Per-class详细指标表格
    print("\n" + "=" * 65)
    print("  PER-CLASS METRICS (Best Model)")
    print("=" * 65)
    
    if best_model and best_model in all_metrics:
        per_class = all_metrics[best_model]["per_class"]
        class_names = list(per_class.keys())
        
        print(f"\n[{best_model.upper()} - Per-Class Performance]")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12} {'Support':<10}")
        print("-" * 80)
        
        for cls in class_names:
            prec = per_class[cls]["precision"]
            rec = per_class[cls]["recall"]
            f1 = per_class[cls]["f1"]
            auc_val = per_class[cls]["auc"]
            support = per_class[cls]["support"]
            
            print(f"{cls:<20} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {auc_val:<12.4f} {support:<10}")
        
        # 计算并显示宏平均和加权平均
        macro_prec = np.mean([per_class[c]["precision"] for c in class_names])
        macro_rec = np.mean([per_class[c]["recall"] for c in class_names])
        macro_f1 = np.mean([per_class[c]["f1"] for c in class_names])
        macro_auc = np.mean([per_class[c]["auc"] for c in class_names if not np.isnan(per_class[c]["auc"])])
        total_support = sum([per_class[c]["support"] for c in class_names])
        
        print("-" * 80)
        print(f"{'Macro Average':<20} {macro_prec:<12.4f} {macro_rec:<12.4f} {macro_f1:<12.4f} {macro_auc:<12.4f} {total_support:<10}")
        
        # 加权平均
        weights = np.array([per_class[c]["support"] for c in class_names])
        if weights.sum() > 0:
            weighted_prec = np.average([per_class[c]["precision"] for c in class_names], weights=weights)
            weighted_rec = np.average([per_class[c]["recall"] for c in class_names], weights=weights)
            weighted_f1 = np.average([per_class[c]["f1"] for c in class_names], weights=weights)
            print(f"{'Weighted Average':<20} {weighted_prec:<12.4f} {weighted_rec:<12.4f} {weighted_f1:<12.4f} {'':<12} {total_support:<10}")
    
    # 保存结果
    results["metrics"] = all_metrics
    results["best_model"] = best_model
    
    # 转换numpy类型
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(cfg.out_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, default=convert, ensure_ascii=False, indent=2)
    
    print(f"\n[保存] {cfg.out_dir / 'evaluation_results.json'}")
    print(f"[图表] {cfg.out_dir}")
    
    return results


# =====================================================================
# 主函数
# =====================================================================

def main():
    cfg = EvalConfig()
    results = evaluate_all(cfg)
    
    print("\n" + "=" * 65)
    print("  EVALUATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()