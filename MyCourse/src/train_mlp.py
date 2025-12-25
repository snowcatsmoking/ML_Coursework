"""MLP-only training, validation, test, and plots with progress."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

DEFAULT_TRAIN = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/splits/train_full.npz")
DEFAULT_VAL = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/splits/val.npz")
DEFAULT_TEST = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/splits/test.npz")
DEFAULT_OUT = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/mlp")
DEFAULT_MODEL = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/models/mlp_model.pkl")


@dataclass
class MLPConfig:
    hidden_layer_sizes: Tuple[int, int] = (128, 64)
    alpha: float = 1e-2
    lr: float = 1e-3  # 提高学习率，配合batch_size更稳定
    epochs: int = 80


def load_features(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """加载特征文件"""
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float64)  # 改用float64提高数值稳定性
    y = data["labels"].astype(str)
    return X, y


def sanitize_features(X: np.ndarray) -> np.ndarray:
    """处理NaN和Inf值"""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def compute_clip_bounds(X: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> Tuple[np.ndarray, np.ndarray]:
    """计算clip边界（基于训练集）"""
    low = np.percentile(X, low_q, axis=0)
    high = np.percentile(X, high_q, axis=0)
    return low, high


def apply_clip(X: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """应用clip边界"""
    return np.clip(X, low, high)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        metrics["macro_auc"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        )
    except Exception:
        metrics["macro_auc"] = float("nan")
    return metrics


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    """绘制混淆矩阵"""
    conf = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
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
    ax.set_title("MLP Confusion (Row-Normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    """绘制ROC曲线"""
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(labels):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{label} (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("MLP ROC Curves (OvR)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_history(history: Dict[str, list], out_path: Path) -> None:
    """绘制训练历史"""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["val_macro_auc"], label="val_macro_auc")
    ax.plot(history["val_acc"], label="val_acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("MLP Validation Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_mlp_stable(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: MLPConfig,
) -> Tuple[MLPClassifier, Dict[str, list]]:
    """稳定的MLP训练（不使用partial_fit）"""
    
    # 使用标准fit而不是partial_fit，更稳定
    model = MLPClassifier(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        solver="adam",
        activation="relu",
        learning_rate_init=cfg.lr,
        learning_rate="adaptive",
        alpha=cfg.alpha,
        max_iter=cfg.epochs,
        early_stopping=True,  # 启用早停
        validation_fraction=0.1,  # 内部验证集比例
        n_iter_no_change=10,  # 早停patience
        random_state=42,
        verbose=True,  # 显示训练过程
    )
    
    # 直接训练
    model.fit(X_train, y_train)
    
    # 构造history（从loss_curve_获取）
    history = {
        "val_macro_auc": [],
        "val_acc": [],
        "loss": model.loss_curve_ if hasattr(model, 'loss_curve_') else []
    }
    
    # 在验证集上评估
    y_proba = model.predict_proba(X_val)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = eval_metrics(y_val, y_pred, y_proba)
    history["val_macro_auc"].append(metrics["macro_auc"])
    history["val_acc"].append(metrics["accuracy"])
    
    return model, history


def plot_loss_curve(loss_curve: list, out_path: Path) -> None:
    """绘制loss曲线"""
    if not loss_curve:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_curve, label="Training Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("MLP Training Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MLP-only training with plots.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="Train .npz")
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL, help="Val .npz")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST, help="Test .npz")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output dir")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL, help="Model output")
    parser.add_argument("--epochs", type=int, default=200, help="Max iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1e-3, help="L2 regularization")
    args = parser.parse_args()

    # ===== 加载数据 =====
    X_train, y_train = load_features(args.train)
    X_val, y_val = load_features(args.val)
    X_test, y_test = load_features(args.test)

    # ===== 打印原始数据统计信息 =====
    print(f"[DEBUG] X_train shape: {X_train.shape}, range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"[DEBUG] NaN count: {np.isnan(X_train).sum()}, Inf count: {np.isinf(X_train).sum()}")

    # ===== Step 1: 处理NaN和Inf =====
    X_train = sanitize_features(X_train)
    X_val = sanitize_features(X_val)
    X_test = sanitize_features(X_test)

    # ===== Step 2: 基于原始训练集计算clip边界 =====
    clip_low, clip_high = compute_clip_bounds(X_train, low_q=1.0, high_q=99.0)
    X_train = apply_clip(X_train, clip_low, clip_high)
    X_val = apply_clip(X_val, clip_low, clip_high)
    X_test = apply_clip(X_test, clip_low, clip_high)

    # ===== Step 3: 标准化 =====
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ===== Step 4: 标准化后clip =====
    X_train = np.clip(X_train, -3.0, 3.0)
    X_val = np.clip(X_val, -3.0, 3.0)
    X_test = np.clip(X_test, -3.0, 3.0)

    # ===== 检查数据 =====
    print(f"[DEBUG] After preprocessing - X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"[DEBUG] X_train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")

    # ===== 标签编码 =====
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    
    print(f"[DEBUG] Classes: {le.classes_}")
    print(f"[DEBUG] Train class distribution: {np.bincount(y_train_enc)}")

    # ===== 训练 =====
    cfg = MLPConfig(alpha=args.alpha, lr=args.lr, epochs=args.epochs)
    print(f"\n[INFO] Training MLP with config: {cfg}")
    model, history = train_mlp_stable(X_train, y_train_enc, X_val, y_val_enc, cfg)

    # ===== 测试集评估 =====
    y_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = eval_metrics(y_test_enc, y_pred, y_proba)

    print(f"\n[RESULT] Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"[RESULT] Test Macro F1: {metrics['macro_f1']:.4f}")
    print(f"[RESULT] Test Macro AUC: {metrics['macro_auc']:.4f}")

    # ===== 保存结果 =====
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"test": metrics}, f, ensure_ascii=False, indent=2)

    plot_confusion(y_test_enc, y_pred, le.classes_, fig_dir / "mlp_confusion.png")
    plot_roc(y_test_enc, y_proba, le.classes_, fig_dir / "mlp_roc.png")
    
    # 绘制loss曲线
    if "loss" in history and history["loss"]:
        plot_loss_curve(history["loss"], fig_dir / "mlp_loss.png")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": model, "label_encoder": le, "scaler": scaler, "clip_bounds": (clip_low, clip_high)}, args.model_path)

    print(f"\n[INFO] Model saved to {args.model_path}")
    print(f"[INFO] Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()