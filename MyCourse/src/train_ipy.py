"""训练脚本：KNN + RF + MLP + DeepMLP (PyTorch)

修复问题：
1. MLP预处理逻辑修复（先计算clip边界再clip）
2. 用PyTorch DeepMLP替换CNN（CNN不适合手工特征）
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# PyTorch（可选）
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# tqdm（可选）
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =====================================================================
# 配置
# =====================================================================
@dataclass
class TrainConfig:
    # 数据路径
    train_path: Path = Path("C:\\Users\\ASUS\\Desktop\\ML_Course\\ML_Coursework\\MyCourse\\results\\splits\\train_selected.npz")
    val_path: Path = Path("C:\\Users\\ASUS\\Desktop\\ML_Course\\ML_Coursework\\MyCourse\\results\\splits\\val_selected.npz")
    test_path: Path = Path("C:\\Users\\ASUS\\Desktop\\ML_Course\\ML_Coursework\\MyCourse\\results\\splits\\test_selected.npz")
    
    # 输出路径
    out_dir: Path = Path("C:\\Users\\ASUS\\Desktop\\ML_Course\\ML_Coursework\\MyCourse\\results\\training")
    model_dir: Path = Path("C:\\Users\\ASUS\\Desktop\\ML_Course\\ML_Coursework\\MyCourse\\models")
    
    # 交叉验证
    n_splits: int = 5
    random_seed: int = 42
    
    # 是否包含各模型
    include_knn: bool = True
    include_rf: bool = True
    include_mlp: bool = True
    include_deep: bool = True  # PyTorch DeepMLP


@dataclass
class MLPConfig:
    """sklearn MLP配置"""
    hidden_layer_sizes: Tuple[int, ...] = (128, 64)
    alpha: float = 1e-3        # L2正则化
    lr: float = 1e-3           # 学习率
    max_iter: int = 300        # 最大迭代次数


@dataclass
class DeepMLPConfig:
    """PyTorch DeepMLP配置"""
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 64
    patience: int = 15         # 早停patience


# =====================================================================
# 数据加载与预处理
# =====================================================================

def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """加载npz文件"""
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["labels"].astype(str)
    names = data["feature_names"].tolist() if "feature_names" in data else []
    return X, y, names


def sanitize_features(X: np.ndarray) -> np.ndarray:
    """处理NaN和Inf"""
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def preprocess_for_mlp(
    X_train: np.ndarray, 
    X_val: np.ndarray,
    X_test: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], StandardScaler, Tuple]:
    """MLP预处理：clip + 标准化（修复版）
    
    关键修复：先在原始数据上计算clip边界，再应用clip
    """
    # Step 1: 处理NaN/Inf
    X_train = sanitize_features(X_train)
    X_val = sanitize_features(X_val)
    if X_test is not None:
        X_test = sanitize_features(X_test)
    
    # Step 2: 基于原始训练集计算clip边界（1-99百分位）
    clip_low = np.percentile(X_train, 1, axis=0)
    clip_high = np.percentile(X_train, 99, axis=0)
    
    # Step 3: 应用clip
    X_train = np.clip(X_train, clip_low, clip_high)
    X_val = np.clip(X_val, clip_low, clip_high)
    if X_test is not None:
        X_test = np.clip(X_test, clip_low, clip_high)
    
    # Step 4: 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    if X_test is not None:
        X_test = scaler.transform(X_test)
    
    # Step 5: 标准化后再clip到[-5, 5]防止极端值
    X_train = np.clip(X_train, -5.0, 5.0)
    X_val = np.clip(X_val, -5.0, 5.0)
    if X_test is not None:
        X_test = np.clip(X_test, -5.0, 5.0)
    
    clip_bounds = (clip_low, clip_high)
    return X_train, X_val, X_test, scaler, clip_bounds


# =====================================================================
# 评估指标
# =====================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    
    # Macro AUC（OvR）
    try:
        if y_proba.shape[1] == 2:
            metrics["macro_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics["macro_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        metrics["macro_auc"] = float("nan")
    
    return metrics


# =====================================================================
# 模型构建
# =====================================================================

def build_knn(params: Dict) -> KNeighborsClassifier:
    return KNeighborsClassifier(**params)


def build_rf(params: Dict) -> RandomForestClassifier:
    return RandomForestClassifier(**params, random_state=42, n_jobs=-1)


def build_mlp(cfg: MLPConfig) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        solver="adam",
        activation="relu",
        learning_rate_init=cfg.lr,
        learning_rate="adaptive",
        alpha=cfg.alpha,
        max_iter=cfg.max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    )


# =====================================================================
# PyTorch DeepMLP
# =====================================================================

class DeepMLP(nn.Module):
    """更深的MLP网络，带BatchNorm和Dropout"""
    
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


def train_deep_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: DeepMLPConfig,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """训练PyTorch DeepMLP"""
    
    if not HAS_TORCH:
        raise RuntimeError("PyTorch未安装")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据转换
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    # 模型
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = DeepMLP(input_dim, cfg.hidden_sizes, num_classes, cfg.dropout).to(device)
    
    # 优化器和损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_auc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_auc": [], "val_acc": []}
    
    iterator = range(cfg.epochs)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="DeepMLP", unit="ep")
    
    for epoch in iterator:
        # 训练
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        
        # 验证
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t.to(device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            pred = np.argmax(proba, axis=1)
        
        metrics = evaluate(y_val, pred, proba)
        history["val_auc"].append(metrics["macro_auc"])
        history["val_acc"].append(metrics["accuracy"])
        
        scheduler.step(metrics["macro_auc"])
        
        # 早停
        if metrics["macro_auc"] > best_auc:
            best_auc = metrics["macro_auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.patience:
            if tqdm is not None:
                iterator.set_postfix({"early_stop": epoch})
            break
        
        if tqdm is not None:
            iterator.set_postfix({"loss": f"{avg_loss:.4f}", "auc": f"{metrics['macro_auc']:.4f}"})
    
    # 加载最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, {"history": history, "best_auc": best_auc}


# =====================================================================
# 交叉验证
# =====================================================================

def cv_search(
    builder_fn,
    param_grid: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[Dict, float]:
    """网格搜索 + 交叉验证（用于原始特征模型：KNN/RF）"""
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_seed)

    best_params = None
    best_score = -1.0

    for params in param_grid:
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_vl = X[train_idx], X[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            try:
                model = builder_fn(params)
                model.fit(X_tr, y_tr)
                y_proba = model.predict_proba(X_vl)
                y_pred = np.argmax(y_proba, axis=1)
                m = evaluate(y_vl, y_pred, y_proba)
                scores.append(m["macro_auc"])
            except Exception:
                scores.append(0.0)

        mean_score = np.nanmean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score


def cv_search_mlp(
    param_grid: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[Dict, float]:
    """sklearn MLP的5折CV，按折进行独立预处理以避免泄漏"""
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_seed)

    best_params = None
    best_score = -1.0

    for params in param_grid:
        # 将参数映射到MLPConfig（未给出则使用默认）
        mlp_cfg = MLPConfig(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (128, 64)),
            alpha=params.get("alpha", 1e-3),
            lr=params.get("lr", 1e-3),
            max_iter=params.get("max_iter", 300),
        )

        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_vl = X[train_idx], X[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            # 每折独立预处理
            X_tr_nn, X_vl_nn, _, _, _ = preprocess_for_mlp(X_tr, X_vl)

            try:
                model = build_mlp(mlp_cfg)
                model.fit(X_tr_nn, y_tr)
                y_proba = model.predict_proba(X_vl_nn)
                y_pred = np.argmax(y_proba, axis=1)
                m = evaluate(y_vl, y_pred, y_proba)
                scores.append(m.get("macro_auc", 0.0))
            except Exception:
                scores.append(0.0)

        mean_score = np.nanmean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = {
                "hidden_layer_sizes": mlp_cfg.hidden_layer_sizes,
                "alpha": mlp_cfg.alpha,
                "lr": mlp_cfg.lr,
                "max_iter": mlp_cfg.max_iter,
            }

    return best_params, best_score


def cv_search_deep(
    param_grid: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[Dict, float]:
    """PyTorch DeepMLP的5折CV，按折进行独立预处理以避免泄漏"""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch未安装，无法进行DeepMLP CV")

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_seed)

    best_params = None
    best_score = -1.0

    for params in param_grid:
        # 将参数映射到DeepMLPConfig
        deep_cfg = DeepMLPConfig(
            hidden_sizes=params.get("hidden_sizes", [256, 128, 64]),
            dropout=params.get("dropout", 0.3),
            lr=params.get("lr", 1e-3),
            weight_decay=params.get("weight_decay", 1e-4),
            epochs=params.get("epochs", 60),
            batch_size=params.get("batch_size", 64),
            patience=params.get("patience", 10),
        )

        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_vl = X[train_idx], X[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            # 每折独立预处理
            X_tr_nn, X_vl_nn, _, _, _ = preprocess_for_mlp(X_tr, X_vl)

            try:
                model, _ = train_deep_mlp(X_tr_nn, y_tr, X_vl_nn, y_vl, deep_cfg)
                model.eval()
                device = next(model.parameters()).device
                with torch.no_grad():
                    X_vl_t = torch.FloatTensor(X_vl_nn).to(device)
                    logits = model(X_vl_t)
                    y_proba = torch.softmax(logits, dim=1).cpu().numpy()
                    y_pred = np.argmax(y_proba, axis=1)
                m = evaluate(y_vl, y_pred, y_proba)
                scores.append(m.get("macro_auc", 0.0))
            except Exception:
                scores.append(0.0)

        mean_score = np.nanmean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = {
                "hidden_sizes": deep_cfg.hidden_sizes,
                "dropout": deep_cfg.dropout,
                "lr": deep_cfg.lr,
                "weight_decay": deep_cfg.weight_decay,
                "epochs": deep_cfg.epochs,
                "batch_size": deep_cfg.batch_size,
                "patience": deep_cfg.patience,
            }

    return best_params, best_score


# =====================================================================
# 主训练流程
# =====================================================================

def train_all(cfg: TrainConfig) -> Dict[str, Any]:
    """训练所有模型"""
    
    results = {}
    
    # ========== 加载数据 ==========
    print("=" * 60)
    print("  LOADING DATA (train + val for CV)")
    print("=" * 60)
    
    X_train, y_train, feat_names = load_npz(cfg.train_path)
    X_val, y_val, _ = load_npz(cfg.val_path)
    
    print(f"[Train] {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"[Val]   {X_val.shape[0]} samples")

    # 合并训练与验证用于CV
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    
    # 标签编码
    le = LabelEncoder()
    le.fit(y_full)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_full_enc = le.transform(y_full)
    
    print(f"[Classes] {le.classes_.tolist()}")
    
    # 预处理（用于最终训练 MLP 和 DeepMLP）将在选参后对全量数据进行
    
    # 确保模型保存目录存在
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    
    # 超参网格
    knn_grid = [{"n_neighbors": k, "weights": w} for k in [3, 5, 7, 9] for w in ["uniform", "distance"]]
    rf_grid = [{"n_estimators": n, "max_depth": d} for n in [100, 200] for d in [5, 10, None]]
    mlp_grid = [
        {"hidden_layer_sizes": (128, 64), "alpha": 1e-3, "lr": 1e-3},
        {"hidden_layer_sizes": (256, 128), "alpha": 1e-4, "lr": 1e-3},
        {"hidden_layer_sizes": (128, 64, 32), "alpha": 1e-3, "lr": 1e-3},
    ]
    deep_grid = [
        {"hidden_sizes": [256, 128, 64], "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 60, "batch_size": 64, "patience": 10},
        {"hidden_sizes": [128, 64], "dropout": 0.3, "lr": 5e-4, "weight_decay": 5e-5, "epochs": 80, "batch_size": 64, "patience": 10},
    ]
    
    # ========== KNN ==========
    if cfg.include_knn:
        print("\n" + "-" * 60)
        print("  CV Selecting KNN params (5-fold)")
        print("-" * 60)
        
        best_params, cv_auc = cv_search(build_knn, knn_grid, X_full, y_full_enc, cfg)
        
        # 用全部数据训练最终模型
        model = build_knn(best_params)
        model.fit(X_full, y_full_enc)
        
        results["knn"] = {
            "best_params": best_params,
            "cv_auc": cv_auc,
            "final_model_trained": True,
        }
        print(f"  params: {best_params}")
        print(f"  cv_auc={cv_auc:.4f}")
        # 保存模型（含标签编码器）
        try:
            knn_path = cfg.model_dir / "knn_model.pkl"
            dump({"model": model, "label_encoder": le}, knn_path)
            print(f"  [保存模型] {knn_path}")
        except Exception as e:
            print(f"  [保存失败] KNN: {e}")
    
    # ========== Random Forest ==========
    if cfg.include_rf:
        print("\n" + "-" * 60)
        print("  CV Selecting Random Forest params (5-fold)")
        print("-" * 60)
        
        best_params, cv_auc = cv_search(build_rf, rf_grid, X_full, y_full_enc, cfg)
        
        # 用全部数据训练最终模型
        model = build_rf(best_params)
        model.fit(X_full, y_full_enc)
        
        results["rf"] = {
            "best_params": best_params,
            "cv_auc": cv_auc,
            "final_model_trained": True,
        }
        print(f"  params: {best_params}")
        print(f"  cv_auc={cv_auc:.4f}")
        # 保存模型（含标签编码器）
        try:
            rf_path = cfg.model_dir / "rf_model.pkl"
            dump({"model": model, "label_encoder": le}, rf_path)
            print(f"  [保存模型] {rf_path}")
        except Exception as e:
            print(f"  [保存失败] RF: {e}")
    
    # ========== sklearn MLP ==========
    if cfg.include_mlp:
        print("\n" + "-" * 60)
        print("  CV Selecting MLP (sklearn) params (5-fold)")
        print("-" * 60)
        
        try:
            best_params, cv_auc = cv_search_mlp(mlp_grid, X_full, y_full_enc, cfg)
            # 用全部数据进行最终预处理并训练
            X_full_nn, _, _, scaler, clip_bounds = preprocess_for_mlp(X_full, X_full)
            mlp_cfg = MLPConfig(
                hidden_layer_sizes=best_params.get("hidden_layer_sizes", (128, 64)),
                alpha=best_params.get("alpha", 1e-3),
                lr=best_params.get("lr", 1e-3),
                max_iter=best_params.get("max_iter", 300),
            )
            model = build_mlp(mlp_cfg)
            model.fit(X_full_nn, y_full_enc)
            
            results["mlp"] = {
                "best_params": best_params,
                "cv_auc": cv_auc,
                "final_model_trained": True,
                "scaler": {
                    "mean_": scaler.mean_.tolist(),
                    "scale_": scaler.scale_.tolist(),
                },
                "clip_bounds": {
                    "low": clip_bounds[0].tolist(),
                    "high": clip_bounds[1].tolist(),
                },
            }
            print(f"  params: {best_params}  cv_auc={cv_auc:.4f}")
            # 保存模型（连同预处理参数）
            try:
                mlp_path = cfg.model_dir / "mlp.pkl"
                artifact = {
                    "model": model,
                    "config": mlp_cfg.__dict__,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "clip_low": clip_bounds[0].tolist(),
                    "clip_high": clip_bounds[1].tolist(),
                    "classes": le.classes_.tolist(),
                }
                dump(artifact, mlp_path)
                print(f"  [保存模型] {mlp_path}")
            except Exception as e:
                print(f"  [保存失败] MLP: {e}")
        except Exception as e:
            results["mlp"] = {"error": str(e)}
            print(f"  [ERROR] {e}")
    
    # ========== PyTorch DeepMLP ==========
    if cfg.include_deep and HAS_TORCH:
        print("\n" + "-" * 60)
        print("  CV Selecting DeepMLP (PyTorch) params (5-fold)")
        print("-" * 60)
        
        try:
            best_params, cv_auc = cv_search_deep(deep_grid, X_full, y_full_enc, cfg)
            # 用全部数据进行最终预处理并训练（采用全部数据，val用于训练过程记录，不用于评估）
            X_full_nn, _, _, _, _ = preprocess_for_mlp(X_full, X_full)
            deep_cfg = DeepMLPConfig(
                hidden_sizes=best_params.get("hidden_sizes", [256, 128, 64]),
                dropout=best_params.get("dropout", 0.3),
                lr=best_params.get("lr", 1e-3),
                weight_decay=best_params.get("weight_decay", 1e-4),
                epochs=best_params.get("epochs", 60),
                batch_size=best_params.get("batch_size", 64),
                patience=best_params.get("patience", 10),
            )
            model, info = train_deep_mlp(X_full_nn, y_full_enc, X_full_nn, y_full_enc, deep_cfg)
            
            results["deep_mlp"] = {
                "best_params": best_params,
                "cv_auc": cv_auc,
                "final_model_trained": True,
                "best_train_auc": info.get("best_auc", None),
            }
            print(f"  params: {best_params}  cv_auc={cv_auc:.4f}")
            # 保存模型（.pt，包含必要的恢复信息）
            try:
                deep_path = cfg.model_dir / "mlp_model.pt"
                # 训练所用的预处理（用于恢复）
                X_full_nn, _, _, scaler_save, clip_bounds_save = preprocess_for_mlp(X_full, X_full)
                torch.save({
                    "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                    "config": deep_cfg.__dict__,
                    "input_dim": int(X_full_nn.shape[1]),
                    "num_classes": int(len(le.classes_)),
                    "label_encoder": le,
                    "scaler": scaler_save,
                    "clip_bounds": (clip_bounds_save[0], clip_bounds_save[1]),
                }, deep_path)
                print(f"  [保存模型] {deep_path}")
            except Exception as e:
                print(f"  [保存失败] DeepMLP: {e}")
        except Exception as e:
            results["deep_mlp"] = {"error": str(e)}
            print(f"  [ERROR] {e}")
    elif cfg.include_deep and not HAS_TORCH:
        print("\n[SKIP] DeepMLP: PyTorch未安装")
        results["deep_mlp"] = {"error": "PyTorch not installed"}
    
    # ========== 可选：在测试集上评估（如果存在） ==========
    try:
        X_test, y_test, _ = load_npz(cfg.test_path)
        y_test_enc = le.transform(y_test)
        print("\n" + "-" * 60)
        print("  Evaluating on TEST set (optional)")
        print("-" * 60)
        
        # KNN
        if "knn" in results and "error" not in results["knn"]:
            model = build_knn(results["knn"]["best_params"]) if results["knn"].get("best_params") else build_knn({})
            model.fit(X_full, y_full_enc)
            y_proba = model.predict_proba(X_test)
            y_pred = np.argmax(y_proba, axis=1)
            results["knn"]["test"] = evaluate(y_test_enc, y_pred, y_proba)
        
        # RF
        if "rf" in results and "error" not in results["rf"]:
            model = build_rf(results["rf"]["best_params"]) if results["rf"].get("best_params") else build_rf({})
            model.fit(X_full, y_full_enc)
            y_proba = model.predict_proba(X_test)
            y_pred = np.argmax(y_proba, axis=1)
            results["rf"]["test"] = evaluate(y_test_enc, y_pred, y_proba)
        
        # MLP（使用全量训练得到的scaler/clip）
        if "mlp" in results and "error" not in results["mlp"] and results["mlp"].get("scaler"):
            # 复用已保存的clip和标准化参数
            clip_low = np.array(results["mlp"]["clip_bounds"]["low"], dtype=np.float64)
            clip_high = np.array(results["mlp"]["clip_bounds"]["high"], dtype=np.float64)
            X_test_proc = np.clip(sanitize_features(X_test), clip_low, clip_high)
            # 使用保存的scaler参数构建scaler
            scaler = StandardScaler()
            scaler.mean_ = np.array(results["mlp"]["scaler"]["mean_"], dtype=np.float64)
            scaler.scale_ = np.array(results["mlp"]["scaler"]["scale_"], dtype=np.float64)
            X_full_nn, _, _, _, _ = preprocess_for_mlp(X_full, X_full)  # 重新获得训练用特征
            X_test_proc = scaler.transform(X_test_proc)
            X_test_proc = np.clip(X_test_proc, -5.0, 5.0)
            # 训练并评估
            mlp_cfg = MLPConfig(
                hidden_layer_sizes=tuple(results["mlp"]["best_params"]["hidden_layer_sizes"]),
                alpha=float(results["mlp"]["best_params"]["alpha"]),
                lr=float(results["mlp"]["best_params"]["lr"]),
                max_iter=int(results["mlp"]["best_params"].get("max_iter", 300)),
            )
            model = build_mlp(mlp_cfg)
            model.fit(X_full_nn, y_full_enc)
            y_proba = model.predict_proba(X_test_proc)
            y_pred = np.argmax(y_proba, axis=1)
            results["mlp"]["test"] = evaluate(y_test_enc, y_pred, y_proba)
        
        # DeepMLP（使用全量训练得到的模型直接评估）
        if HAS_TORCH and "deep_mlp" in results and "error" not in results["deep_mlp"]:
            # 预处理与训练同分布
            X_full_nn, _, _, _, _ = preprocess_for_mlp(X_full, X_full)
            X_test_nn, _, _, _, _ = preprocess_for_mlp(X_full, X_test)
            deep_cfg = DeepMLPConfig(**results["deep_mlp"]["best_params"]) if results["deep_mlp"].get("best_params") else DeepMLPConfig()
            model, _ = train_deep_mlp(X_full_nn, y_full_enc, X_full_nn, y_full_enc, deep_cfg)
            model.eval()
            device = next(model.parameters()).device
            with torch.no_grad():
                X_t = torch.FloatTensor(X_test_nn).to(device)
                logits = model(X_t)
                y_proba = torch.softmax(logits, dim=1).cpu().numpy()
                y_pred = np.argmax(y_proba, axis=1)
            results["deep_mlp"]["test"] = evaluate(y_test_enc, y_pred, y_proba)
    except Exception:
        print("[INFO] 测试集评估跳过（未找到或标签不匹配）")

    # ========== 选择最佳模型 ==========
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    def get_cv_auc(name: str) -> float:
        return results.get(name, {}).get("cv_auc", -1.0)
    
    model_names = [k for k in results.keys() if "cv_auc" in results.get(k, {})]
    if model_names:
        best_name = max(model_names, key=get_cv_auc)
        best_auc = get_cv_auc(best_name)
        results["best_model"] = best_name
        
        print(f"\n  Model Performance (CV AUC, 5-fold):")
        for name in model_names:
            auc = get_cv_auc(name)
            marker = " <-- BEST" if name == best_name else ""
            print(f"    {name:12s}: {auc:.4f}{marker}")
        
        print(f"\n  Best Model: {best_name} (AUC={best_auc:.4f})")
    
    # ========== 保存结果 ==========
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存metrics
    metrics_path = cfg.out_dir / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        # 转换numpy类型
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, default=convert, ensure_ascii=False, indent=2)
    
    print(f"\n[保存] {metrics_path}")
    
    return results


# =====================================================================
# 主函数
# =====================================================================

def main():
    cfg = TrainConfig()
    
    # 可以在这里修改配置
    # cfg.train_path = Path("your/path/train.npz")
    # cfg.include_deep = False  # 不训练PyTorch模型
    
    results = train_all(cfg)
    
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()