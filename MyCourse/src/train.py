"""Train models for hum/whistle song classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

import librosa

DEFAULT_TRAIN = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/splits/train_full.npz")
DEFAULT_VAL = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/splits/val.npz")
DEFAULT_OUT = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results")
DEFAULT_MODEL = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/models/final_model.pkl")


@dataclass
class TrainConfig:
    n_splits: int = 5
    random_seed: int = 42


@dataclass
class MLPStableConfig:
    hidden_layer_sizes: Tuple[int, int] = (128, 64)
    alpha: float = 1e-3
    lr: float = 1e-3
    epochs: int = 200


def load_features(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["labels"].astype(str)
    paths = data["paths"].astype(str)
    return X, y, paths


def log_feature_stats(name: str, X: np.ndarray) -> None:
    finite_mask = np.isfinite(X)
    finite_ratio = float(np.mean(finite_mask))
    print(
        f"[{name}] shape={X.shape} finite={finite_ratio:.3f} "
        f"min={np.nanmin(X):.4f} max={np.nanmax(X):.4f}"
    )


def log_label_stats(name: str, y: np.ndarray) -> None:
    unique, counts = np.unique(y, return_counts=True)
    dist = ", ".join([f"{u}:{c}" for u, c in zip(unique, counts)])
    print(f"[{name}] samples={len(y)} classes={len(unique)} dist={{ {dist} }}")


def sanitize_features(
    X_train: np.ndarray, X_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    low = np.percentile(X_train, 1, axis=0)
    high = np.percentile(X_train, 99, axis=0)
    X_train = np.clip(X_train, low, high)
    X_val = np.clip(X_val, low, high)
    return X_train, X_val


def compute_clip_bounds(
    X: np.ndarray, low_q: float = 1.0, high_q: float = 99.0
) -> Tuple[np.ndarray, np.ndarray]:
    low = np.percentile(X, low_q, axis=0)
    high = np.percentile(X, high_q, axis=0)
    return low, high


def apply_clip(X: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(X, low, high)


def preprocess_mlp(
    X_train: np.ndarray, X_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, Tuple[np.ndarray, np.ndarray]]:
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    low, high = compute_clip_bounds(X_train, low_q=1.0, high_q=99.0)
    X_train = apply_clip(X_train, low, high)
    X_val = apply_clip(X_val, low, high)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = np.clip(X_train, -3.0, 3.0)
    X_val = np.clip(X_val, -3.0, 3.0)
    return X_train, X_val, scaler, (low, high)


def macro_auc(y_true: np.ndarray, y_proba: np.ndarray, classes: List[int]) -> float:
    try:
        return float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=classes,
            )
        )
    except Exception:
        return float("nan")


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    classes = sorted(np.unique(y_true).tolist())
    return {
        "macro_auc": macro_auc(y_true, y_proba, classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def cv_score(
    build_model: Any,
    param_grid: List[Dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[Dict[str, Any], float, int]:
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_seed)
    best_params: Dict[str, Any] = {}
    best_score = -np.inf
    failed_folds = 0
    iterator = tqdm(param_grid, desc="CV params", unit="cfg") if tqdm else param_grid
    for params in iterator:
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            model = build_model(params)
            try:
                model.fit(X[train_idx], y[train_idx])
                y_proba = model.predict_proba(X[val_idx])
                score = macro_auc(y[val_idx], y_proba, sorted(np.unique(y).tolist()))
                if not np.isnan(score):
                    scores.append(score)
            except ValueError:
                failed_folds += 1
                continue
        mean_score = float(np.mean(scores)) if scores else float("-inf")
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    return best_params, best_score, failed_folds


def build_knn(params: Dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(**params)),
        ]
    )


def build_rf(params: Dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=42, **params)


def train_mlp_stable(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: MLPStableConfig,
) -> Tuple[MLPClassifier, Dict[str, float]]:
    model = MLPClassifier(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        solver="adam",
        activation="relu",
        learning_rate_init=cfg.lr,
        learning_rate="adaptive",
        alpha=cfg.alpha,
        max_iter=cfg.epochs,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_val)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = evaluate_metrics(y_val, y_pred, y_proba)
    return model, metrics


class MFCCDataset(Dataset):
    def __init__(self, paths: List[str], labels: np.ndarray, label_encoder: LabelEncoder, sr: int, n_mfcc: int):
        self.paths = paths
        self.labels = label_encoder.transform(labels)
        self.sr = sr
        self.n_mfcc = n_mfcc

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        path = self.paths[idx]
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        return mfcc.astype(np.float32), int(self.labels[idx])


class SimpleCNN(nn.Module):
    def __init__(self, n_mfcc: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mfcc, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)


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


def train_cnn(
    train_paths: List[str],
    train_labels: np.ndarray,
    val_paths: List[str],
    val_labels: np.ndarray,
    label_encoder: LabelEncoder,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for CNN training.")

    n_mfcc = 13
    n_classes = len(label_encoder.classes_)
    train_ds = MFCCDataset(train_paths, train_labels, label_encoder, sr=22050, n_mfcc=n_mfcc)
    val_ds = MFCCDataset(val_paths, val_labels, label_encoder, sr=22050, n_mfcc=n_mfcc)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

    model = SimpleCNN(n_mfcc=n_mfcc, n_classes=n_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

    model.eval()
    all_probs = []
    all_true = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_true.append(yb.numpy())

    y_true = np.concatenate(all_true)
    y_proba = np.vstack(all_probs)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = evaluate_metrics(y_true, y_pred, y_proba)
    return metrics, {"model": model, "label_encoder": label_encoder}


def filter_existing_audio(paths: np.ndarray, labels: np.ndarray) -> Tuple[List[str], np.ndarray]:
    out_paths: List[str] = []
    out_labels: List[str] = []
    for p, y in zip(paths, labels):
        if "::" in p:
            continue
        if Path(p).is_file():
            out_paths.append(p)
            out_labels.append(y)
    return out_paths, np.asarray(out_labels)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train KNN/RF/MLP/CNN models.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="Train .npz")
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL, help="Val .npz")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL, help="Final model path")
    parser.add_argument("--include-cnn", action="store_true", help="Enable CNN training (requires torch)")
    parser.add_argument("--cnn-epochs", type=int, default=20, help="CNN epochs")
    args = parser.parse_args()

    X_train, y_train, train_paths = load_features(args.train)
    X_val, y_val, val_paths = load_features(args.val)
    X_train, X_val = sanitize_features(X_train, X_val)
    log_feature_stats("train", X_train)
    log_feature_stats("val", X_val)
    log_label_stats("train", y_train)
    log_label_stats("val", y_val)
    print(f"[paths] train_paths={len(train_paths)} val_paths={len(val_paths)}")

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val]))
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)

    cfg = TrainConfig()
    results: Dict[str, Any] = {}

    knn_grid = [
        {"n_neighbors": k, "weights": w}
        for k in [3, 5, 7, 9]
        for w in ["uniform", "distance"]
    ]
    rf_grid = [
        {"n_estimators": n, "max_depth": d}
        for n in [100, 200]
        for d in [5, 10, None]
    ]
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    for name, builder, grid in [
        ("knn", build_knn, knn_grid),
        ("rf", build_rf, rf_grid),
    ]:
        if tqdm is None:
            print(f"Training {name}...")
        print(f"[{name}] cv_folds={cfg.n_splits} grid_size={len(grid)}")
        best_params, cv_auc, failed_folds = cv_score(builder, grid, X_train, y_train_enc, cfg)
        try:
            model = builder(best_params)
            model.fit(X_train, y_train_enc)
            y_proba = model.predict_proba(X_val)
            y_pred = np.argmax(y_proba, axis=1)
            metrics = evaluate_metrics(y_val_enc, y_pred, y_proba)
            results[name] = {"best_params": best_params, "cv_macro_auc": cv_auc, "val": metrics}
            print(
                f"[{name}] cv_macro_auc={cv_auc:.4f} "
                f"val_auc={metrics['macro_auc']:.4f} "
                f"val_acc={metrics['accuracy']:.4f} "
                f"val_f1={metrics['macro_f1']:.4f} "
                f"failed_folds={failed_folds}"
            )
        except ValueError as exc:
            results[name] = {
                "best_params": best_params,
                "cv_macro_auc": cv_auc,
                "error": str(exc),
            }
            print(f"[{name}] failed after CV: {exc} failed_folds={failed_folds}")

    mlp_cfg = MLPStableConfig()
    X_train_mlp, X_val_mlp, mlp_scaler, clip_bounds = preprocess_mlp(X_train, X_val)
    try:
        mlp_model, mlp_metrics = train_mlp_stable(
            X_train_mlp, y_train_enc, X_val_mlp, y_val_enc, mlp_cfg
        )
        results["mlp"] = {
            "config": mlp_cfg.__dict__,
            "val": mlp_metrics,
        }
        print(
            f"[mlp] val_auc={mlp_metrics['macro_auc']:.4f} "
            f"val_acc={mlp_metrics['accuracy']:.4f} "
            f"val_f1={mlp_metrics['macro_f1']:.4f}"
        )
    except ValueError as exc:
        results["mlp"] = {"config": mlp_cfg.__dict__, "error": str(exc)}
        print(f"[mlp] failed: {exc}")

    if args.include_cnn:
        train_audio, y_train_audio = filter_existing_audio(train_paths, y_train)
        val_audio, y_val_audio = filter_existing_audio(val_paths, y_val)
        if train_audio and val_audio:
            cnn_metrics, cnn_artifacts = train_cnn(
                train_audio,
                y_train_audio,
                val_audio,
                y_val_audio,
                le,
                epochs=args.cnn_epochs,
            )
            results["cnn"] = {"val": cnn_metrics}
            best_cnn = cnn_artifacts
        else:
            results["cnn"] = {"error": "No valid audio paths for CNN."}
            best_cnn = None
    else:
        best_cnn = None

    def _score(name: str) -> float:
        val_score = results.get(name, {}).get("val", {}).get("macro_auc", float("-inf"))
        return val_score if not np.isnan(val_score) else float("-inf")

    best_name = max(results.keys(), key=_score)
    results["best_model"] = best_name

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if best_name == "cnn" and best_cnn is not None:
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(best_cnn, args.model_path)
    else:
        # Retrain best sklearn model on Train+Val
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train_enc, y_val_enc])
        if best_name == "knn":
            final_model = build_knn(results["knn"]["best_params"])
            final_model.fit(X_full, y_full)
            payload = {"model": final_model, "label_encoder": le}
        elif best_name == "rf":
            final_model = build_rf(results["rf"]["best_params"])
            final_model.fit(X_full, y_full)
            payload = {"model": final_model, "label_encoder": le}
        else:
            X_full_mlp, _, mlp_scaler, clip_bounds = preprocess_mlp(X_full, X_full)
            final_model = MLPClassifier(
                hidden_layer_sizes=mlp_cfg.hidden_layer_sizes,
                solver="adam",
                activation="relu",
                learning_rate_init=mlp_cfg.lr,
                learning_rate="adaptive",
                alpha=mlp_cfg.alpha,
                max_iter=mlp_cfg.epochs,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42,
            )
            final_model.fit(X_full_mlp, y_full)
            payload = {
                "model": final_model,
                "label_encoder": le,
                "scaler": mlp_scaler,
                "clip_bounds": clip_bounds,
            }
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(payload, args.model_path)


if __name__ == "__main__":
    main()
