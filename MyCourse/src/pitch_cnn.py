"""Pitch-contour extraction + 1D-CNN training (single-script pipeline)."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None

DATASET_DIR = Path("/Users/panmingh/Code/ML_Coursework/Data/MLEndHWII_sample_800")
OUTPUT_DIR = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/pitch_cnn")

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class PitchConfig:
    sr: int = 22050
    hop_hz: int = 64
    target_seconds: float = 20.0
    fmin: float = librosa.note_to_hz("C2")
    fmax: float = librosa.note_to_hz("C7")
    frame_length: int = 2048

    @property
    def hop_length(self) -> int:
        return max(1, int(self.sr / self.hop_hz))

    @property
    def target_len(self) -> int:
        return int(self.target_seconds * self.hop_hz)


def list_wavs(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.glob("*.wav") if p.is_file()])


def parse_label(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 4:
        return "_".join(parts[3:])
    return path.stem


def _normalize_pitch(f0: np.ndarray) -> np.ndarray:
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced = f0 > 0
    if not np.any(voiced):
        return np.zeros_like(f0, dtype=np.float32)
    log_f0 = np.zeros_like(f0, dtype=np.float32)
    log_f0[voiced] = np.log2(f0[voiced])
    mean = np.mean(log_f0[voiced])
    std = np.std(log_f0[voiced])
    if std > 0:
        log_f0[voiced] = (log_f0[voiced] - mean) / std
    else:
        log_f0[voiced] = log_f0[voiced] - mean
    log_f0 = np.clip(log_f0, -3.0, 3.0)
    return log_f0.astype(np.float32)


def extract_pitch_vector(path: Path, cfg: PitchConfig) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=cfg.sr, mono=True)
    if y.size == 0:
        return np.zeros(cfg.target_len, dtype=np.float32)
    f0 = librosa.yin(
        y,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        sr=cfg.sr,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
    )
    f0 = _normalize_pitch(f0)
    duration = len(y) / float(cfg.sr)
    if duration <= 0:
        return np.zeros(cfg.target_len, dtype=np.float32)
    times = np.arange(f0.shape[0]) * (cfg.hop_length / float(cfg.sr))
    t_norm = times / max(duration, 1e-6)
    target_t = np.linspace(0.0, 1.0, cfg.target_len)
    vec = np.interp(target_t, t_norm, f0).astype(np.float32)
    return vec


def save_plots(
    vec: np.ndarray, label: str, out_dir: Path, stem: str, cfg: PitchConfig
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # "Spectrogram-like" pitch map.
    pitch_map = np.tile(vec[np.newaxis, :], (50, 1))
    fig, ax = plt.subplots(figsize=(6, 2))
    im = ax.imshow(pitch_map, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(f"Pitch Map - {label}")
    ax.set_xlabel("Time (norm)")
    ax.set_ylabel("Pitch (norm)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_pitch_map.png", dpi=150)
    plt.close(fig)

    # Bar chart of the vector.
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.bar(np.arange(cfg.target_len), vec, width=1.0)
    ax.set_title(f"Pitch Vector - {label}")
    ax.set_xlabel("Frame (8 Hz)")
    ax.set_ylabel("Pitch (norm)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_pitch_bar.png", dpi=150)
    plt.close(fig)


class PitchDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class PitchCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = np.argmax(y_prob, axis=1)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        out["macro_auc"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except Exception:
        out["macro_auc"] = float("nan")
    return out


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> Tuple[PitchCNN, dict]:
    model = PitchCNN(n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = PitchDataset(X_train, y_train)
    val_ds = PitchDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

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
    y_prob = np.vstack(all_probs)
    y_true = np.concatenate(all_true)
    metrics = evaluate(y_true, y_prob)
    return model, metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pitch vector + 1D-CNN pipeline.")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR, help="Audio dir.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output dir.")
    parser.add_argument("--plot-limit", type=int, default=5, help="Number of plots (0 = all).")
    parser.add_argument("--epochs", type=int, default=20, help="CNN epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--reuse", action="store_true", help="Reuse cached features if present.")
    args = parser.parse_args()

    if torch is None:
        raise RuntimeError("PyTorch is required for the 1D-CNN training.")

    cfg = PitchConfig()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.out_dir / "pitch_vectors.npz"

    if args.reuse and feature_path.exists():
        data = np.load(feature_path, allow_pickle=True)
        X = data["X"]
        labels = data["labels"].astype(str)
        paths = data["paths"].astype(str)
    else:
        wavs = list_wavs(args.data_dir)
        if not wavs:
            raise FileNotFoundError(f"No .wav files found in {args.data_dir}")
        X_list = []
        labels = []
        paths = []
        iterator = tqdm(wavs, desc="Extracting pitch vectors", unit="file") if tqdm else wavs
        for p in iterator:
            vec = extract_pitch_vector(p, cfg)
            X_list.append(vec)
            labels.append(parse_label(p))
            paths.append(str(p))
        X = np.stack(X_list, axis=0)
        labels = np.asarray(labels)
        paths = np.asarray(paths)
        np.savez_compressed(feature_path, X=X, labels=labels, paths=paths)

    # Plot pitch maps/bars
    plot_dir = args.out_dir / "plots"
    limit = len(X) if args.plot_limit == 0 else min(args.plot_limit, len(X))
    for i in range(limit):
        stem = Path(paths[i]).stem
        save_plots(X[i], labels[i], plot_dir, stem, cfg)

    # Prepare data for CNN
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X = X[:, np.newaxis, :]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, val_metrics = train_cnn(
        X_train,
        y_train,
        X_val,
        y_val,
        n_classes=len(le.classes_),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # Test metrics
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float().to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    test_metrics = evaluate(y_test, probs)

    metrics = {"val": val_metrics, "test": test_metrics}
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    torch.save(
        {"state_dict": model.state_dict(), "classes": le.classes_.tolist()},
        args.out_dir / "pitch_cnn.pt",
    )


if __name__ == "__main__":
    main()
