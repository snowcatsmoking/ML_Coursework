"""Notebook-friendly feature extraction for hum/whistle audio classification.

This file mirrors `features.py` but is organized with section comments and
step-by-step notes so you can paste blocks into a Jupyter notebook.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------
# 0) Environment setup (keeps CPU threading stable inside notebooks)
# ---------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import librosa
import numpy as np
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------
# 1) Paths and constants (edit for your machine)
# ---------------------------------------------------------------------
DATASET_DIR = Path("/Users/panmingh/Code/ML_Coursework/Data/MLEndHWII_sample_800")
DEFAULT_OUTPUT_DIR = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results")

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


# ---------------------------------------------------------------------
# 2) Configuration object
# ---------------------------------------------------------------------
@dataclass
class FeatureConfig:
    sr: int = 22050
    n_mfcc: int = 13
    hop_length: int = 512
    fmin: float = librosa.note_to_hz("C2")
    fmax: float = librosa.note_to_hz("C7")
    onset_backtrack: bool = True


# ---------------------------------------------------------------------
# 3) Small numeric helpers
# ---------------------------------------------------------------------
def _safe_stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(np.mean(x)), float(np.std(x)), float(np.max(x)), float(np.min(x))


def _safe_mean_std(x: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    return float(np.mean(x)), float(np.std(x))


def _nan_to_num(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------
# 4) Audio loading
# ---------------------------------------------------------------------
def load_audio(path: str, cfg: FeatureConfig) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=cfg.sr, mono=True)
    if y.size == 0:
        return np.zeros(1, dtype=np.float32), cfg.sr
    return y, sr


# ---------------------------------------------------------------------
# 5) MFCC + delta features
# ---------------------------------------------------------------------
def mfcc_features(y: np.ndarray, cfg: FeatureConfig) -> List[float]:
    mfcc = librosa.feature.mfcc(
        y=y, sr=cfg.sr, n_mfcc=cfg.n_mfcc, hop_length=cfg.hop_length
    )
    delta = librosa.feature.delta(mfcc)

    mfcc = _nan_to_num(mfcc)
    delta = _nan_to_num(delta)

    feats: List[float] = []
    feats.extend(np.mean(mfcc, axis=1).tolist())
    feats.extend(np.std(mfcc, axis=1).tolist())
    feats.extend(np.mean(delta, axis=1).tolist())
    feats.extend(np.std(delta, axis=1).tolist())
    return feats


# ---------------------------------------------------------------------
# 6) F0 / pitch contour features (pyin)
# ---------------------------------------------------------------------
def f0_features(y: np.ndarray, cfg: FeatureConfig) -> List[float]:
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        sr=cfg.sr,
        hop_length=cfg.hop_length,
    )
    f0 = _nan_to_num(f0)
    voiced_flag = voiced_flag.astype(np.float32) if voiced_flag is not None else None

    voiced_ratio = 0.0
    if voiced_flag is not None and voiced_flag.size > 0:
        voiced_ratio = float(np.mean(voiced_flag))

    f0_mean, f0_std = _safe_mean_std(f0)
    if f0_std > 0:
        f0_norm = (f0 - f0_mean) / f0_std
    else:
        f0_norm = f0 - f0_mean

    f0n_mean, f0n_std = _safe_mean_std(f0_norm)
    f0n_max, f0n_min = (float(np.max(f0_norm)), float(np.min(f0_norm))) if f0_norm.size else (0.0, 0.0)
    f0n_median = float(np.median(f0_norm)) if f0_norm.size else 0.0

    # Interval in semitones between adjacent frames.
    f0_nonzero = np.where(f0 > 0, f0, np.nan)
    intervals = 12.0 * np.log2(f0_nonzero[1:] / f0_nonzero[:-1])
    intervals = _nan_to_num(intervals)

    int_mean, int_std, int_max, int_min = _safe_stats(intervals)
    int_median = float(np.median(intervals)) if intervals.size else 0.0
    int_iqr = float(np.percentile(intervals, 75) - np.percentile(intervals, 25)) if intervals.size else 0.0
    int_abs_mean = float(np.mean(np.abs(intervals))) if intervals.size else 0.0
    int_abs_std = float(np.std(np.abs(intervals))) if intervals.size else 0.0
    int_pos_ratio = float(np.mean(intervals > 0)) if intervals.size else 0.0
    int_neg_ratio = float(np.mean(intervals < 0)) if intervals.size else 0.0

    # Melodic contour proportions.
    eps = 1e-4
    up_ratio = float(np.mean(intervals > eps)) if intervals.size else 0.0
    down_ratio = float(np.mean(intervals < -eps)) if intervals.size else 0.0
    flat_ratio = float(np.mean(np.abs(intervals) <= eps)) if intervals.size else 0.0

    feats = [
        f0n_mean,
        f0n_std,
        f0n_max,
        f0n_min,
        f0n_median,
        int_mean,
        int_std,
        int_max,
        int_min,
        int_median,
        int_iqr,
        int_abs_mean,
        int_abs_std,
        int_pos_ratio,
        int_neg_ratio,
        up_ratio,
        down_ratio,
        flat_ratio,
        voiced_ratio,
        float(np.mean(_nan_to_num(voiced_prob))) if voiced_prob is not None else 0.0,
    ]
    return feats


# ---------------------------------------------------------------------
# 7) Rhythm / onset features
# ---------------------------------------------------------------------
def _estimate_tempo(onset_env: np.ndarray, cfg: FeatureConfig) -> float:
    if onset_env.size < 2:
        return 0.0
    onset_env = onset_env - np.mean(onset_env)
    if np.allclose(onset_env, 0.0):
        return 0.0
    ac = np.correlate(onset_env, onset_env, mode="full")[onset_env.size - 1 :]
    min_bpm, max_bpm = 30.0, 240.0
    min_lag = int((60.0 * cfg.sr) / (max_bpm * cfg.hop_length))
    max_lag = int((60.0 * cfg.sr) / (min_bpm * cfg.hop_length))
    min_lag = max(min_lag, 1)
    max_lag = min(max_lag, ac.size - 1)
    if max_lag <= min_lag:
        return 0.0
    lag = int(np.argmax(ac[min_lag : max_lag + 1]) + min_lag)
    return float(60.0 * cfg.sr / (cfg.hop_length * lag))


def rhythm_features(y: np.ndarray, cfg: FeatureConfig) -> List[float]:
    onset_env = librosa.onset.onset_strength(y=y, sr=cfg.sr, hop_length=cfg.hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=cfg.sr,
        hop_length=cfg.hop_length,
        backtrack=cfg.onset_backtrack,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=cfg.sr, hop_length=cfg.hop_length)
    ioi = np.diff(onset_times)
    ioi_mean, ioi_std, ioi_max, ioi_min = _safe_stats(ioi)

    tempo = _estimate_tempo(onset_env, cfg)
    duration = float(len(y)) / float(cfg.sr) if cfg.sr > 0 else 0.0
    onsets_per_sec = float(len(onset_times) / duration) if duration > 0 else 0.0

    return [
        ioi_mean,
        ioi_std,
        ioi_max,
        ioi_min,
        tempo,
        onsets_per_sec,
    ]


# ---------------------------------------------------------------------
# 8) Full feature vector helpers
# ---------------------------------------------------------------------
def extract_features(path: str, cfg: FeatureConfig | None = None) -> np.ndarray:
    cfg = cfg or FeatureConfig()
    y, _ = load_audio(path, cfg)
    feats: List[float] = []
    feats.extend(mfcc_features(y, cfg))
    feats.extend(f0_features(y, cfg))
    feats.extend(rhythm_features(y, cfg))
    return np.asarray(feats, dtype=np.float32)


def batch_extract(
    paths: Iterable[str], cfg: FeatureConfig | None = None
) -> Tuple[np.ndarray, List[str]]:
    cfg = cfg or FeatureConfig()
    features: List[np.ndarray] = []
    ok_paths: List[str] = []
    path_list = list(paths)
    if tqdm is not None:
        iterator = tqdm(path_list, desc="Extracting features", unit="file")
    else:
        iterator = path_list
    for idx, p in enumerate(iterator, start=1):
        feats = extract_features(p, cfg)
        features.append(feats)
        ok_paths.append(p)
        if tqdm is None and idx % 50 == 0:
            print(f"Processed {idx}/{len(path_list)} files...")
    return np.vstack(features), ok_paths


def feature_names(cfg: FeatureConfig | None = None) -> List[str]:
    cfg = cfg or FeatureConfig()
    names: List[str] = []
    for i in range(cfg.n_mfcc):
        names.append(f"mfcc_mean_{i+1}")
    for i in range(cfg.n_mfcc):
        names.append(f"mfcc_std_{i+1}")
    for i in range(cfg.n_mfcc):
        names.append(f"mfcc_delta_mean_{i+1}")
    for i in range(cfg.n_mfcc):
        names.append(f"mfcc_delta_std_{i+1}")

    names.extend(
        [
            "f0n_mean",
            "f0n_std",
            "f0n_max",
            "f0n_min",
            "f0n_median",
            "interval_mean",
            "interval_std",
            "interval_max",
            "interval_min",
            "interval_median",
            "interval_iqr",
            "interval_abs_mean",
            "interval_abs_std",
            "interval_pos_ratio",
            "interval_neg_ratio",
            "contour_up_ratio",
            "contour_down_ratio",
            "contour_flat_ratio",
            "voiced_ratio",
            "voiced_prob_mean",
        ]
    )

    names.extend(
        [
            "ioi_mean",
            "ioi_std",
            "ioi_max",
            "ioi_min",
            "tempo",
            "onsets_per_sec",
        ]
    )
    return names


def as_dict(path: str, cfg: FeatureConfig | None = None) -> Dict[str, float]:
    cfg = cfg or FeatureConfig()
    feats = extract_features(path, cfg)
    names = feature_names(cfg)
    return {k: float(v) for k, v in zip(names, feats)}


# ---------------------------------------------------------------------
# 9) Dataset helpers (metadata + saving)
# ---------------------------------------------------------------------
def _parse_metadata(path: Path) -> Dict[str, str]:
    stem = path.stem
    parts = stem.split("_")
    meta = {"subject": "", "mode": "", "take": "", "song": ""}
    if len(parts) >= 4:
        meta["subject"] = parts[0]
        meta["mode"] = parts[1]
        meta["take"] = parts[2]
        meta["song"] = "_".join(parts[3:])
    else:
        meta["song"] = stem
    return meta


def collect_wav_paths(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.glob("*.wav") if p.is_file()])


def save_features(
    features: np.ndarray,
    paths: List[str],
    names: List[str],
    meta: List[Dict[str, str]],
    out_dir: Path,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{prefix}.npz"
    csv_path = out_dir / f"{prefix}.csv"

    labels = [m.get("song", "") for m in meta]
    np.savez_compressed(
        npz_path,
        X=features,
        labels=np.asarray(labels),
        paths=np.asarray(paths),
        feature_names=np.asarray(names),
        subject=np.asarray([m.get("subject", "") for m in meta]),
        mode=np.asarray([m.get("mode", "") for m in meta]),
        take=np.asarray([m.get("take", "") for m in meta]),
    )

    header = ["path", "label", "subject", "mode", "take"] + names
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for row_idx, p in enumerate(paths):
            row = [
                p,
                labels[row_idx],
                meta[row_idx].get("subject", ""),
                meta[row_idx].get("mode", ""),
                meta[row_idx].get("take", ""),
            ]
            feat_str = [f"{v:.8f}" for v in features[row_idx].tolist()]
            f.write(",".join(row + feat_str) + "\n")


# ---------------------------------------------------------------------
# 10) Example "notebook flow" (copy into separate cells if desired)
# ---------------------------------------------------------------------
# # NOTE: We already extracted and saved the features offline, so re-running
# this block is unnecessary and can be time-consuming.
# cfg = FeatureConfig()
# wav_paths = collect_wav_paths(DATASET_DIR)
# features, ok_paths = batch_extract([str(p) for p in wav_paths], cfg)
# meta = [_parse_metadata(Path(p)) for p in ok_paths]
# names = feature_names(cfg)
# save_features(features, ok_paths, names, meta, DEFAULT_OUTPUT_DIR, "features")
