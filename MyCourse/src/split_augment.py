"""Split by subject, augment train only, and extract augmented features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import augment
import features

DATASET_DIR = Path("/Users/panmingh/Code/ML_Coursework/Data/MLEndHWII_sample_800")
DEFAULT_FEATURES = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/features.npz")
DEFAULT_OUT_DIR = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/results/splits")

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


def load_features_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def save_features_npz(
    out_path: Path,
    X: np.ndarray,
    labels: np.ndarray,
    paths: np.ndarray,
    feature_names: np.ndarray,
    subject: np.ndarray,
    mode: np.ndarray,
    take: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        labels=labels,
        paths=paths,
        feature_names=feature_names,
        subject=subject,
        mode=mode,
        take=take,
    )


def split_by_subject(
    subjects: np.ndarray, cfg: SplitConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    gss = GroupShuffleSplit(
        n_splits=1, train_size=cfg.train_ratio, random_state=cfg.seed
    )
    idx = np.arange(subjects.shape[0])
    train_idx, temp_idx = next(gss.split(idx, groups=subjects))

    temp_subjects = subjects[temp_idx]
    val_ratio = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_ratio, random_state=cfg.seed)
    val_sub_idx, test_sub_idx = next(gss2.split(temp_idx, groups=temp_subjects))
    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]
    return train_idx, val_idx, test_idx


def extract_features_from_audio(y: np.ndarray, cfg: features.FeatureConfig) -> np.ndarray:
    feats: List[float] = []
    feats.extend(features.mfcc_features(y, cfg))
    feats.extend(features.f0_features(y, cfg))
    feats.extend(features.rhythm_features(y, cfg))
    return np.asarray(feats, dtype=np.float32)


def augment_and_extract(
    paths: List[str],
    subjects: np.ndarray,
    modes: np.ndarray,
    takes: np.ndarray,
    cfg_feat: features.FeatureConfig,
    cfg_aug: augment.AugmentConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    feats_list: List[np.ndarray] = []
    labels_list: List[str] = []
    paths_list: List[str] = []
    subj_list: List[str] = []
    mode_list: List[str] = []
    take_list: List[str] = []

    iterator = tqdm(paths, desc="Augmenting train", unit="file") if tqdm else paths
    for i, p in enumerate(iterator):
        y = augment.load_audio(p, cfg_aug)
        variants = augment.augment_sample(y, cfg_aug, rng)
        base_label = Path(p).stem.split("_", 3)[-1]
        for j, y_aug in enumerate(variants):
            feats_list.append(extract_features_from_audio(y_aug, cfg_feat))
            labels_list.append(base_label)
            paths_list.append(f"{p}::aug{j+1}")
            subj_list.append(str(subjects[i]))
            mode_list.append(str(modes[i]))
            take_list.append(str(takes[i]))
        if tqdm is None and (i + 1) % 50 == 0:
            print(f"Augmented {i+1}/{len(paths)} files...")

    return (
        np.vstack(feats_list) if feats_list else np.zeros((0, 0), dtype=np.float32),
        np.asarray(labels_list),
        np.asarray(paths_list),
        np.asarray(subj_list),
        np.asarray(mode_list),
        np.asarray(take_list),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Split by subject, augment train only, and extract augmented features."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Path to features.npz produced by features.py",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory with original .wav files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to save split/augmented features.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data = load_features_npz(args.features)
    X = data["X"]
    labels = data["labels"].astype(str)
    paths = data["paths"].astype(str)
    feature_names = data["feature_names"]
    subject = data.get("subject")
    mode = data.get("mode")
    take = data.get("take")

    if subject is None or mode is None or take is None:
        raise ValueError("features.npz missing subject/mode/take; re-run features.py.")

    split_cfg = SplitConfig(seed=args.seed)
    train_idx, val_idx, test_idx = split_by_subject(subject, split_cfg)

    save_features_npz(
        args.out_dir / "train.npz",
        X[train_idx],
        labels[train_idx],
        paths[train_idx],
        feature_names,
        subject[train_idx],
        mode[train_idx],
        take[train_idx],
    )
    save_features_npz(
        args.out_dir / "val.npz",
        X[val_idx],
        labels[val_idx],
        paths[val_idx],
        feature_names,
        subject[val_idx],
        mode[val_idx],
        take[val_idx],
    )
    save_features_npz(
        args.out_dir / "test.npz",
        X[test_idx],
        labels[test_idx],
        paths[test_idx],
        feature_names,
        subject[test_idx],
        mode[test_idx],
        take[test_idx],
    )

    cfg_feat = features.FeatureConfig()
    cfg_aug = augment.AugmentConfig()
    train_paths = [str(args.audio_dir / Path(p).name) for p in paths[train_idx]]
    (
        X_aug,
        y_aug,
        p_aug,
        s_aug,
        m_aug,
        t_aug,
    ) = augment_and_extract(
        train_paths,
        subject[train_idx],
        mode[train_idx],
        take[train_idx],
        cfg_feat,
        cfg_aug,
        args.seed,
    )

    save_features_npz(
        args.out_dir / "train_aug.npz",
        X_aug,
        y_aug,
        p_aug,
        feature_names,
        s_aug,
        m_aug,
        t_aug,
    )

    X_train_full = np.vstack([X[train_idx], X_aug]) if X_aug.size else X[train_idx]
    y_train_full = np.concatenate([labels[train_idx], y_aug]) if y_aug.size else labels[train_idx]
    p_train_full = np.concatenate([paths[train_idx], p_aug]) if p_aug.size else paths[train_idx]
    s_train_full = np.concatenate([subject[train_idx], s_aug]) if s_aug.size else subject[train_idx]
    m_train_full = np.concatenate([mode[train_idx], m_aug]) if m_aug.size else mode[train_idx]
    t_train_full = np.concatenate([take[train_idx], t_aug]) if t_aug.size else take[train_idx]

    save_features_npz(
        args.out_dir / "train_full.npz",
        X_train_full,
        y_train_full,
        p_train_full,
        feature_names,
        s_train_full,
        m_train_full,
        t_train_full,
    )


if __name__ == "__main__":
    main()
