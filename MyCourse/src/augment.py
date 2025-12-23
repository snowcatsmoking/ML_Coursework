"""Audio data augmentation for hum/whistle classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import librosa
import numpy as np
from scipy.io import wavfile

DATASET_DIR = Path("/Users/panmingh/Code/ML_Coursework/Data/MLEndHWII_sample_800")
DEFAULT_OUTPUT_DIR = Path("/Users/panmingh/Code/ML_Coursework/MyCourse/data/augmented")


@dataclass
class AugmentConfig:
    sr: int = 22050
    pitch_shift_steps: Tuple[int, int] = (-3, 3)
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    snr_db_range: Tuple[float, float] = (20.0, 40.0)


def load_audio(path: str | Path, cfg: AugmentConfig) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=cfg.sr, mono=True)
    if y.size == 0:
        return np.zeros(1, dtype=np.float32)
    return y.astype(np.float32, copy=False)


def pitch_shift(y: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    steps = rng.integers(cfg.pitch_shift_steps[0], cfg.pitch_shift_steps[1] + 1)
    if steps == 0:
        return y
    return librosa.effects.pitch_shift(y, sr=cfg.sr, n_steps=int(steps))


def time_stretch(y: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    rate = rng.uniform(cfg.time_stretch_range[0], cfg.time_stretch_range[1])
    if np.isclose(rate, 1.0):
        return y
    return librosa.effects.time_stretch(y, rate=float(rate))


def add_noise(y: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    snr_db = rng.uniform(cfg.snr_db_range[0], cfg.snr_db_range[1])
    if y.size == 0:
        return y
    signal_power = np.mean(y**2) + 1e-12
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape).astype(np.float32)
    return y + noise


def augment_sample(
    y: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator
) -> List[np.ndarray]:
    variants = []
    variants.append(pitch_shift(y, cfg, rng))
    variants.append(time_stretch(y, cfg, rng))
    variants.append(add_noise(y, cfg, rng))
    return variants


def augment_path(
    path: str | Path,
    cfg: AugmentConfig | None = None,
    seed: int | None = None,
) -> List[np.ndarray]:
    cfg = cfg or AugmentConfig()
    rng = np.random.default_rng(seed)
    y = load_audio(path, cfg)
    return augment_sample(y, cfg, rng)


def batch_augment(
    paths: Iterable[str | Path],
    cfg: AugmentConfig | None = None,
    seed: int | None = None,
) -> List[Tuple[Path, List[np.ndarray]]]:
    cfg = cfg or AugmentConfig()
    rng = np.random.default_rng(seed)
    out: List[Tuple[Path, List[np.ndarray]]] = []
    for p in paths:
        path = Path(p)
        y = load_audio(path, cfg)
        out.append((path, augment_sample(y, cfg, rng)))
    return out


def _normalize_to_int16(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return np.zeros(1, dtype=np.int16)
    peak = np.max(np.abs(y))
    if peak <= 0:
        return np.zeros_like(y, dtype=np.int16)
    y = y / peak
    return (y * 32767.0).astype(np.int16)


def save_augmented(
    src_path: Path,
    variants: List[np.ndarray],
    out_dir: Path,
    cfg: AugmentConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src_path.stem
    suffixes = ["ps", "ts", "noise"]
    for idx, y_aug in enumerate(variants):
        tag = suffixes[idx] if idx < len(suffixes) else f"aug{idx+1}"
        out_path = out_dir / f"{stem}_{tag}.wav"
        wavfile.write(out_path, cfg.sr, _normalize_to_int16(y_aug))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Augment audio dataset with pitch/tempo/noise.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory with .wav files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save augmented .wav files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for augmentation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of files to augment (0 = all).",
    )
    args = parser.parse_args()

    wav_paths = sorted([p for p in args.data_dir.glob("*.wav") if p.is_file()])
    if args.limit > 0:
        wav_paths = wav_paths[: args.limit]
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in {args.data_dir}")

    cfg = AugmentConfig()
    rng = np.random.default_rng(args.seed)
    for idx, p in enumerate(wav_paths, start=1):
        y = load_audio(p, cfg)
        variants = augment_sample(y, cfg, rng)
        save_augmented(p, variants, args.output_dir, cfg)
        if idx % 50 == 0:
            print(f"Augmented {idx}/{len(wav_paths)} files...")


if __name__ == "__main__":
    main()
