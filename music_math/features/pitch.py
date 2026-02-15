"""Pitch (ses yüksekliği) tabanlı feature'lar."""

from __future__ import annotations

from typing import Iterable, Dict

import numpy as np

from music_math.features.utils import safe_entropy


def pitch_class_histogram(notes: Iterable[int]) -> np.ndarray:
    """12-bin normalize pitch-class histogramı (mod 12)."""
    pcs = [int(n) % 12 for n in notes]
    if not pcs:
        return np.zeros(12, dtype=float)
    hist = np.bincount(pcs, minlength=12).astype(float)
    total = hist.sum()
    if total == 0:
        return np.zeros(12, dtype=float)
    return hist / total


def pitch_entropy(notes: Iterable[int]) -> float:
    """
    Pitch-class dağılımının Shannon entropisi (bit cinsinden).
    Yüksek değer = çok çeşitli nota kullanımı.
    """
    hist = pitch_class_histogram(notes)
    return safe_entropy(hist)


def tonal_center_strength(notes: Iterable[int]) -> float:
    """
    En baskın pitch-class oranı.
    Yüksek değer = güçlü tonal merkez.
    """
    hist = pitch_class_histogram(notes)
    return float(hist.max()) if hist.size > 0 else 0.0


def pitch_range(notes: Iterable[int]) -> int:
    """Kullanılan pitch aralığı (semitone)."""
    arr = np.asarray(list(notes), dtype=int)
    if arr.size == 0:
        return 0
    return int(arr.max() - arr.min())


def pitch_mean(notes: Iterable[int]) -> float:
    """Ortalama pitch (merkez)."""
    arr = np.asarray(list(notes), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def pitch_std(notes: Iterable[int]) -> float:
    """Pitch standart sapması."""
    arr = np.asarray(list(notes), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.std())


def chromatic_saturation(notes: Iterable[int]) -> int:
    """
    Kullanılan farklı pitch-class sayısı.

    12'ye yakın değerler kromatik zenginlik, küçük değerler ise
    modal/tonal sadelik göstergesi olabilir.
    """
    pcs = {int(n) % 12 for n in notes}
    return len(pcs)


def extract_pitch_features(notes: Iterable[int]) -> Dict[str, float]:
    """Tüm pitch tabanlı feature'ları tek dict olarak döndür."""
    notes_list = list(notes)
    hist = pitch_class_histogram(notes_list)

    features: Dict[str, float] = {
        "pitch_entropy": pitch_entropy(notes_list),
        "tonal_center_strength": tonal_center_strength(notes_list),
        "pitch_range": float(pitch_range(notes_list)),
        "pitch_mean": pitch_mean(notes_list),
        "pitch_std": pitch_std(notes_list),
        "chromatic_saturation": float(chromatic_saturation(notes_list)),
    }

    for i in range(12):
        features[f"pc_{i}"] = float(hist[i])

    return features


__all__ = [
    "pitch_class_histogram",
    "pitch_entropy",
    "tonal_center_strength",
    "pitch_range",
    "pitch_mean",
    "pitch_std",
    "chromatic_saturation",
    "extract_pitch_features",
]

