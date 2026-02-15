"""Interval (nota aralıkları) tabanlı feature'lar."""

from __future__ import annotations

from typing import Iterable, Dict

import numpy as np

from music_math.features.utils import safe_entropy


def extract_intervals(notes: Iterable[int]) -> np.ndarray:
    """Ardışık notalar arası fark (semitone)."""
    arr = np.asarray(list(notes), dtype=float)
    if arr.size < 2:
        return np.array([], dtype=float)
    return np.diff(arr)


def interval_entropy(notes: Iterable[int]) -> float:
    """
    Interval dağılımının entropisi.
    Bach: Küçük adımlar → düşük entropi
    Liszt: Büyük atlamalar → yüksek entropi
    """
    intervals = extract_intervals(notes)
    if intervals.size == 0:
        return 0.0
    intervals = np.clip(intervals, -12, 12)
    hist, _ = np.histogram(intervals, bins=25, range=(-12.5, 12.5))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.0
    hist /= total
    return safe_entropy(hist)


def directional_bias(notes: Iterable[int]) -> float:
    """
    Yükselen / alçalan nota tercihi.

    +1.0 = tamamen yükselen
    -1.0 = tamamen alçalan
    0.0  = dengeli
    """
    intervals = extract_intervals(notes)
    if intervals.size == 0:
        return 0.0
    ascending = np.sum(intervals > 0)
    descending = np.sum(intervals < 0)
    total = ascending + descending
    if total == 0:
        return 0.0
    return float((ascending - descending) / total)


def step_ratio(notes: Iterable[int]) -> float:
    """Küçük adım (1–2 semitone) oranı."""
    intervals = np.abs(extract_intervals(notes))
    if intervals.size == 0:
        return 0.0
    return float(np.sum(intervals <= 2) / intervals.size)


def leap_ratio(notes: Iterable[int]) -> float:
    """Büyük atlama (>4 semitone) oranı."""
    intervals = np.abs(extract_intervals(notes))
    if intervals.size == 0:
        return 0.0
    return float(np.sum(intervals > 4) / intervals.size)


def mean_interval_size(notes: Iterable[int]) -> float:
    """Ortalama interval büyüklüğü (mutlak değer)."""
    intervals = np.abs(extract_intervals(notes))
    if intervals.size == 0:
        return 0.0
    return float(intervals.mean())


def interval_transition_matrix(notes: Iterable[int], normalize: bool = True) -> np.ndarray:
    """
    12x12 pitch-class geçiş matrisi.

    M[i, j] = i pitch-class'tan j pitch-class'a geçiş olasılığı.
    """
    pcs = [int(n) % 12 for n in notes]
    if len(pcs) < 2:
        return np.zeros((12, 12), dtype=float)

    matrix = np.zeros((12, 12), dtype=float)
    for i in range(len(pcs) - 1):
        matrix[pcs[i], pcs[i + 1]] += 1.0

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums

    return matrix


def extract_interval_features(notes: Iterable[int]) -> Dict[str, float]:
    """Interval bazlı özet feature'lar."""
    intervals = extract_intervals(notes)
    abs_intervals = np.abs(intervals)

    features: Dict[str, float] = {
        "interval_entropy": interval_entropy(notes),
        "directional_bias": directional_bias(notes),
        "step_ratio": step_ratio(notes),
        "leap_ratio": leap_ratio(notes),
        "mean_interval": mean_interval_size(notes),
        "interval_std": float(abs_intervals.std()) if abs_intervals.size > 0 else 0.0,
    }
    return features


__all__ = [
    "extract_intervals",
    "interval_entropy",
    "directional_bias",
    "step_ratio",
    "leap_ratio",
    "mean_interval_size",
    "interval_transition_matrix",
    "extract_interval_features",
]

