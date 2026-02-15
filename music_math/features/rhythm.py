"""Ritim ve tempo ile ilişkili feature'lar."""

from __future__ import annotations

from typing import Iterable, Dict, Sequence

import numpy as np

from music_math.features.utils import safe_entropy
from music_math.core.types import NoteEvent


def rhythmic_entropy(durations: Sequence[float]) -> float:
    """
    Nota sürelerinin entropisi.

    Yüksek = çok çeşitli ritmik değerler,
    Düşük = daha tekdüze ritim.
    """
    if not durations:
        return 0.0
    # 1/16'lık birimlere kabaca kuantize et
    quantized = np.round(np.asarray(durations, dtype=float) * 4) / 4
    unique, counts = np.unique(quantized, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    return safe_entropy(probs)


def note_density(events: Iterable[NoteEvent], total_duration: float) -> float:
    """Birim zamandaki nota sayısı."""
    events_list = list(events)
    if total_duration <= 0 or not events_list:
        return 0.0
    return float(len(events_list) / total_duration)


def syncopation_estimate(durations: Sequence[float]) -> float:
    """
    Senkopasyon tahmini için basit bir metrik: süre varyansına bakar.
    """
    if not durations:
        return 0.0
    arr = np.asarray(durations, dtype=float)
    mean = float(arr.mean())
    if mean == 0:
        return 0.0
    return float(arr.std() / (mean + 1e-8))


def tempo_variance(events: Iterable[NoteEvent]) -> float:
    """
    Inter-Onset Interval (IOI) varyansı.

    Yüksek = tempo değişkenliği (rubato),
    Düşük = daha sabit tempo.
    """
    events_list = list(events)
    if len(events_list) < 2:
        return 0.0
    starts = sorted(e.start for e in events_list)
    iois = np.diff(starts)
    mean = float(iois.mean())
    if mean == 0:
        return 0.0
    return float(iois.std() / (mean + 1e-8))


def extract_rhythm_features(events: Iterable[NoteEvent]) -> Dict[str, float]:
    """NoteEvent serisinden ritim/tempo ile ilgili feature'ları çıkar."""
    events_list = list(events)
    if not events_list:
        return {
            "rhythmic_entropy": 0.0,
            "note_density": 0.0,
            "syncopation_estimate": 0.0,
            "tempo_variance": 0.0,
            "duration_mean": 0.0,
            "duration_std": 0.0,
        }

    durations = [e.duration for e in events_list]
    starts = [e.start for e in events_list]
    total_dur = max(starts) + durations[-1] if starts else 0.0

    arr_dur = np.asarray(durations, dtype=float)

    return {
        "rhythmic_entropy": rhythmic_entropy(durations),
        "note_density": note_density(events_list, total_dur),
        "syncopation_estimate": syncopation_estimate(durations),
        "tempo_variance": tempo_variance(events_list),
        "duration_mean": float(arr_dur.mean()),
        "duration_std": float(arr_dur.std()),
    }


__all__ = [
    "rhythmic_entropy",
    "note_density",
    "syncopation_estimate",
    "tempo_variance",
    "extract_rhythm_features",
]

