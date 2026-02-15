"""Harmoni ve konsonans tabanlı feature'lar."""

from __future__ import annotations

from typing import Iterable, Dict, Sequence

import numpy as np

# Basit konsonans/dissonans haritası (0–1 arası skorlar)
CONSONANCE_MAP = {
    0: 1.0,   # Unison
    1: 0.0,   # m2
    2: 0.2,   # M2
    3: 0.8,   # m3
    4: 0.8,   # M3
    5: 0.9,   # P4
    6: 0.1,   # Tritone
    7: 1.0,   # P5
    8: 0.7,   # m6
    9: 0.7,   # M6
    10: 0.3,  # m7
    11: 0.2,  # M7
}


def consonance_score(notes: Iterable[int]) -> float:
    """
    Ortalama konsonans skoru (0–1).

    1.0 = tamamen konsonant
    0.0 = tamamen dissonant
    """
    arr = np.asarray(list(notes), dtype=int)
    if arr.size < 2:
        return 0.0
    intervals = np.abs(np.diff(arr)) % 12
    scores = [CONSONANCE_MAP.get(int(i), 0.5) for i in intervals]
    return float(np.mean(scores)) if scores else 0.0


def dissonance_index(notes: Iterable[int]) -> float:
    """Dissonans indeksini (1 - konsonans) döndür."""
    return 1.0 - consonance_score(notes)


def harmonic_rhythm_variance(durations: Sequence[float]) -> float:
    """
    Basit süre varyansı metriği.

    Daha gelişmiş bir modelde gerçek akor değişimlerine bakılabilir.
    """
    if not durations:
        return 0.0
    arr = np.asarray(durations, dtype=float)
    return float(arr.std())


def extract_harmony_features(
    notes: Iterable[int],
    durations: Sequence[float] | None = None,
) -> Dict[str, float]:
    """Harmoni ile ilişkili özet feature'ları döndür."""
    feats: Dict[str, float] = {
        "consonance_score": consonance_score(notes),
        "dissonance_index": dissonance_index(notes),
    }
    if durations is not None:
        feats["duration_variance"] = harmonic_rhythm_variance(durations)
    return feats


__all__ = [
    "CONSONANCE_MAP",
    "consonance_score",
    "dissonance_index",
    "harmonic_rhythm_variance",
    "extract_harmony_features",
]

