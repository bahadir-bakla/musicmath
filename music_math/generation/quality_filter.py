"""Üretilen müzikler için otomatik kalite filtresi."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from music_math.core.types import NoteEvent
from music_math.features.interval import extract_intervals
from music_math.features.pitch import pitch_entropy


@dataclass
class QualityConfig:
    """Kalite skorunu oluşturan ağırlık ve eşik değerleri."""

    entropy_ranges: Dict[str, Tuple[float, float]] = None  # type: ignore[assignment]
    threshold: float = 0.65

    def __post_init__(self) -> None:
        if self.entropy_ranges is None:
            self.entropy_ranges = {
                "baroque": (1.8, 2.4),
                "classical": (2.0, 2.6),
                "romantic": (2.3, 3.0),
            }


def quality_score(
    notes: Iterable[NoteEvent],
    style: str,
    cfg: QualityConfig | None = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Üretilen müzik için 0–1 arası otomatik kalite skoru.
    """
    if cfg is None:
        cfg = QualityConfig()

    events: List[NoteEvent] = list(notes)
    if not events:
        return 0.0, {}

    pitches = [e.pitch for e in events]
    ent = pitch_entropy(pitches)

    # 1. Entropi skoru
    low, high = cfg.entropy_ranges.get(style, (2.0, 2.6))
    if low <= ent <= high:
        entropy_score = 1.0
    else:
        entropy_score = max(0.0, 1.0 - min(abs(ent - low), abs(ent - high)) / 0.5)

    # 2. Aralık çeşitliliği
    intervals = np.abs(extract_intervals(pitches))
    if intervals.size == 0:
        interval_variety = 0.0
    else:
        interval_variety = len(np.unique(intervals)) / 13.0
    interval_variety = float(min(interval_variety, 1.0))

    # 3. Pitch aralığı
    pitch_range = max(pitches) - min(pitches)
    if 12 <= pitch_range <= 36:
        range_score = 1.0
    else:
        range_score = max(0.0, 1.0 - abs(pitch_range - 24) / 24.0)

    weights = {
        "entropy": 0.3,
        "interval_variety": 0.25,
        "range": 0.45,
    }

    total = (
        weights["entropy"] * entropy_score
        + weights["interval_variety"] * interval_variety
        + weights["range"] * range_score
    )

    details = {
        "entropy": entropy_score,
        "interval_variety": interval_variety,
        "range": range_score,
    }

    return float(total), details


def filter_by_quality(
    items: Iterable[Dict],
    cfg: QualityConfig | None = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Üretilen eserleri kalite eşiğine göre iki listeye ayır.

    items: Her bir eleman en azından `notes` ve `style` alanlarına sahip olmalı.
    """
    if cfg is None:
        cfg = QualityConfig()

    passed: List[Dict] = []
    rejected: List[Dict] = []

    for item in items:
        score, details = quality_score(item["notes"], item.get("style", "classical"), cfg)
        item["quality_score"] = score
        item["quality_details"] = details
        if score >= cfg.threshold:
            passed.append(item)
        else:
            rejected.append(item)

    return passed, rejected


__all__ = ["QualityConfig", "quality_score", "filter_by_quality"]

