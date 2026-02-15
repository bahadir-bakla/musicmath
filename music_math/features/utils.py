"""Feature extraction için ortak yardımcı fonksiyonlar."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def safe_entropy(probabilities: Iterable[float]) -> float:
    """
    Basit Shannon entropi hesaplayıcı.

    Args:
        probabilities: Toplamı 1 olan olasılık değerleri.
    """
    probs = np.asarray(list(probabilities), dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


__all__ = ["safe_entropy"]

