"""Pitch serisi üzerinden spektral feature'lar."""

from __future__ import annotations

from typing import Iterable, Dict

import numpy as np


def spectral_features_from_pitch(notes: Iterable[int]) -> Dict[str, float]:
    """
    Pitch serisinin Fourier dönüşümüne dayalı spektral özetler.
    """
    arr = np.asarray(list(notes), dtype=float)
    if arr.size == 0:
        return {
            "spectral_centroid": 0.0,
            "spectral_entropy": 0.0,
            "dominant_frequency": 0.0,
            "spectral_flatness": 0.0,
        }

    arr = arr - arr.mean()
    fft_vals = np.abs(np.fft.rfft(arr))
    power = fft_vals**2
    total_power = power.sum()
    if total_power == 0:
        return {
            "spectral_centroid": 0.0,
            "spectral_entropy": 0.0,
            "dominant_frequency": 0.0,
            "spectral_flatness": 0.0,
        }

    freqs = np.arange(len(fft_vals))

    centroid = float((freqs * power).sum() / total_power)

    prob = power / total_power
    prob = prob[prob > 0]
    spectral_entropy = float(-(prob * np.log2(prob)).sum())

    dominant_frequency = int(np.argmax(power))

    spectral_flatness = float(
        np.exp(np.mean(np.log(power + 1e-8))) / (np.mean(power) + 1e-8)
    )

    return {
        "spectral_centroid": centroid,
        "spectral_entropy": spectral_entropy,
        "dominant_frequency": float(dominant_frequency),
        "spectral_flatness": spectral_flatness,
    }


__all__ = ["spectral_features_from_pitch"]

