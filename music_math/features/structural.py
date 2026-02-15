"""Makro yapı, tekrar ve fraktal benzerlik ile ilgili feature'lar."""

from __future__ import annotations

from typing import Iterable, Dict

import numpy as np


def self_similarity_matrix(notes: Iterable[int], window: int = 20) -> np.ndarray:
    """
    Pitch-class histogramlarına dayalı basit self-similarity matrix.
    """
    pcs = [int(n) % 12 for n in notes]
    if len(pcs) <= window:
        return np.zeros((0, 0), dtype=float)

    n_windows = len(pcs) - window
    windows = np.array([pcs[i : i + window] for i in range(n_windows)], dtype=int)

    def to_hist(w: np.ndarray) -> np.ndarray:
        h = np.bincount(w, minlength=12).astype(float)
        total = h.sum()
        if total == 0:
            return np.zeros_like(h)
        return h / total

    hists = np.array([to_hist(w) for w in windows])

    n = len(hists)
    ssm = np.zeros((n, n), dtype=float)
    for i in range(n):
        vi = hists[i]
        norm_i = np.linalg.norm(vi) or 1.0
        for j in range(i, n):
            vj = hists[j]
            norm_j = np.linalg.norm(vj) or 1.0
            sim = float(np.dot(vi, vj) / (norm_i * norm_j))
            ssm[i, j] = ssm[j, i] = sim

    return ssm


def repetition_index(notes: Iterable[int], window: int = 20) -> float:
    """
    Ortalama öz-benzerlik skoru.

    Yüksek değer = çok tekrar eden temalar,
    Düşük değer = daha serbest yapı.
    """
    ssm = self_similarity_matrix(notes, window=window)
    n = ssm.shape[0]
    if n < 10:
        return 0.0

    off_diag = []
    for i in range(n):
        for j in range(i + 5, n):  # en az 5 adım uzaklık
            off_diag.append(ssm[i, j])

    return float(np.mean(off_diag)) if off_diag else 0.0


def fractal_dimension_estimate(notes: Iterable[int], n_segments: int = 8) -> float:
    """
    Pitch serisine dayalı basit fraktal boyut tahmini.

    ~1.0 = çok basit
    ~1.5 = orta karmaşıklık
    ~2.0 = daha rastgele
    """
    arr = np.asarray(list(notes), dtype=float)
    if arr.size < 4:
        return 1.0

    # 0–1 aralığına normalize et
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    counts = []
    sizes = []
    for s in range(2, n_segments + 1):
        box_size = len(arr) / s
        occupied = set()
        for idx, p in enumerate(arr):
            box_x = int(idx / box_size)
            box_y = int(p * s)
            occupied.add((box_x, box_y))
        counts.append(len(occupied))
        sizes.append(1.0 / s)

    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return float(-slope)


def extract_structural_features(notes: Iterable[int]) -> Dict[str, float]:
    """Yapısal feature'lar için giriş noktası."""
    notes_list = list(notes)
    return {
        "repetition_index": repetition_index(notes_list),
        "fractal_dimension": fractal_dimension_estimate(notes_list),
        "unique_pitch_classes": float(len({int(n) % 12 for n in notes_list})),
        "total_notes": float(len(notes_list)),
    }


__all__ = [
    "self_similarity_matrix",
    "repetition_index",
    "fractal_dimension_estimate",
    "extract_structural_features",
]

