"""
Mantikli brute-force motif arayici.

Strateji:
- Melodiyi interval serisine cevir (transpozisyon invaryant).
- Sabit uzunlukta (L = 4..12) tum n-gram'lari cikar.
- Support >= min_support olanlari tut.
- Null model: interval serisini shuffle edip ayni L icin n-gram frekanslari;
  gozlenen count icin z-score (surprisal) hesapla.
- Cikti: pattern, support, z_score, ornek pozisyonlar.

Brute-force "mantikli" sinirlarla: max pattern uzunlugu, top-K, hash tabanli sayim.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

from music_math.core.types import NoteEvent


def events_to_interval_sequence(events: List[NoteEvent]) -> np.ndarray:
    """Pitch serisinden interval serisi (signed)."""
    pitches = np.array([e.pitch for e in events], dtype=int)
    if len(pitches) < 2:
        return np.array([], dtype=int)
    return np.diff(pitches)


def extract_ngrams(intervals: np.ndarray, L: int) -> Dict[Tuple[int, ...], List[int]]:
    """
    Uzunluk L olan tum interval n-gram'larini cikarir.
    Key: tuple of L integers (interval degeri), value: baslangic indeksleri listesi.
    """
    n = len(intervals)
    if n < L:
        return {}
    out = defaultdict(list)
    for i in range(n - L + 1):
        key = tuple(int(intervals[i + j]) for j in range(L))
        out[key].append(i)
    return dict(out)


def _null_mean_std_for_pattern(
    intervals: np.ndarray,
    pattern_tuple: Tuple[int, ...],
    L: int,
    n_shuffles: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Bu pattern icin shuffle null dagiliminda ortalama ve std (Poisson varsayimi ile)."""
    arr = np.array(intervals, copy=True)
    counts = []
    for _ in range(n_shuffles):
        rng.shuffle(arr)
        ngrams = extract_ngrams(arr, L)
        counts.append(len(ngrams.get(pattern_tuple, [])))
    mu = float(np.mean(counts))
    sigma = float(np.std(counts))
    if sigma < 1e-6:
        sigma = np.sqrt(mu + 1e-6)  # Poisson
    return mu, sigma


def mine_patterns_one_piece(
    events: List[NoteEvent],
    min_length: int = 4,
    max_length: int = 12,
    min_support: int = 3,
    n_shuffles: int = 25,
    top_k: int = 500,
    random_state: int = 42,
) -> List[Dict]:
    """
    Tek eserde interval n-gram motiflerini bulur.
    Donen liste: her motif icin pattern (str), length, support, z_score, positions (ilk 20).
    """
    intervals = events_to_interval_sequence(events)
    if len(intervals) < min_length:
        return []

    rng = np.random.default_rng(random_state)
    results = []

    for L in range(min_length, min(max_length + 1, len(intervals) + 1)):
        ngrams = extract_ngrams(intervals, L)
        if not ngrams:
            continue

        for pattern_tuple, positions in ngrams.items():
            support = len(positions)
            if support < min_support:
                continue
            mean_null, std_null = _null_mean_std_for_pattern(
                intervals, pattern_tuple, L, n_shuffles, rng
            )
            z = (support - mean_null) / (std_null + 1e-6)
            pattern_str = ",".join(map(str, pattern_tuple))
            results.append({
                "pattern": pattern_str,
                "length": L,
                "support": support,
                "z_score": round(float(z), 4),
                "positions": positions[:20],
            })

    results.sort(key=lambda x: -x["z_score"])
    return results[:top_k]


__all__ = [
    "events_to_interval_sequence",
    "extract_ngrams",
    "mine_patterns_one_piece",
]
