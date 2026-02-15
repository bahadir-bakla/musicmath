"""
Bilinen matematiksel oruntuleri nota/aralik/sure serileri uzerinde test eder.

Once literaturde var olan yapilar denenir:
- Fibonacci ve Lucas sayilari (aralik buyuklukleri, bolum uzunluklari)
- Asal sayilar (kumulatif toplam, aralik, sure)
- Altin oran (bolum oranlari, climax pozisyonu)
- Perfect numbers (6, 28, 496)
- Ucgensel sayilar (1, 3, 6, 10, 15, 21)
- Pitagorik oranlar (3:4:5, 5:12:13) – sure/aralik oranlari

Her test bir skor veya oran doner; brute-force ayri bir katmanda calisir.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Sequence

from music_math.core.types import NoteEvent
from music_math.analysis.prime_harmony import is_prime, interval_prime_density, duration_prime_ratio
from music_math.analysis.golden_ratio import (
    PHI,
    fibonacci_sequence,
    is_fibonacci,
    golden_ratio_proximity,
    climax_position_golden_ratio,
)

# --- Sabit kumeler (bilinen matematiksel sayi dizileri) ---
FIBONACCI_SET = set(fibonacci_sequence(25))  # 1,1,2,3,5,8,...


def lucas_sequence(n: int) -> List[int]:
    """Lucas sayilari: L0=2, L1=1, Ln = Ln-1 + Ln-2."""
    if n <= 0:
        return []
    if n == 1:
        return [2]
    lucas = [2, 1]
    for _ in range(2, n):
        lucas.append(lucas[-1] + lucas[-2])
    return lucas


LUCAS_SET = set(lucas_sequence(20))

# Perfect numbers: 6, 28, 496, 8128 (cok buyukleri atliyoruz)
PERFECT_SET = {6, 28, 496, 8128}

# Ucgensel sayilar: n(n+1)/2 -> 1,3,6,10,15,21,28,36,45,55
TRIANGULAR_SET = {n * (n + 1) // 2 for n in range(1, 20)}

# Pitagorik ucluler (oranlar): (3,4,5), (5,12,13), (8,15,17) -> oran 3:4:5 vb.
def _pythagorean_ratios() -> List[tuple]:
    """Kucuk tam sayi oranlari (Pitagorik uclulerden)."""
    return [(3, 4), (3, 5), (4, 5), (5, 12), (5, 13), (8, 15)]


def _score_ratio_near(a: float, b: float, target_ratio: float, tol: float = 0.08) -> float:
    """a/b orani target_ratio'ya tol icinde mi? 1.0 = evet, 0.0 = hayir."""
    if b == 0:
        return 0.0
    r = a / b
    dist = abs(r - target_ratio)
    if dist <= tol:
        return 1.0 - (dist / tol) * 0.5
    return 0.0


def test_fibonacci_intervals(pitches: np.ndarray) -> float:
    """Araliklarin (mutlak deger) kaci Fibonacci sayisi? Oran doner."""
    if len(pitches) < 2:
        return 0.0
    intervals = np.abs(np.diff(pitches.astype(int)))
    count = sum(1 for iv in intervals if int(iv) in FIBONACCI_SET)
    return count / len(intervals)


def test_fibonacci_phrase_lengths(pitches: np.ndarray, percentile: float = 75) -> float:
    """Buyuk araliklarla bolunen 'fraz' uzunluklarinin kaci Fibonacci?"""
    if len(pitches) < 4:
        return 0.0
    intervals = np.abs(np.diff(pitches.astype(int)))
    th = np.percentile(intervals, percentile) if len(intervals) > 0 else 5
    boundaries = [0]
    for i, iv in enumerate(intervals):
        if iv >= th:
            boundaries.append(i + 1)
    boundaries.append(len(pitches))
    lengths = np.diff(boundaries)
    if len(lengths) == 0:
        return 0.0
    count = sum(1 for L in lengths if int(L) in FIBONACCI_SET)
    return count / len(lengths)


def test_lucas_intervals(pitches: np.ndarray) -> float:
    """Araliklarin kaci Lucas sayisi? Oran."""
    if len(pitches) < 2:
        return 0.0
    intervals = np.abs(np.diff(pitches.astype(int)))
    count = sum(1 for iv in intervals if int(iv) in LUCAS_SET)
    return count / len(intervals)


def test_perfect_triangular_intervals(pitches: np.ndarray) -> Dict[str, float]:
    """Aralik buyukluklerinde perfect ve ucgensel sayi oranlari."""
    if len(pitches) < 2:
        return {"perfect_interval_ratio": 0.0, "triangular_interval_ratio": 0.0}
    intervals = np.abs(np.diff(pitches.astype(int)))
    n = len(intervals)
    perf = sum(1 for iv in intervals if int(iv) in PERFECT_SET) / n
    tri = sum(1 for iv in intervals if int(iv) in TRIANGULAR_SET) / n
    return {"perfect_interval_ratio": float(perf), "triangular_interval_ratio": float(tri)}


def test_golden_section_ratios(events: Sequence[NoteEvent]) -> float:
    """Ardisik bolum uzunluklari (nota sayisi) oranlarinin phi'ye yaklinligi."""
    if len(events) < 10:
        return 0.0
    n = len(events)
    n_sec = 5
    section_size = n // n_sec
    lengths = [section_size] * (n_sec - 1) + [n - section_size * (n_sec - 1)]
    if lengths[-1] <= 0:
        return 0.0
    scores = []
    for i in range(len(lengths) - 1):
        len_a, len_b = lengths[i], lengths[i + 1]
        if len_b == 0:
            continue
        prox = golden_ratio_proximity(float(len_a), float(len_b))
        if prox < 0.15:
            scores.append(1.0 - prox / 0.15)
    return float(np.mean(scores)) if scores else 0.0


def test_pythagorean_duration_ratios(events: Sequence[NoteEvent]) -> float:
    """Ardisik nota sureleri oranlarinin Pitagorik (3:4, 3:5, 5:12 vb.) oranlara yaklinligi."""
    if len(events) < 2:
        return 0.0
    durations = [max(1e-6, e.duration) for e in events]
    ratios_list = _pythagorean_ratios()
    hits = 0
    total = 0
    for i in range(len(durations) - 1):
        a, b = durations[i], durations[i + 1]
        if b == 0:
            continue
        r = a / b
        for (p, q) in ratios_list:
            tr = p / q
            if abs(r - tr) / max(tr, 1e-6) < 0.12:
                hits += 1
                break
        total += 1
    return hits / total if total else 0.0


def test_cumulative_prime_ratio(pitches: np.ndarray) -> float:
    """Kumulatif pitch toplamlarinda asal orani (note_numeric_patterns ile uyumlu)."""
    if len(pitches) == 0:
        return 0.0
    cum = np.cumsum(pitches.astype(int))
    return float(np.mean([1.0 if is_prime(int(x)) else 0.0 for x in cum]))


def analyze_known_patterns(events: Sequence[NoteEvent]) -> Dict[str, float]:
    """
    Tek bir eser icin tum bilinen matematiksel oruntu testlerini calistirir.
    Donen sozluk CSV'ye veya feature birlestirmeye uygundur.
    """
    events_list = list(events)
    if not events_list:
        return {}
    pitches = np.array([e.pitch for e in events_list])

    out = {}
    # Mevcut moduller
    out["interval_prime_density"] = interval_prime_density(pitches)
    out["duration_prime_ratio"] = duration_prime_ratio(events_list)
    climax_pos, climax_dist = climax_position_golden_ratio(pitches)
    out["climax_position_ratio"] = climax_pos
    out["climax_golden_distance"] = climax_dist
    out["climax_near_golden"] = 1.0 if climax_dist < 0.05 else 0.0

    # Fibonacci / Lucas
    out["fibonacci_interval_ratio"] = test_fibonacci_intervals(pitches)
    out["fibonacci_phrase_ratio"] = test_fibonacci_phrase_lengths(pitches)
    out["lucas_interval_ratio"] = test_lucas_intervals(pitches)

    # Perfect & Triangular
    pt = test_perfect_triangular_intervals(pitches)
    out.update(pt)

    # Golden section (bolum uzunluk oranlari)
    out["golden_section_ratio_score"] = test_golden_section_ratios(events_list)

    # Pitagorik sure oranlari
    out["pythagorean_duration_ratio"] = test_pythagorean_duration_ratios(events_list)

    # Kumulatif asal (zaten note_numeric_patterns'ta var, tutarlilik icin)
    out["cumulative_prime_ratio"] = test_cumulative_prime_ratio(pitches)

    return out


__all__ = [
    "analyze_known_patterns",
    "test_fibonacci_intervals",
    "test_fibonacci_phrase_lengths",
    "test_lucas_intervals",
    "test_perfect_triangular_intervals",
    "test_golden_section_ratios",
    "test_pythagorean_duration_ratios",
    "test_cumulative_prime_ratio",
]
