"""
Asal sayı tabanlı harmoni analizi.

Müzikte asal sayı örüntüleri:
- Interval dizileri asal sayı mı?
- Nota uzunlukları asal sayı oranlarında mı?
- Fraz yapıları asal sayı periodlarında mı?
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Tuple
from collections import Counter

from music_math.core.types import NoteEvent


def is_prime(n: int) -> bool:
    """Bir sayının asal olup olmadığını kontrol et."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Trial division
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def sieve_of_eratosthenes(limit: int) -> List[int]:
    """Verilen limite kadar tüm asal sayıları bul."""
    if limit < 2:
        return []
    
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(np.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    
    return [i for i in range(limit + 1) if sieve[i]]


def interval_prime_density(pitches: Iterable[int]) -> float:
    """
    Interval dizisinde asal sayı yoğunluğu.
    
    Returns:
        0.0-1.0 arası oran (kaç interval asal sayı?)
    """
    pitches_arr = np.array(list(pitches), dtype=int)
    if len(pitches_arr) < 2:
        return 0.0
    
    intervals = np.abs(np.diff(pitches_arr))
    
    if len(intervals) == 0:
        return 0.0
    
    prime_count = sum(1 for iv in intervals if is_prime(int(iv)))
    return prime_count / len(intervals)


def duration_prime_ratio(events: Iterable[NoteEvent]) -> float:
    """
    Nota sürelerinin asal sayı katlarında olma oranı.
    
    Süreleri en yakın integer'a yuvarlar, asal olanların oranını hesaplar.
    """
    events_list = list(events)
    if not events_list:
        return 0.0
    
    # Süreleri 16'lık birimlerine çevir (quarter = 4, eighth = 2, etc.)
    durations_int = [max(1, int(round(e.duration * 4))) for e in events_list]
    
    prime_count = sum(1 for d in durations_int if is_prime(d))
    return prime_count / len(durations_int)


def phrase_length_primes(pitches: Iterable[int], window: int = 8) -> List[int]:
    """
    Fraz uzunluklarını tespit et ve asal olanları bul.
    
    Basit heuristic: Pitch pattern tekrarı veya rest işareti ile fraz ayrımı.
    """
    pitches_arr = np.array(list(pitches), dtype=int)
    
    # Basit yaklaşım: Büyük interval atlamalarını fraz sınırı say
    intervals = np.abs(np.diff(pitches_arr))
    
    # Threshold'dan büyük intervallar fraz sınırı
    threshold = np.percentile(intervals, 75) if len(intervals) > 0 else 5
    
    phrase_boundaries = [0]
    for i, iv in enumerate(intervals):
        if iv > threshold:
            phrase_boundaries.append(i + 1)
    phrase_boundaries.append(len(pitches_arr))
    
    phrase_lengths = np.diff(phrase_boundaries)
    prime_phrases = [l for l in phrase_lengths if is_prime(int(l))]
    
    return prime_phrases


def prime_harmonic_series(pitches: Iterable[int], max_harmonic: int = 20) -> dict[int, float]:
    """
    Pitch-class dağılımını harmonik seriye (asal sayı harmonikleri) fit et.
    
    Returns:
        {asal_harmonic: amplitude} dict
    """
    pitches_arr = np.array(list(pitches), dtype=int)
    if len(pitches_arr) == 0:
        return {}
    
    # Pitch class histogram
    pc_hist = np.bincount(pitches_arr % 12, minlength=12).astype(float)
    pc_hist /= (pc_hist.sum() + 1e-8)
    
    # Asal harmonikler (2, 3, 5, 7, 11, 13, ...)
    primes = [p for p in sieve_of_eratosthenes(max_harmonic) if p >= 2]
    
    result = {}
    for p in primes:
        # p. harmoniğin pitch-class'taki ağırlığını hesapla
        # Basit yaklaşım: p mod 12 konumundaki değer
        pc_index = p % 12
        result[p] = float(pc_hist[pc_index])
    
    return result


def analyze_prime_structure(events: Iterable[NoteEvent]) -> dict:
    """
    Bir eser için tüm asal sayı tabanlı metrikleri hesapla.
    """
    events_list = list(events)
    if not events_list:
        return {}
    
    pitches = [e.pitch for e in events_list]
    
    return {
        "interval_prime_density": interval_prime_density(pitches),
        "duration_prime_ratio": duration_prime_ratio(events_list),
        "phrase_length_primes": phrase_length_primes(pitches),
        "prime_harmonic_amplitudes": prime_harmonic_series(pitches),
        "total_notes": len(events_list),
        "unique_intervals": len(set(np.abs(np.diff(pitches)).tolist())),
    }


__all__ = [
    "is_prime",
    "sieve_of_eratosthenes",
    "interval_prime_density",
    "duration_prime_ratio",
    "phrase_length_primes",
    "prime_harmonic_series",
    "analyze_prime_structure",
]
