"""
Golden Ratio (φ ≈ 1.618) ve Fibonacci analizleri.

Mozart'ın eserlerinde Golden Ratio kullanımı:
- Bölüm uzunlukları Fibonacci oranında mı?
- Climax noktası Golden Ratio'da mı?
- Tema tekrarları Fibonacci sayılarında mı?
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Tuple

from music_math.core.types import NoteEvent

PHI = (1 + np.sqrt(5)) / 2  # Golden Ratio ≈ 1.618034


def fibonacci_sequence(n: int) -> List[int]:
    """İlk n Fibonacci sayısını üret."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    
    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib


def is_fibonacci(num: int, tolerance: int = 0) -> bool:
    """Bir sayının Fibonacci sayısı olup olmadığını kontrol et."""
    if num < 1:
        return False
    
    fibs = fibonacci_sequence(30)  # İlk 30 Fibonacci sayısı
    return any(abs(num - f) <= tolerance for f in fibs)


def golden_ratio_proximity(a: float, b: float) -> float:
    """
    İki sayının oranının Golden Ratio'ya ne kadar yakın olduğunu ölç.
    
    Returns:
        0.0 = tam eşleşme, daha büyük değer = daha uzak
    """
    if b == 0:
        return float('inf')
    
    ratio = a / b
    return abs(ratio - PHI)


def climax_position_golden_ratio(pitches: Iterable[int]) -> Tuple[float, float]:
    """
    Eserin en yüksek pitch'inin (climax) konumunu bul.
    Golden Ratio pozisyonunda mı?
    
    Returns:
        (actual_position_ratio, distance_from_golden_ratio)
    """
    pitches_arr = np.array(list(pitches), dtype=int)
    if len(pitches_arr) == 0:
        return 0.0, float('inf')
    
    # En yüksek pitch'in pozisyonu
    max_pitch_idx = int(np.argmax(pitches_arr))
    total_len = len(pitches_arr)
    
    # Pozisyon oranı
    position_ratio = max_pitch_idx / total_len
    
    # Golden Ratio'ya uzaklık
    golden_position = 1.0 / PHI  # ≈ 0.618
    distance = abs(position_ratio - golden_position)
    
    return position_ratio, distance


def section_length_fibonacci(events: Iterable[NoteEvent], n_sections: int = 5) -> List[Tuple[int, bool]]:
    """
    Eseri n_sections bölüme ayır, her bölümün uzunluğu Fibonacci sayısı mı kontrol et.
    
    Returns:
        [(section_length, is_fibonacci), ...]
    """
    events_list = list(events)
    total_len = len(events_list)
    
    if total_len == 0 or n_sections == 0:
        return []
    
    section_size = total_len // n_sections
    sections = []
    
    for i in range(n_sections):
        start = i * section_size
        end = start + section_size if i < n_sections - 1 else total_len
        length = end - start
        
        sections.append((length, is_fibonacci(length, tolerance=2)))
    
    return sections


def golden_ratio_in_durations(events: Iterable[NoteEvent]) -> float:
    """
    Nota sürelerinin oranlarında Golden Ratio kullanımı.
    
    Ardışık nota süre çiftlerinden kaçı Golden Ratio'ya yakın?
    """
    events_list = list(events)
    if len(events_list) < 2:
        return 0.0
    
    durations = [e.duration for e in events_list]
    
    close_to_golden = 0
    total_pairs = 0
    
    for i in range(len(durations) - 1):
        d1, d2 = durations[i], durations[i + 1]
        if d2 == 0:
            continue
        
        proximity = golden_ratio_proximity(d1, d2)
        
        # Eğer 0.1 tolerance içindeyse, Golden Ratio sayılır
        if proximity < 0.1:
            close_to_golden += 1
        
        total_pairs += 1
    
    return close_to_golden / total_pairs if total_pairs > 0 else 0.0


def fibonacci_interval_patterns(pitches: Iterable[int]) -> dict[int, int]:
    """
    Fibonacci sayısı olan interval'ların histogramı.
    
    Returns:
        {fibonacci_interval: count}
    """
    pitches_arr = np.array(list(pitches), dtype=int)
    if len(pitches_arr) < 2:
        return {}
    
    intervals = np.abs(np.diff(pitches_arr))
    
    fibs = set(fibonacci_sequence(20))
    fib_intervals = {}
    
    for iv in intervals:
        iv_int = int(iv)
        if iv_int in fibs:
            fib_intervals[iv_int] = fib_intervals.get(iv_int, 0) + 1
    
    return fib_intervals


def analyze_golden_ratio_structure(events: Iterable[NoteEvent]) -> dict:
    """
    Bir eser için tüm Golden Ratio & Fibonacci metriklerini hesapla.
    """
    events_list = list(events)
    if not events_list:
        return {}
    
    pitches = [e.pitch for e in events_list]
    
    climax_pos, climax_dist = climax_position_golden_ratio(pitches)
    sections = section_length_fibonacci(events_list, n_sections=5)
    
    return {
        "climax_position_ratio": climax_pos,
        "climax_golden_distance": climax_dist,
        "climax_is_golden": climax_dist < 0.05,  # 5% tolerance
        "section_lengths": [s[0] for s in sections],
        "sections_fibonacci": [s[1] for s in sections],
        "fibonacci_section_ratio": sum(s[1] for s in sections) / len(sections) if sections else 0.0,
        "golden_ratio_in_durations": golden_ratio_in_durations(events_list),
        "fibonacci_intervals": fibonacci_interval_patterns(pitches),
    }


__all__ = [
    "PHI",
    "fibonacci_sequence",
    "is_fibonacci",
    "golden_ratio_proximity",
    "climax_position_golden_ratio",
    "section_length_fibonacci",
    "golden_ratio_in_durations",
    "fibonacci_interval_patterns",
    "analyze_golden_ratio_structure",
]
