"""
Palindromic Structure Detector - Simetrik Müzik Yapıları Keşfi

Müzikteki palindromik (simetrik) yapıları tespit eder:
- Melodic palindromes (ters çevrilebilir melodiler)
- Rhythmic palindromes (ritmik simetri)
- Intervallic palindromes (ortalama noktaya göre simetri)
- Structural mirrors (yapısal aynalar)
- Retrograde motion (retrograde hareket)

Klasik örnekler:
- Bach: Crab Canon
- Haydn: Symphony No. 47 (Palindrome)
- Webern: 12-tone palindromes
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict

from music_math.core.types import NoteEvent


@dataclass
class PalindromeMatch:
    """Bulunan palindrom match'i."""
    start_idx: int
    end_idx: int
    length: int
    type: str  # "melodic", "rhythmic", "intervallic", "structural"
    sequence: List
    mirror_point: float  # Simetri noktası (zaman veya index)
    confidence: float


@dataclass
class StructuralMirror:
    """Yapısal ayna (bölüm tekrarı)."""
    section_a_start: int
    section_a_end: int
    section_b_start: int
    section_b_end: int
    similarity_score: float
    mirror_type: str  # "exact", "approximate", "inverted"


@dataclass
class PalindromeAnalysis:
    """Palindrom analizi sonuçları."""
    melodic_palindromes: List[PalindromeMatch]
    rhythmic_palindromes: List[PalindromeMatch]
    intervallic_palindromes: List[PalindromeMatch]
    structural_mirrors: List[StructuralMirror]
    palindrome_density: float
    has_retrograde: bool
    symmetry_score: float


def is_palindrome(sequence: List, tolerance: int = 0) -> bool:
    """
    Dizinin palindrom olup olmadığını kontrol eder.
    
    Args:
        sequence: Kontrol edilecek dizi
        tolerance: Sayısal tolerans (pitch farkı için)
        
    Returns:
        True ise palindrom
    """
    n = len(sequence)
    if n < 2:
        return False
    
    for i in range(n // 2):
        left = sequence[i]
        right = sequence[n - 1 - i]
        
        if tolerance == 0:
            if left != right:
                return False
        else:
            if abs(left - right) > tolerance:
                return False
    
    return True


def find_melodic_palindromes(
    pitches: List[int],
    min_length: int = 4,
    tolerance: int = 0
) -> List[PalindromeMatch]:
    """
    Melodik palindromları bulur.
    
    Args:
        pitches: Pitch değerleri
        min_length: Minimum palindrom uzunluğu
        tolerance: Sayısal tolerans (semitone cinsinden)
        
    Returns:
        PalindromeMatch listesi
    """
    matches = []
    n = len(pitches)
    
    for length in range(min_length, min(n + 1, 50)):  # Max 50
        for start in range(n - length + 1):
            end = start + length
            window = pitches[start:end]
            
            if is_palindrome(window, tolerance):
                # Simetri noktası (ortalama)
                mirror_point = (start + end - 1) / 2
                
                # Güven skoru: uzunluk ve toleransa göre
                confidence = min(1.0, length / 20) * (1 - tolerance / 12)
                
                matches.append(PalindromeMatch(
                    start_idx=start,
                    end_idx=end - 1,
                    length=length,
                    type="melodic",
                    sequence=window,
                    mirror_point=mirror_point,
                    confidence=round(confidence, 3)
                ))
    
    # En uzun ve en yüksek güvenilirlikteki palindromları seç
    matches.sort(key=lambda x: (x.length, x.confidence), reverse=True)
    
    # Overlapping olmayan en iyi palindromları seç
    non_overlapping = []
    used_ranges = set()
    
    for match in matches:
        # Çakışma kontrolü
        overlaps = False
        for used_start, used_end in used_ranges:
            if not (match.end_idx < used_start or match.start_idx > used_end):
                overlaps = True
                break
        
        if not overlaps:
            non_overlapping.append(match)
            used_ranges.add((match.start_idx, match.end_idx))
    
    return non_overlapping[:20]  # En fazla 20


def find_rhythmic_palindromes(
    durations: List[float],
    min_length: int = 4,
    tolerance_ratio: float = 0.1
) -> List[PalindromeMatch]:
    """
    Ritmik palindromları bulur (süre simetrisi).
    
    Args:
        durations: Nota süreleri
        min_length: Minimum uzunluk
        tolerance_ratio: Süre tolerans oranı
        
    Returns:
        PalindromeMatch listesi
    """
    matches = []
    n = len(durations)
    
    for length in range(min_length, min(n + 1, 50)):
        for start in range(n - length + 1):
            end = start + length
            window = durations[start:end]
            
            # Toleranslı palindrom kontrolü
            is_pal = True
            for i in range(length // 2):
                left = window[i]
                right = window[length - 1 - i]
                tolerance = max(left, right) * tolerance_ratio
                
                if abs(left - right) > tolerance:
                    is_pal = False
                    break
            
            if is_pal:
                mirror_point = (start + end - 1) / 2
                confidence = min(1.0, length / 16) * (1 - tolerance_ratio)
                
                matches.append(PalindromeMatch(
                    start_idx=start,
                    end_idx=end - 1,
                    length=length,
                    type="rhythmic",
                    sequence=[round(d, 3) for d in window],
                    mirror_point=mirror_point,
                    confidence=round(confidence, 3)
                ))
    
    # Non-overlapping seçimi
    matches.sort(key=lambda x: (x.length, x.confidence), reverse=True)
    non_overlapping = []
    used_ranges = set()
    
    for match in matches:
        overlaps = False
        for used_start, used_end in used_ranges:
            if not (match.end_idx < used_start or match.start_idx > used_end):
                overlaps = True
                break
        
        if not overlaps:
            non_overlapping.append(match)
            used_ranges.add((match.start_idx, match.end_idx))
    
    return non_overlapping[:20]


def find_intervallic_palindromes(
    pitches: List[int],
    min_length: int = 3
) -> List[PalindromeMatch]:
    """
    Aralık palindromlarını bulur (simetrik aralık yapısı).
    
    Örnek: C-E-G-E-C (aralıklar: +4, +3, -3, -4) → simetrik
    
    Args:
        pitches: Pitch değerleri
        min_length: Minimum uzunluk
        
    Returns:
        PalindromeMatch listesi
    """
    if len(pitches) < min_length + 1:
        return []
    
    intervals = list(np.diff(pitches))
    matches = []
    n = len(intervals)
    
    for length in range(min_length, min(n + 1, 30)):
        for start in range(n - length + 1):
            end = start + length
            window = intervals[start:end]
            
            # Aralık palindromu: aralıkların işaretleri ters, büyüklükleri aynı
            is_intervallic_pal = True
            for i in range(length // 2):
                left_iv = window[i]
                right_iv = window[length - 1 - i]
                
                # Simetrik: sağdaki aralık, soldakinin negatifi olmalı
                if left_iv != -right_iv:
                    is_intervallic_pal = False
                    break
            
            if is_intervallic_pal:
                mirror_point = (start + end) / 2  # Notalar arasında
                confidence = min(1.0, length / 15)
                
                matches.append(PalindromeMatch(
                    start_idx=start,
                    end_idx=end,  # Aralık sonrası notaya kadar
                    length=length + 1,  # Nota sayısı
                    type="intervallic",
                    sequence=intervals[start:end],
                    mirror_point=mirror_point,
                    confidence=round(confidence, 3)
                ))
    
    return matches[:15]


def find_structural_mirrors(
    pitches: List[int],
    min_section_length: int = 8,
    similarity_threshold: float = 0.8
) -> List[StructuralMirror]:
    """
    Yapısal aynaları bulur (A-A', A-B-A, rondo form vb.).
    
    Args:
        pitches: Pitch değerleri
        min_section_length: Minimum bölüm uzunluğu
        similarity_threshold: Benzerlik eşiği
        
    Returns:
        StructuralMirror listesi
    """
    mirrors = []
    n = len(pitches)
    
    # Olası bölüm uzunlukları
    for section_len in range(min_section_length, min(n // 2 + 1, 100)):
        # Her başlangıç pozisyonu için
        for start_a in range(n - 2 * section_len + 1):
            end_a = start_a + section_len
            section_a = pitches[start_a:end_a]
            
            # Eşleşen ikinci bölümü ara
            for start_b in range(end_a, n - section_len + 1):
                end_b = start_b + section_len
                section_b = pitches[start_b:end_b]
                
                # Benzerlik hesapla
                similarity = calculate_section_similarity(section_a, section_b)
                
                if similarity >= similarity_threshold:
                    # Ayna tipi belirle
                    mirror_type = classify_mirror_type(section_a, section_b)
                    
                    mirrors.append(StructuralMirror(
                        section_a_start=start_a,
                        section_a_end=end_a - 1,
                        section_b_start=start_b,
                        section_b_end=end_b - 1,
                        similarity_score=round(similarity, 3),
                        mirror_type=mirror_type
                    ))
    
    # En iyi eşleşmeleri seç (çakışmayan)
    mirrors.sort(key=lambda x: x.similarity_score, reverse=True)
    non_overlapping = []
    used_ranges = set()
    
    for mirror in mirrors:
        overlaps = False
        for used_start, used_end in used_ranges:
            if not (mirror.section_b_end < used_start or mirror.section_b_start > used_end):
                overlaps = True
                break
        
        if not overlaps:
            non_overlapping.append(mirror)
            used_ranges.add((mirror.section_b_start, mirror.section_b_end))
    
    return non_overlapping[:10]


def calculate_section_similarity(section_a: List[int], section_b: List[int]) -> float:
    """
    İki bölüm arası benzerlik.
    
    Args:
        section_a: Birinci bölüm
        section_b: İkinci bölüm
        
    Returns:
        Benzerlik skoru [0, 1]
    """
    if len(section_a) != len(section_b):
        # Farklı uzunluklar için DTW benzerliği
        from music_math.analysis.similarity import dtw_similarity
        return dtw_similarity(np.array(section_a), np.array(section_b))
    
    # Aynı uzunluk: doğrudan karşılaştırma
    matches = sum(1 for a, b in zip(section_a, section_b) if a == b)
    return matches / len(section_a)


def classify_mirror_type(section_a: List[int], section_b: List[int]) -> str:
    """
    Ayna tipini sınıflandırır.
    
    Args:
        section_a: Birinci bölüm
        section_b: İkinci bölüm
        
    Returns:
        Ayna tipi: "exact", "approximate", "inverted", "retrograde"
    """
    if section_a == section_b:
        return "exact"
    
    # Inverted (ters işaretli)
    if len(section_a) == len(section_b):
        intervals_a = list(np.diff(section_a))
        intervals_b = list(np.diff(section_b))
        
        # Inverted kontrol
        is_inverted = all(a == -b for a, b in zip(intervals_a, intervals_b))
        if is_inverted:
            return "inverted"
        
        # Retrograde (ters çevrilmiş)
        if section_a == section_b[::-1]:
            return "retrograde"
    
    return "approximate"


def detect_retrograde_motion(
    pitches: List[int],
    min_length: int = 6,
    window_size: int = 8
) -> bool:
    """
    Retrograde hareket tespiti (tersine melodik hareket).
    
    Args:
        pitches: Pitch değerleri
        min_length: Minimum uzunluk
        window_size: Kontrol pencere boyutu
        
    Returns:
        True ise retrograde hareket var
    """
    n = len(pitches)
    
    for i in range(n - 2 * window_size + 1):
        window1 = pitches[i:i + window_size]
        window2 = pitches[i + window_size:i + 2 * window_size]
        
        # İkinci pencere birincinin tersi mi?
        if window2 == window1[::-1]:
            return True
    
    return False


def calculate_symmetry_score(
    melodic: List[PalindromeMatch],
    rhythmic: List[PalindromeMatch],
    intervallic: List[PalindromeMatch],
    structural: List[StructuralMirror],
    total_notes: int
) -> float:
    """
    Genel simetri skoru hesaplar.
    
    Args:
        melodic: Melodik palindromlar
        rhythmic: Ritmik palindromlar
        intervallic: Aralık palindromları
        structural: Yapısal aynalar
        total_notes: Toplam nota sayısı
        
    Returns:
        Simetri skoru [0, 1]
    """
    if total_notes == 0:
        return 0.0
    
    # Her tip için skor
    melodic_coverage = sum(p.length for p in melodic) / total_notes
    rhythmic_coverage = sum(p.length for p in rhythmic) / total_notes
    intervallic_coverage = sum(p.length for p in intervallic) / total_notes
    
    # Yapısal aynalar için skor
    structural_score = len(structural) * 0.1
    
    # Ağırlıklı ortalama
    score = (
        melodic_coverage * 0.3 +
        rhythmic_coverage * 0.25 +
        intervallic_coverage * 0.25 +
        min(structural_score, 0.2) * 0.2
    )
    
    return min(1.0, score)


def analyze_palindromic_structure(events: List[NoteEvent]) -> PalindromeAnalysis:
    """
    Kapsamlı palindromik yapı analizi.
    
    Args:
        events: Nota olayları
        
    Returns:
        PalindromeAnalysis sonuçları
    """
    pitches = [e.pitch for e in events]
    durations = [e.duration for e in events]
    
    if len(pitches) < 4:
        return PalindromeAnalysis(
            melodic_palindromes=[],
            rhythmic_palindromes=[],
            intervallic_palindromes=[],
            structural_mirrors=[],
            palindrome_density=0.0,
            has_retrograde=False,
            symmetry_score=0.0
        )
    
    # Palindromları bul
    melodic = find_melodic_palindromes(pitches, min_length=4, tolerance=0)
    rhythmic = find_rhythmic_palindromes(durations, min_length=4, tolerance_ratio=0.1)
    intervallic = find_intervallic_palindromes(pitches, min_length=3)
    structural = find_structural_mirrors(pitches, min_section_length=8)
    
    # Retrograde kontrolü
    has_retrograde = detect_retrograde_motion(pitches)
    
    # Palindrom yoğunluğu
    total_palindrome_notes = (
        sum(p.length for p in melodic) +
        sum(p.length for p in rhythmic) +
        sum(p.length for p in intervallic)
    )
    density = total_palindrome_notes / len(pitches) if pitches else 0.0
    
    # Simetri skoru
    symmetry = calculate_symmetry_score(
        melodic, rhythmic, intervallic, structural, len(pitches)
    )
    
    return PalindromeAnalysis(
        melodic_palindromes=melodic,
        rhythmic_palindromes=rhythmic,
        intervallic_palindromes=intervallic,
        structural_mirrors=structural,
        palindrome_density=round(density, 3),
        has_retrograde=has_retrograde,
        symmetry_score=round(symmetry, 3)
    )


def extract_palindrome_features(events: List[NoteEvent]) -> Dict[str, float]:
    """
    Feature extraction için palindrom özellikleri.
    
    Args:
        events: Nota olayları
        
    Returns:
        Feature sözlüğü
    """
    analysis = analyze_palindromic_structure(events)
    
    features = {
        "palindrome_melodic_count": len(analysis.melodic_palindromes),
        "palindrome_rhythmic_count": len(analysis.rhythmic_palindromes),
        "palindrome_intervallic_count": len(analysis.intervallic_palindromes),
        "palindrome_structural_count": len(analysis.structural_mirrors),
        "palindrome_density": analysis.palindrome_density,
        "palindrome_symmetry_score": analysis.symmetry_score,
        "has_retrograde": 1.0 if analysis.has_retrograde else 0.0,
    }
    
    # En uzun palindrom
    longest_melodic = max((p.length for p in analysis.melodic_palindromes), default=0)
    longest_rhythmic = max((p.length for p in analysis.rhythmic_palindromes), default=0)
    
    features["palindrome_longest_melodic"] = longest_melodic
    features["palindrome_longest_rhythmic"] = longest_rhythmic
    
    return features


__all__ = [
    "PalindromeMatch",
    "StructuralMirror",
    "PalindromeAnalysis",
    "is_palindrome",
    "find_melodic_palindromes",
    "find_rhythmic_palindromes",
    "find_intervallic_palindromes",
    "find_structural_mirrors",
    "calculate_section_similarity",
    "classify_mirror_type",
    "detect_retrograde_motion",
    "calculate_symmetry_score",
    "analyze_palindromic_structure",
    "extract_palindrome_features",
]
