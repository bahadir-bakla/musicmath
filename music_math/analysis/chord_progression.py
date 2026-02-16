"""
Chord Progression Miner - Akor Progression Pattern Keşfi

Klasik müzikte yaygın akor progression'larını tespit eder:
- II-V-I (Jazz/Classical)
- I-V-vi-IV (Pop progression)
- Circle of Fifths (Beşte bir çemberi)
- vi-IV-I-V (Andalusian cadence)
- i-VII-VI-V (Flamenco progression)

Roman numeral analizi yaparak tonal merkeze göre akorları sınıflandırır.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum

from music_math.core.types import NoteEvent


class ChordQuality(Enum):
    """Akor kalitesi."""
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    DOMINANT = "dominant"
    MAJOR7 = "maj7"
    MINOR7 = "min7"
    DOMINANT7 = "dom7"


@dataclass
class Chord:
    """Akor temsili."""
    root: int  # MIDI pitch class (0-11)
    quality: ChordQuality
    bass: Optional[int] = None  # Inversion için
    
    def __repr__(self):
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        root_name = names[self.root]
        return f"{root_name}{self.quality.value}"


@dataclass
class ChordProgression:
    """Akor progression'ı."""
    chords: List[Chord]
    roman_numerals: List[str]
    key: int  # 0-11, tonal merkez
    key_name: str
    is_minor: bool
    start_time: float
    end_time: float
    confidence: float


@dataclass
class ProgressionPattern:
    """Bulunan progression pattern'ı."""
    pattern_type: str  # "II-V-I", "circle_of_fifths", vb.
    roman_sequence: List[str]
    occurrences: List[Tuple[int, int]]  # (start_idx, end_idx) listesi
    frequency: float
    contexts: List[str]  # Hangi bestecilerde/dönemlerde bulundu


# Yaygın progression pattern'ları (Roman numerals)
KNOWN_PROGRESSIONS = {
    "II-V-I": ["ii", "V", "I"],
    "II-V-I-EXTENDED": ["ii", "V", "I", "vi"],
    "I-V-vi-IV": ["I", "V", "vi", "IV"],
    "vi-IV-I-V": ["vi", "IV", "I", "V"],  # Andalusian/Pop
    "I-V-vi-iii-IV": ["I", "V", "vi", "iii", "IV"],  # Axis progression
    "i-VII-VI-V": ["i", "VII", "VI", "V"],  # Flamenco/Andalusian
    "i-VI-III-VII": ["i", "VI", "III", "VII"],  # Minor circle
    "I-vi-IV-V": ["I", "vi", "IV", "V"],  # 50s progression
    "ii-IV-I": ["ii", "IV", "I"],  # Jazz substitution
    "iii-vi-ii-V": ["iii", "vi", "ii", "V"],  # Circle turnaround
    "vi-ii-V-I": ["vi", "ii", "V", "I"],  # Extended turnaround
    "I-IV-vii-iii-vi": ["I", "IV", "vii", "iii", "vi"],  # Extended circle
}

# Beşte bir çemberi progression'ları
CIRCLE_OF_FIFTHS_MAJOR = ["I", "IV", "vii", "iii", "vi", "ii", "V", "I"]
CIRCLE_OF_FIFTHS_MINOR = ["i", "iv", "VII", "III", "VI", "ii", "V", "i"]

# Major scale degrees
MAJOR_SCALE_DEGREES = [0, 2, 4, 5, 7, 9, 11]  # C, D, E, F, G, A, B
MINOR_SCALE_DEGREES = [0, 2, 3, 5, 7, 8, 10]  # C, D, Eb, F, G, Ab, Bb

# Roman numeral mappings
ROMAN_MAJOR = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii"}
ROMAN_MINOR = {0: "i", 2: "ii", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}


def extract_simultaneous_notes(events: List[NoteEvent], time_window: float = 0.1) -> List[Tuple[float, List[int]]]:
    """
    Aynı anda çalan notaları gruplar.
    
    Args:
        events: Nota olayları
        time_window: Aynı zaman dilimi (saniye)
        
    Returns:
        [(zaman, [pitch'ler]), ...]
    """
    if not events:
        return []
    
    # Zamanlara göre sırala
    sorted_events = sorted(events, key=lambda e: e.start)
    
    groups = []
    current_time = sorted_events[0].start
    current_pitches = [sorted_events[0].pitch]
    
    for event in sorted_events[1:]:
        if abs(event.start - current_time) <= time_window:
            current_pitches.append(event.pitch)
        else:
            groups.append((current_time, sorted(set(current_pitches))))
            current_time = event.start
            current_pitches = [event.pitch]
    
    # Son grubu ekle
    if current_pitches:
        groups.append((current_time, sorted(set(current_pitches))))
    
    return groups


def identify_chord(pitches: List[int]) -> Optional[Chord]:
    """
    Nota koleksiyonundan akor tanımlar.
    
    Args:
        pitches: Nota pitch'leri listesi
        
    Returns:
        Chord objesi veya None
    """
    if len(pitches) < 2:
        return None
    
    # Pitch class'lara dönüştür (0-11)
    pitch_classes = sorted(set([p % 12 for p in pitches]))
    
    if len(pitch_classes) < 2:
        return None
    
    # Her pitch class'ı kök olarak dene
    for root in pitch_classes:
        intervals = [(pc - root) % 12 for pc in pitch_classes]
        
        # Temel üçlü intervaller
        has_third = 4 in intervals or 3 in intervals
        has_fifth = 7 in intervals
        
        if has_third and has_fifth:
            if 4 in intervals and 7 in intervals:
                return Chord(root, ChordQuality.MAJOR)
            elif 3 in intervals and 7 in intervals:
                return Chord(root, ChordQuality.MINOR)
            elif 3 in intervals and 6 in intervals:
                return Chord(root, ChordQuality.DIMINISHED)
            elif 4 in intervals and 8 in intervals:
                return Chord(root, ChordQuality.AUGMENTED)
        
        # Dominant 7
        if 4 in intervals and 7 in intervals and 10 in intervals:
            return Chord(root, ChordQuality.DOMINANT7)
        
        # Major 7
        if 4 in intervals and 7 in intervals and 11 in intervals:
            return Chord(root, ChordQuality.MAJOR7)
        
        # Minor 7
        if 3 in intervals and 7 in intervals and 10 in intervals:
            return Chord(root, ChordQuality.MINOR7)
    
    return None


def estimate_key(chords: List[Chord]) -> Tuple[int, bool]:
    """
    Akor dizisinden tonal merkez tahmini.
    
    Args:
        chords: Akor listesi
        
    Returns:
        (tonal_merkez, minor_mi)
    """
    if not chords:
        return 0, False
    
    # Pitch class histogram
    pc_counts = Counter([c.root for c in chords])
    
    # Her tonal merkez için olasılık hesapla
    best_key = 0
    best_score = -1
    is_minor = False
    
    for tonic in range(12):
        # Major skor
        major_score = 0
        major_degrees = MAJOR_SCALE_DEGREES
        
        for chord in chords:
            interval = (chord.root - tonic) % 12
            if interval in major_degrees:
                major_score += 1
                # V ve I akorlarına ekstra puan
                if interval == 7:  # V
                    major_score += 2
                elif interval == 0:  # I
                    major_score += 3
        
        # Minor skor
        minor_score = 0
        minor_degrees = MINOR_SCALE_DEGREES
        
        for chord in chords:
            interval = (chord.root - tonic) % 12
            if interval in minor_degrees:
                minor_score += 1
                if interval == 7:  # V
                    minor_score += 2
                elif interval == 0:  # i
                    minor_score += 3
        
        # En iyi skoru seç
        if major_score > best_score:
            best_score = major_score
            best_key = tonic
            is_minor = False
        
        if minor_score > best_score:
            best_score = minor_score
            best_key = tonic
            is_minor = True
    
    return best_key, is_minor


def chord_to_roman(chord: Chord, key: int, is_minor: bool) -> str:
    """
    Akoru Roman numeral'a dönüştürür.
    
    Args:
        chord: Akor
        key: Tonal merkez (0-11)
        is_minor: Minor tonalite mi
        
    Returns:
        Roman numeral string
    """
    interval = (chord.root - key) % 12
    
    if is_minor:
        roman_map = ROMAN_MINOR
    else:
        roman_map = ROMAN_MAJOR
    
    roman = roman_map.get(interval, str(interval))
    
    # Kalite belirteci ekle
    if chord.quality in [ChordQuality.DIMINISHED, ChordQuality.DIMINISHED]:
        roman += "°"
    elif chord.quality == ChordQuality.AUGMENTED:
        roman += "+"
    elif chord.quality in [ChordQuality.DOMINANT7, ChordQuality.DOMINANT]:
        roman += "7"
    elif chord.quality == ChordQuality.MAJOR7:
        roman += "maj7"
    elif chord.quality == ChordQuality.MINOR7:
        roman += "m7"
    
    return roman


def extract_chord_progressions(
    events: List[NoteEvent],
    time_window: float = 0.1,
    min_chord_duration: float = 0.2
) -> List[ChordProgression]:
    """
    Nota olaylarından akor progression'ları çıkarır.
    
    Args:
        events: Nota olayları
        time_window: Aynı anda çalma toleransı
        min_chord_duration: Minimum akor süresi
        
    Returns:
        ChordProgression listesi
    """
    # Eşzamanlı notaları grupla
    note_groups = extract_simultaneous_notes(events, time_window)
    
    if len(note_groups) < 2:
        return []
    
    # Akorları tanımla
    chords_with_time = []
    for time, pitches in note_groups:
        chord = identify_chord(pitches)
        if chord:
            chords_with_time.append((time, chord))
    
    if len(chords_with_time) < 2:
        return []
    
    # Arka arkaya aynı akorları birleştir
    merged = [chords_with_time[0]]
    for time, chord in chords_with_time[1:]:
        last_time, last_chord = merged[-1]
        if chord.root == last_chord.root and chord.quality == last_chord.quality:
            # Aynı akor, birleştirme yapma (zamanı güncelle)
            continue
        else:
            merged.append((time, chord))
    
    chords = [c for _, c in merged]
    times = [t for t, _ in merged]
    
    if len(chords) < 2:
        return []
    
    # Tonal merkez tahmini
    key, is_minor = estimate_key(chords)
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key_name = key_names[key]
    if is_minor:
        key_name += "m"
    
    # Roman numeral'ları oluştur
    roman_numerals = [chord_to_roman(c, key, is_minor) for c in chords]
    
    # Progression segmentasyonu (uzun ve monolitik olanı parçalara ayır)
    progressions = []
    segment_size = min(16, len(chords))  # Maksimum 16 akor
    
    for i in range(0, len(chords), segment_size):
        end_idx = min(i + segment_size, len(chords))
        segment_chords = chords[i:end_idx]
        segment_roman = roman_numerals[i:end_idx]
        
        progression = ChordProgression(
            chords=segment_chords,
            roman_numerals=segment_roman,
            key=key,
            key_name=key_name,
            is_minor=is_minor,
            start_time=times[i],
            end_time=times[end_idx - 1] if end_idx <= len(times) else times[-1],
            confidence=0.7  # Basit tahmin
        )
        progressions.append(progression)
    
    return progressions


def find_progression_patterns(
    roman_numerals: List[str],
    min_length: int = 3
) -> List[ProgressionPattern]:
    """
    Roman numeral dizisinde bilinen pattern'leri bulur.
    
    Args:
        roman_numerals: Roman numeral listesi
        min_length: Minimum pattern uzunluğu
        
    Returns:
        ProgressionPattern listesi
    """
    patterns = []
    n = len(roman_numerals)
    
    # Temizleme: Sadece temel roman numerals (7, maj7, m7 gibi ekleri kaldır)
    clean_romans = []
    for r in roman_numerals:
        # Kalite belirteçlerini kaldır
        base = r.replace("°", "").replace("+", "").replace("7", "").replace("maj", "").replace("m", "")
        clean_romans.append(base)
    
    # Bilinen pattern'leri ara
    for pattern_name, pattern_sequence in KNOWN_PROGRESSIONS.items():
        occurrences = []
        pattern_len = len(pattern_sequence)
        
        for i in range(n - pattern_len + 1):
            window = clean_romans[i:i + pattern_len]
            if window == pattern_sequence:
                occurrences.append((i, i + pattern_len))
        
        if occurrences:
            frequency = len(occurrences) / max(1, n / pattern_len)
            patterns.append(ProgressionPattern(
                pattern_type=pattern_name,
                roman_sequence=pattern_sequence,
                occurrences=occurrences,
                frequency=frequency,
                contexts=[]
            ))
    
    # Circle of fifths ara
    circle_occurrences = []
    circle_len = len(CIRCLE_OF_FIFTHS_MAJOR)
    
    for i in range(n - circle_len + 1):
        window = clean_romans[i:i + circle_len]
        if window == CIRCLE_OF_FIFTHS_MAJOR or window == CIRCLE_OF_FIFTHS_MINOR:
            circle_occurrences.append((i, i + circle_len))
    
    if circle_occurrences:
        frequency = len(circle_occurrences) / max(1, n / circle_len)
        patterns.append(ProgressionPattern(
            pattern_type="CIRCLE_OF_FIFTHS",
            roman_sequence=CIRCLE_OF_FIFTHS_MAJOR,
            occurrences=circle_occurrences,
            frequency=frequency,
            contexts=[]
        ))
    
    # Frekansa göre sırala
    patterns.sort(key=lambda p: p.frequency, reverse=True)
    return patterns


def analyze_progression_complexity(progression: ChordProgression) -> Dict:
    """
    Progression karmaşıklık metrikleri.
    
    Args:
        progression: ChordProgression
        
    Returns:
        Karmaşıklık metrikleri
    """
    chords = progression.chords
    n = len(chords)
    
    if n < 2:
        return {"complexity": 0.0}
    
    # Benzersiz akor sayısı
    unique_chords = len(set([(c.root, c.quality) for c in chords]))
    uniqueness_ratio = unique_chords / n
    
    # Akor değişim hızı
    change_rate = (n - 1) / (progression.end_time - progression.start_time) if progression.end_time > progression.start_time else 0
    
    # Tonik dışı akor oranı
    non_tonic = sum(1 for c in chords if c.root != progression.key)
    non_tonic_ratio = non_tonic / n
    
    # Roman numeral çeşitliliği
    unique_roman = len(set(progression.roman_numerals))
    roman_diversity = unique_roman / min(n, 7)  # 7 scale degree var
    
    # Karmaşıklık skoru (0-1)
    complexity = (uniqueness_ratio * 0.3 + 
                  min(change_rate / 2, 1.0) * 0.2 + 
                  non_tonic_ratio * 0.3 + 
                  roman_diversity * 0.2)
    
    return {
        "complexity": round(complexity, 3),
        "unique_chords": unique_chords,
        "uniqueness_ratio": round(uniqueness_ratio, 3),
        "change_rate": round(change_rate, 3),
        "non_tonic_ratio": round(non_tonic_ratio, 3),
        "roman_diversity": round(roman_diversity, 3),
    }


def mine_chord_progressions(events: List[NoteEvent]) -> Dict:
    """
    Bir parça için kapsamlı akor progression analizi.
    
    Args:
        events: Nota olayları
        
    Returns:
        Analiz sonuçları
    """
    # Progression'ları çıkar
    progressions = extract_chord_progressions(events)
    
    if not progressions:
        return {
            "found": False,
            "message": "Yeterli akor verisi bulunamadı",
        }
    
    # Tüm roman numeral'ları birleştir
    all_romans = []
    for prog in progressions:
        all_romans.extend(prog.roman_numerals)
    
    # Pattern'leri bul
    patterns = find_progression_patterns(all_romans)
    
    # Karmaşıklık analizi
    complexity_metrics = []
    for prog in progressions:
        metrics = analyze_progression_complexity(prog)
        complexity_metrics.append(metrics)
    
    avg_complexity = np.mean([m["complexity"] for m in complexity_metrics]) if complexity_metrics else 0
    
    # Sonuçları hazırla
    return {
        "found": True,
        "key": progressions[0].key_name if progressions else None,
        "is_minor": progressions[0].is_minor if progressions else None,
        "total_chords": len(all_romans),
        "num_progressions": len(progressions),
        "avg_complexity": round(avg_complexity, 3),
        "progressions": [
            {
                "chords": [str(c) for c in prog.chords],
                "roman_numerals": prog.roman_numerals,
                "start_time": round(prog.start_time, 2),
                "end_time": round(prog.end_time, 2),
            }
            for prog in progressions
        ],
        "patterns": [
            {
                "type": p.pattern_type,
                "sequence": p.roman_sequence,
                "occurrences": len(p.occurrences),
                "frequency": round(p.frequency, 3),
            }
            for p in patterns[:10]  # Top 10 pattern
        ],
        "top_pattern": patterns[0].pattern_type if patterns else None,
    }


__all__ = [
    "Chord",
    "ChordQuality",
    "ChordProgression",
    "ProgressionPattern",
    "KNOWN_PROGRESSIONS",
    "extract_simultaneous_notes",
    "identify_chord",
    "estimate_key",
    "chord_to_roman",
    "extract_chord_progressions",
    "find_progression_patterns",
    "analyze_progression_complexity",
    "mine_chord_progressions",
]
