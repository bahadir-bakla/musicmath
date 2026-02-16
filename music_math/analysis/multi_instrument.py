"""
Multi-Instrument Transition Analyzer - Çoklu Enstrüman Analizi

MIDI dosyalarındaki farklı enstrümanların (track/channel) giriş-çıkış 
zamanlarını, geçişlerini ve etkileyiciliklerini analiz eder.

Özellikler:
- Enstrüman giriş/çıkış tespiti
- Enstrüman geçiş analizi
- Duygusal etki skoru
- Layer kalınlığı analizi
- Solo vs tutti tespiti

Kullanım:
    from music_math.analysis.multi_instrument import (
        analyze_instrument_transitions,
        calculate_emotional_impact,
        extract_instrument_features,
    )
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from music_math.core.types import NoteEvent
from music_math.data.loader import parse_midi_to_note_events


class InstrumentFamily(Enum):
    """Enstrüman aileleri."""
    STRINGS = "strings"
    WOODWINDS = "woodwinds"
    BRASS = "brass"
    PERCUSSION = "percussion"
    KEYBOARD = "keyboard"
    PLUCKED = "plucked"
    UNKNOWN = "unknown"


@dataclass
class InstrumentEntry:
    """Enstrüman giriş/çıkış bilgisi."""
    track_idx: int
    channel: int
    program: int  # MIDI program number
    instrument_name: str
    family: InstrumentFamily
    entry_time: float
    exit_time: float
    note_count: int
    total_duration: float
    avg_velocity: float
    pitch_range: Tuple[int, int]  # (min, max)


@dataclass
class InstrumentTransition:
    """Enstrüman geçişi."""
    time: float
    exiting_instruments: List[InstrumentEntry]
    entering_instruments: List[InstrumentEntry]
    continuing_instruments: List[InstrumentEntry]
    transition_type: str  # "solo_to_tutti", "tutti_to_solo", "layer_change", etc.
    impact_score: float  # 0-1 arası etki
    velocity_change: float  # Hız değişimi
    density_change: float  # Nota yoğunluğu değişimi


@dataclass
class InstrumentLayer:
    """Zaman dilimindeki enstrüman katmanı."""
    start_time: float
    end_time: float
    instruments: List[InstrumentEntry]
    layer_thickness: int  # Eşzamanlı enstrüman sayısı
    is_tutti: bool  # Tüm enstrümanlar çalıyor mu?
    is_solo: bool  # Tek enstrüman mı?
    dominant_family: InstrumentFamily


@dataclass
class EmotionalImpact:
    """Duygusal etki analizi."""
    time: float
    impact_type: str  # "entrance", "exit", "transition"
    intensity: float  # 0-1 şiddet
    description: str
    contributing_factors: List[str]


@dataclass
class MultiInstrumentAnalysis:
    """Çoklu enstrüman analizi sonuçları."""
    instruments: List[InstrumentEntry]
    transitions: List[InstrumentTransition]
    layers: List[InstrumentLayer]
    emotional_impacts: List[EmotionalImpact]
    total_instruments: int
    max_simultaneous: int
    solo_sections: List[Tuple[float, float]]
    tutti_sections: List[Tuple[float, float]]
    instrument_families: Dict[InstrumentFamily, int]


# MIDI Program to Instrument mapping (simplified)
PROGRAM_TO_INSTRUMENT = {
    # Piano (0-7)
    0: ("Acoustic Grand Piano", InstrumentFamily.KEYBOARD),
    1: ("Bright Acoustic Piano", InstrumentFamily.KEYBOARD),
    2: ("Electric Grand Piano", InstrumentFamily.KEYBOARD),
    # ... more piano variants
    
    # Chromatic Percussion (8-15)
    8: ("Celesta", InstrumentFamily.PERCUSSION),
    9: ("Glockenspiel", InstrumentFamily.PERCUSSION),
    12: ("Marimba", InstrumentFamily.PERCUSSION),
    13: ("Xylophone", InstrumentFamily.PERCUSSION),
    
    # Organ (16-23)
    16: ("Hammond Organ", InstrumentFamily.KEYBOARD),
    19: ("Church Organ", InstrumentFamily.KEYBOARD),
    
    # Guitar (24-31)
    24: ("Acoustic Guitar (nylon)", InstrumentFamily.PLUCKED),
    25: ("Acoustic Guitar (steel)", InstrumentFamily.PLUCKED),
    26: ("Electric Guitar (jazz)", InstrumentFamily.PLUCKED),
    27: ("Electric Guitar (clean)", InstrumentFamily.PLUCKED),
    
    # Bass (32-39)
    32: ("Acoustic Bass", InstrumentFamily.STRINGS),
    33: ("Electric Bass (finger)", InstrumentFamily.STRINGS),
    34: ("Electric Bass (pick)", InstrumentFamily.STRINGS),
    
    # Strings (40-47)
    40: ("Violin", InstrumentFamily.STRINGS),
    41: ("Viola", InstrumentFamily.STRINGS),
    42: ("Cello", InstrumentFamily.STRINGS),
    43: ("Contrabass", InstrumentFamily.STRINGS),
    44: ("Tremolo Strings", InstrumentFamily.STRINGS),
    45: ("Pizzicato Strings", InstrumentFamily.STRINGS),
    46: ("Orchestral Harp", InstrumentFamily.STRINGS),
    
    # Ensemble (48-55)
    48: ("String Ensemble 1", InstrumentFamily.STRINGS),
    49: ("String Ensemble 2", InstrumentFamily.STRINGS),
    50: ("Synth Strings 1", InstrumentFamily.STRINGS),
    52: ("Choir Aahs", InstrumentFamily.STRINGS),
    53: ("Voice Oohs", InstrumentFamily.STRINGS),
    
    # Brass (56-63)
    56: ("Trumpet", InstrumentFamily.BRASS),
    57: ("Trombone", InstrumentFamily.BRASS),
    58: ("Tuba", InstrumentFamily.BRASS),
    59: ("Muted Trumpet", InstrumentFamily.BRASS),
    60: ("French Horn", InstrumentFamily.BRASS),
    61: ("Brass Section", InstrumentFamily.BRASS),
    
    # Reed (64-71)
    64: ("Soprano Sax", InstrumentFamily.WOODWINDS),
    65: ("Alto Sax", InstrumentFamily.WOODWINDS),
    66: ("Tenor Sax", InstrumentFamily.WOODWINDS),
    68: ("Oboe", InstrumentFamily.WOODWINDS),
    69: ("English Horn", InstrumentFamily.WOODWINDS),
    70: ("Bassoon", InstrumentFamily.WOODWINDS),
    71: ("Clarinet", InstrumentFamily.WOODWINDS),
    
    # Pipe (72-79)
    72: ("Piccolo", InstrumentFamily.WOODWINDS),
    73: ("Flute", InstrumentFamily.WOODWINDS),
    74: ("Recorder", InstrumentFamily.WOODWINDS),
    75: ("Pan Flute", InstrumentFamily.WOODWINDS),
    
    # Synth Lead (80-87)
    80: ("Lead 1 (square)", InstrumentFamily.KEYBOARD),
    81: ("Lead 2 (sawtooth)", InstrumentFamily.KEYBOARD),
    
    # Synth Pad (88-95)
    88: ("Pad 1 (new age)", InstrumentFamily.KEYBOARD),
    89: ("Pad 2 (warm)", InstrumentFamily.KEYBOARD),
    
    # Drums
    118: ("Synth Drum", InstrumentFamily.PERCUSSION),
}


def get_instrument_info(program: int) -> Tuple[str, InstrumentFamily]:
    """MIDI program numarasından enstrüman bilgisi."""
    if program in PROGRAM_TO_INSTRUMENT:
        return PROGRAM_TO_INSTRUMENT[program]
    elif program >= 118:  # Percussion
        return (f"Drum/FX ({program})", InstrumentFamily.PERCUSSION)
    elif program >= 80:
        return (f"Synth ({program})", InstrumentFamily.KEYBOARD)
    elif program >= 40:
        return (f"String ({program})", InstrumentFamily.STRINGS)
    else:
        return (f"Unknown ({program})", InstrumentFamily.UNKNOWN)


def extract_instruments_from_midi(filepath: str) -> List[InstrumentEntry]:
    """
    MIDI dosyasından enstrüman bilgilerini çıkarır.
    
    Args:
        filepath: MIDI dosya yolu
        
    Returns:
        InstrumentEntry listesi
    """
    try:
        import mido
    except ImportError:
        raise ImportError("mido required: pip install mido")
    
    mid = mido.MidiFile(filepath)
    instruments = []
    
    for track_idx, track in enumerate(mid.tracks):
        current_channel = None
        current_program = 0
        notes_on = defaultdict(list)  # channel -> [(note, velocity, time), ...]
        note_events = []
        
        absolute_time = 0
        
        for msg in track:
            absolute_time += msg.time
            
            if msg.type == 'program_change':
                current_channel = msg.channel
                current_program = msg.program
                
            elif msg.type == 'note_on' and msg.velocity > 0:
                notes_on[msg.channel].append((msg.note, msg.velocity, absolute_time))
                
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                channel = msg.channel
                note = msg.note
                
                # Matching note_on bul
                for i, (n, vel, start) in enumerate(notes_on[channel]):
                    if n == note:
                        note_events.append({
                            'track': track_idx,
                            'channel': channel,
                            'program': current_program,
                            'note': note,
                            'velocity': vel,
                            'start': start,
                            'end': absolute_time,
                            'duration': absolute_time - start
                        })
                        notes_on[channel].pop(i)
                        break
        
        # Enstrüman istatistiklerini hesapla
        if note_events:
            channels = set(e['channel'] for e in note_events)
            
            for channel in channels:
                channel_notes = [e for e in note_events if e['channel'] == channel]
                
                if channel_notes:
                    program = channel_notes[0]['program']
                    name, family = get_instrument_info(program)
                    
                    entry = InstrumentEntry(
                        track_idx=track_idx,
                        channel=channel,
                        program=program,
                        instrument_name=name,
                        family=family,
                        entry_time=min(e['start'] for e in channel_notes),
                        exit_time=max(e['end'] for e in channel_notes),
                        note_count=len(channel_notes),
                        total_duration=sum(e['duration'] for e in channel_notes),
                        avg_velocity=np.mean([e['velocity'] for e in channel_notes]),
                        pitch_range=(
                            min(e['note'] for e in channel_notes),
                            max(e['note'] for e in channel_notes)
                        )
                    )
                    instruments.append(entry)
    
    return instruments


def detect_instrument_transitions(
    instruments: List[InstrumentEntry],
    time_resolution: float = 0.5
) -> List[InstrumentTransition]:
    """
    Enstrüman geçişlerini tespit eder.
    
    Args:
        instruments: Enstrüman listesi
        time_resolution: Zaman çözünürlüğü (saniye)
        
    Returns:
        InstrumentTransition listesi
    """
    if not instruments:
        return []
    
    # Zaman çizelgesi oluştur
    all_times = set()
    for inst in instruments:
        all_times.add(inst.entry_time)
        all_times.add(inst.exit_time)
    
    sorted_times = sorted(all_times)
    transitions = []
    
    for i, time in enumerate(sorted_times):
        # Bu zamanda ne oluyor?
        entering = [inst for inst in instruments if abs(inst.entry_time - time) < time_resolution]
        exiting = [inst for inst in instruments if abs(inst.exit_time - time) < time_resolution]
        
        # Devam eden enstrümanlar
        continuing = [
            inst for inst in instruments
            if inst.entry_time < time and inst.exit_time > time
        ]
        
        if entering or exiting:
            # Geçiş tipi
            before_count = len(continuing) + len(exiting)
            after_count = len(continuing) + len(entering)
            
            if before_count == 1 and after_count > 1:
                trans_type = "solo_to_tutti"
            elif before_count > 1 and after_count == 1:
                trans_type = "tutti_to_solo"
            elif before_count < after_count:
                trans_type = "thickening"
            elif before_count > after_count:
                trans_type = "thinning"
            else:
                trans_type = "voice_change"
            
            # Etki skoru hesapla
            impact = calculate_transition_impact(entering, exiting, continuing)
            
            transitions.append(InstrumentTransition(
                time=time,
                exiting_instruments=exiting,
                entering_instruments=entering,
                continuing_instruments=continuing,
                transition_type=trans_type,
                impact_score=impact['score'],
                velocity_change=impact['velocity_change'],
                density_change=impact['density_change']
            ))
    
    return transitions


def calculate_transition_impact(
    entering: List[InstrumentEntry],
    exiting: List[InstrumentEntry],
    continuing: List[InstrumentEntry]
) -> Dict:
    """
    Geçişin etki skorunu hesaplar.
    
    Returns:
        {'score': float, 'velocity_change': float, 'density_change': float}
    """
    score = 0.0
    
    # Hız (velocity) değişimi
    if entering and (exiting or continuing):
        enter_vel = np.mean([e.avg_velocity for e in entering])
        exit_vel = np.mean([e.avg_velocity for e in exiting + continuing]) if (exiting + continuing) else enter_vel
        velocity_change = (enter_vel - exit_vel) / 127  # Normalize
        score += abs(velocity_change) * 0.3
    else:
        velocity_change = 0.0
    
    # Yoğunluk değişimi
    density_change = len(entering) - len(exiting)
    score += min(abs(density_change) * 0.2, 0.4)
    
    # Enstrüman ailesi değişimi
    enter_families = set(e.family for e in entering)
    exit_families = set(e.family for e in exiting)
    if enter_families != exit_families:
        score += 0.2
    
    # Yüksek tınılı enstrüman girişi (dikkat çekici)
    high_attention = [InstrumentFamily.BRASS, InstrumentFamily.PERCUSSION]
    for e in entering:
        if e.family in high_attention:
            score += 0.15
    
    return {
        'score': min(1.0, score),
        'velocity_change': velocity_change,
        'density_change': density_change
    }


def analyze_instrument_layers(
    instruments: List[InstrumentEntry],
    transitions: List[InstrumentTransition]
) -> List[InstrumentLayer]:
    """
    Enstrüman katmanlarını analiz eder.
    
    Args:
        instruments: Enstrüman listesi
        transitions: Geçiş listesi
        
    Returns:
        InstrumentLayer listesi
    """
    if not transitions:
        return []
    
    layers = []
    
    for i in range(len(transitions) - 1):
        start_time = transitions[i].time
        end_time = transitions[i + 1].time
        
        # Bu zaman aralığındaki aktif enstrümanlar
        active = [
            inst for inst in instruments
            if inst.entry_time <= end_time and inst.exit_time >= start_time
        ]
        
        if active:
            families = [inst.family for inst in active]
            dominant = max(set(families), key=families.count)
            
            layer = InstrumentLayer(
                start_time=start_time,
                end_time=end_time,
                instruments=active,
                layer_thickness=len(active),
                is_tutti=len(active) >= 4,
                is_solo=len(active) == 1,
                dominant_family=dominant
            )
            layers.append(layer)
    
    return layers


def calculate_emotional_impact(
    transitions: List[InstrumentTransition]
) -> List[EmotionalImpact]:
    """
    Geçişlerin duygusal etkilerini analiz eder.
    
    Args:
        transitions: Geçiş listesi
        
    Returns:
        EmotionalImpact listesi
    """
    impacts = []
    
    for trans in transitions:
        factors = []
        
        # Etki tipi
        if trans.entering_instruments and not trans.exiting_instruments:
            impact_type = "entrance"
            desc = "New instruments entering"
        elif trans.exiting_instruments and not trans.entering_instruments:
            impact_type = "exit"
            desc = "Instruments exiting"
        else:
            impact_type = "transition"
            desc = "Instrument change"
        
        # Faktörler
        if trans.impact_score > 0.7:
            factors.append("High dramatic impact")
        
        if trans.transition_type == "solo_to_tutti":
            factors.append("Building intensity")
            desc = "Dramatic buildup: Solo to full ensemble"
        elif trans.transition_type == "tutti_to_solo":
            factors.append("Intimate moment")
            desc = "Focus shift: Full ensemble to solo"
        
        if trans.velocity_change > 0.2:
            factors.append("Louder dynamics")
        elif trans.velocity_change < -0.2:
            factors.append("Softer dynamics")
        
        # Özel enstrümanlar
        for inst in trans.entering_instruments:
            if inst.family == InstrumentFamily.BRASS:
                factors.append("Brass entrance (heroic)")
            elif inst.family == InstrumentFamily.STRINGS and len(trans.entering_instruments) > 3:
                factors.append("String section entrance")
        
        impacts.append(EmotionalImpact(
            time=trans.time,
            impact_type=impact_type,
            intensity=trans.impact_score,
            description=desc,
            contributing_factors=factors
        ))
    
    return impacts


def analyze_instrument_transitions(filepath: str) -> MultiInstrumentAnalysis:
    """
    MIDI dosyası için kapsamlı çoklu enstrüman analizi.
    
    Args:
        filepath: MIDI dosya yolu
        
    Returns:
        MultiInstrumentAnalysis
    """
    # Enstrümanları çıkar
    instruments = extract_instruments_from_midi(filepath)
    
    if not instruments:
        return MultiInstrumentAnalysis(
            instruments=[],
            transitions=[],
            layers=[],
            emotional_impacts=[],
            total_instruments=0,
            max_simultaneous=0,
            solo_sections=[],
            tutti_sections=[],
            instrument_families={}
        )
    
    # Geçişleri tespit et
    transitions = detect_instrument_transitions(instruments)
    
    # Katmanları analiz et
    layers = analyze_instrument_layers(instruments, transitions)
    
    # Duygusal etki
    emotional_impacts = calculate_emotional_impact(transitions)
    
    # Solo ve tutti bölümleri
    solo_sections = [(l.start_time, l.end_time) for l in layers if l.is_solo]
    tutti_sections = [(l.start_time, l.end_time) for l in layers if l.is_tutti]
    
    # Enstrüman aileleri
    families = defaultdict(int)
    for inst in instruments:
        families[inst.family] += 1
    
    return MultiInstrumentAnalysis(
        instruments=instruments,
        transitions=transitions,
        layers=layers,
        emotional_impacts=emotional_impacts,
        total_instruments=len(instruments),
        max_simultaneous=max((l.layer_thickness for l in layers), default=0),
        solo_sections=solo_sections,
        tutti_sections=tutti_sections,
        instrument_families=dict(families)
    )


def extract_instrument_features(filepath: str) -> Dict[str, float]:
    """
    Feature extraction için enstrüman özellikleri.
    
    Args:
        filepath: MIDI dosya yolu
        
    Returns:
        Feature sözlüğü
    """
    analysis = analyze_instrument_transitions(filepath)
    
    features = {
        "num_instruments": analysis.total_instruments,
        "max_simultaneous_instruments": analysis.max_simultaneous,
        "num_transitions": len(analysis.transitions),
        "num_solo_sections": len(analysis.solo_sections),
        "num_tutti_sections": len(analysis.tutti_sections),
        "solo_ratio": len(analysis.solo_sections) / max(1, len(analysis.layers)),
        "tutti_ratio": len(analysis.tutti_sections) / max(1, len(analysis.layers)),
    }
    
    # Ortalama etki skoru
    if analysis.emotional_impacts:
        avg_impact = np.mean([e.intensity for e in analysis.emotional_impacts])
        features["avg_transition_impact"] = round(avg_impact, 3)
    else:
        features["avg_transition_impact"] = 0.0
    
    # Enstrüman çeşitliliği
    features["instrument_family_diversity"] = len(analysis.instrument_families)
    
    # Katman karmaşıklığı
    if analysis.layers:
        avg_thickness = np.mean([l.layer_thickness for l in analysis.layers])
        features["avg_layer_thickness"] = round(avg_thickness, 2)
    else:
        features["avg_layer_thickness"] = 0.0
    
    return features


__all__ = [
    "InstrumentFamily",
    "InstrumentEntry",
    "InstrumentTransition",
    "InstrumentLayer",
    "EmotionalImpact",
    "MultiInstrumentAnalysis",
    "extract_instruments_from_midi",
    "detect_instrument_transitions",
    "analyze_instrument_layers",
    "calculate_emotional_impact",
    "analyze_instrument_transitions",
    "extract_instrument_features",
]
