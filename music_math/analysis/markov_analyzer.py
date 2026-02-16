"""
Markov Chain Analyzer - Müzik Geçiş Olasılıkları Analizi

Notalar arası geçiş olasılıklarını Markov zincirleriyle modeller:
- Pitch transition matrix
- Duration transition matrix
- Interval transition matrix
- Higher-order Markov chains (2nd, 3rd order)
- Entropy ve predictability analizi

Kullanım:
    from music_math.analysis.markov_analyzer import (
        build_transition_matrix,
        calculate_entropy,
        analyze_markov_properties,
        predict_next_notes,
    )
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass

from music_math.core.types import NoteEvent


@dataclass
class MarkovModel:
    """Markov zinciri modeli."""
    order: int
    states: List[str]
    transition_matrix: np.ndarray
    state_to_idx: Dict[str, int]
    entropy: float
    stationary_distribution: Optional[np.ndarray] = None


@dataclass
class TransitionPattern:
    """Geçiş pattern'i."""
    from_state: str
    to_state: str
    probability: float
    frequency: int
    surprise: float  # -log(p), ne kadar beklenmedik


@dataclass
class MarkovAnalysis:
    """Markov analizi sonuçları."""
    pitch_model: MarkovModel
    duration_model: MarkovModel
    interval_model: MarkovModel
    predictability: float
    most_common_transitions: List[TransitionPattern]
    rare_transitions: List[TransitionPattern]
    repeating_patterns: List[str]


def build_transition_matrix(
    sequence: List,
    order: int = 1,
    state_space: Optional[List] = None
) -> Tuple[np.ndarray, List, Dict]:
    """
    Geçiş matrisi oluşturur.
    
    Args:
        sequence: Durum dizisi
        order: Markov zinciri derecesi (1, 2, 3...)
        state_space: Önceden tanımlı durum uzayı (None ise sequence'dan çıkar)
        
    Returns:
        (transition_matrix, states, state_to_idx)
        
    Example:
        >>> pitches = [60, 62, 64, 62, 60, 62, 64]
        >>> matrix, states, mapping = build_transition_matrix(pitches, order=1)
    """
    if len(sequence) < order + 1:
        return np.array([]), [], {}
    
    # Durum uzayını belirle
    if state_space is None:
        state_space = sorted(set(sequence))
    
    states = [str(s) for s in state_space]
    state_to_idx = {s: i for i, s in enumerate(states)}
    n_states = len(states)
    
    if n_states == 0:
        return np.array([]), [], {}
    
    # Order > 1 için n-gram'ları durum olarak kullan
    if order == 1:
        # Basit 1st-order
        transition_counts = np.zeros((n_states, n_states))
        
        for i in range(len(sequence) - 1):
            from_state = str(sequence[i])
            to_state = str(sequence[i + 1])
            
            if from_state in state_to_idx and to_state in state_to_idx:
                from_idx = state_to_idx[from_state]
                to_idx = state_to_idx[to_state]
                transition_counts[from_idx, to_idx] += 1
    else:
        # Higher-order: n-gram'lar
        ngrams = []
        for i in range(len(sequence) - order):
            ngram = tuple(sequence[i:i + order + 1])
            ngrams.append(ngram)
        
        # N-gram durum uzayı
        unique_ngrams = sorted(set(ngrams))
        states = [str(ng) for ng in unique_ngrams]
        state_to_idx = {s: i for i, s in enumerate(states)}
        n_states = len(states)
        
        # Higher-order transition matrix (basitleştirilmiş)
        # Gerçek higher-order daha karmaşık, burada order-1 gibi davranıyoruz
        transition_counts = np.zeros((n_states, n_states))
        
        for i in range(len(ngrams) - 1):
            from_state = str(ngrams[i])
            to_state = str(ngrams[i + 1])
            
            if from_state in state_to_idx and to_state in state_to_idx:
                from_idx = state_to_idx[from_state]
                to_idx = state_to_idx[to_state]
                transition_counts[from_idx, to_idx] += 1
    
    # Olasılıklara normalize et (Laplace smoothing)
    transition_matrix = np.zeros((n_states, n_states))
    for i in range(n_states):
        row_sum = np.sum(transition_counts[i])
        if row_sum > 0:
            # Laplace smoothing
            transition_matrix[i] = (transition_counts[i] + 1) / (row_sum + n_states)
        else:
            # Hiç geçiş yok, uniform dağılım
            transition_matrix[i] = 1.0 / n_states
    
    return transition_matrix, states, state_to_idx


def calculate_entropy(transition_matrix: np.ndarray) -> float:
    """
    Markov zinciri entropisi (tahmin edilebilirlik).
    
    Args:
        transition_matrix: Geçiş matrisi
        
    Returns:
        Entropi değeri (bit)
    """
    if transition_matrix.size == 0:
        return 0.0
    
    # Stationary distribution ( dengede dağılım)
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary = eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))]
    stationary = stationary / np.sum(stationary)  # Normalize
    stationary = np.real(stationary)
    
    # Entropy rate: H = -sum_i pi * sum_j p_ij * log(p_ij)
    entropy = 0.0
    for i in range(len(transition_matrix)):
        for j in range(len(transition_matrix)):
            p = transition_matrix[i, j]
            if p > 0:
                entropy -= stationary[i] * p * np.log2(p)
    
    return float(entropy)


def calculate_predictability(transition_matrix: np.ndarray) -> float:
    """
    Tahmin edilebilirlik skoru [0, 1].
    
    1 = Tamamen tahmin edilebilir (deterministik)
    0 = Tamamen rastgele (uniform)
    
    Args:
        transition_matrix: Geçiş matrisi
        
    Returns:
        Tahmin edilebilirlik skoru
    """
    if transition_matrix.size == 0:
        return 0.0
    
    n_states = len(transition_matrix)
    max_entropy = np.log2(n_states) if n_states > 1 else 1.0
    
    entropy = calculate_entropy(transition_matrix)
    
    # Normalize: 1 - (entropy / max_entropy)
    predictability = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
    
    return max(0.0, min(1.0, predictability))


def find_common_transitions(
    sequence: List,
    min_frequency: int = 2
) -> List[TransitionPattern]:
    """
    En yaygın geçiş pattern'lerini bulur.
    
    Args:
        sequence: Durum dizisi
        min_frequency: Minimum görülme sayısı
        
    Returns:
        TransitionPattern listesi
    """
    if len(sequence) < 2:
        return []
    
    # Geçişleri say
    transition_counts = Counter()
    for i in range(len(sequence) - 1):
        transition = (str(sequence[i]), str(sequence[i + 1]))
        transition_counts[transition] += 1
    
    # Toplam geçiş sayısı
    total = sum(transition_counts.values())
    
    # Pattern'leri oluştur
    patterns = []
    for (from_state, to_state), count in transition_counts.items():
        if count >= min_frequency:
            prob = count / total
            surprise = -np.log2(prob) if prob > 0 else float('inf')
            patterns.append(TransitionPattern(
                from_state=from_state,
                to_state=to_state,
                probability=prob,
                frequency=count,
                surprise=surprise
            ))
    
    # Sıklığa göre sırala
    patterns.sort(key=lambda p: p.frequency, reverse=True)
    return patterns


def find_repeating_patterns(
    sequence: List,
    min_length: int = 3,
    min_occurrences: int = 2
) -> List[str]:
    """
    Tekrar eden pattern'leri bulur.
    
    Args:
        sequence: Durum dizisi
        min_length: Minimum pattern uzunluğu
        min_occurrences: Minimum tekrar sayısı
        
    Returns:
        Pattern listesi (string olarak)
    """
    if len(sequence) < min_length:
        return []
    
    patterns = []
    
    for length in range(min_length, min(len(sequence) // 2 + 1, 20)):
        pattern_counts = Counter()
        
        for i in range(len(sequence) - length + 1):
            pattern = tuple(sequence[i:i + length])
            pattern_counts[pattern] += 1
        
        for pattern, count in pattern_counts.items():
            if count >= min_occurrences:
                pattern_str = "-".join(map(str, pattern))
                patterns.append(pattern_str)
    
    # Benzersiz pattern'leri döndür
    return list(set(patterns))


def find_rare_transitions(
    transition_patterns: List[TransitionPattern],
    threshold: float = 0.1
) -> List[TransitionPattern]:
    """
    Nadir geçişleri bulur (low probability).
    
    Args:
        transition_patterns: Geçiş pattern'leri
        threshold: Olasılık eşiği (altındakiler nadir)
        
    Returns:
        Nadir pattern'ler
    """
    rare = [p for p in transition_patterns if p.probability < threshold]
    # Surprise'a göre sırala (en beklenmedik başta)
    rare.sort(key=lambda p: p.surprise, reverse=True)
    return rare[:20]  # En fazla 20


def analyze_pitch_transitions(events: List[NoteEvent], order: int = 1) -> MarkovModel:
    """
    Pitch geçişlerini Markov modeli olarak analiz eder.
    
    Args:
        events: Nota olayları
        order: Markov derecesi
        
    Returns:
        MarkovModel
    """
    pitches = [e.pitch for e in events]
    
    if len(pitches) < order + 1:
        return MarkovModel(
            order=order,
            states=[],
            transition_matrix=np.array([]),
            state_to_idx={},
            entropy=0.0
        )
    
    matrix, states, state_to_idx = build_transition_matrix(pitches, order)
    entropy = calculate_entropy(matrix) if matrix.size > 0 else 0.0
    
    return MarkovModel(
        order=order,
        states=states,
        transition_matrix=matrix,
        state_to_idx=state_to_idx,
        entropy=entropy
    )


def analyze_duration_transitions(events: List[NoteEvent], order: int = 1) -> MarkovModel:
    """
    Süre geçişlerini analiz eder.
    
    Args:
        events: Nota olayları
        order: Markov derecesi
        
    Returns:
        MarkovModel
    """
    # Süreleri kategorilere ayır (quantize)
    durations = [e.duration for e in events]
    
    # Basit quantizasyon: 16'lık notalar
    quantized = []
    for d in durations:
        if d < 0.25:
            quantized.append("16th")
        elif d < 0.5:
            quantized.append("8th")
        elif d < 1.0:
            quantized.append("quarter")
        elif d < 2.0:
            quantized.append("half")
        else:
            quantized.append("whole")
    
    if len(quantized) < order + 1:
        return MarkovModel(
            order=order,
            states=[],
            transition_matrix=np.array([]),
            state_to_idx={},
            entropy=0.0
        )
    
    matrix, states, state_to_idx = build_transition_matrix(quantized, order)
    entropy = calculate_entropy(matrix) if matrix.size > 0 else 0.0
    
    return MarkovModel(
        order=order,
        states=states,
        transition_matrix=matrix,
        state_to_idx=state_to_idx,
        entropy=entropy
    )


def analyze_interval_transitions(events: List[NoteEvent], order: int = 1) -> MarkovModel:
    """
    Aralık geçişlerini analiz eder.
    
    Args:
        events: Nota olayları
        order: Markov derecesi
        
    Returns:
        MarkovModel
    """
    pitches = [e.pitch for e in events]
    if len(pitches) < 2:
        return MarkovModel(
            order=order,
            states=[],
            transition_matrix=np.array([]),
            state_to_idx={},
            entropy=0.0
        )
    
    intervals = list(np.diff(pitches))
    
    if len(intervals) < order + 1:
        return MarkovModel(
            order=order,
            states=[],
            transition_matrix=np.array([]),
            state_to_idx={},
            entropy=0.0
        )
    
    matrix, states, state_to_idx = build_transition_matrix(intervals, order)
    entropy = calculate_entropy(matrix) if matrix.size > 0 else 0.0
    
    return MarkovModel(
        order=order,
        states=states,
        transition_matrix=matrix,
        state_to_idx=state_to_idx,
        entropy=entropy
    )


def predict_next_notes(
    model: MarkovModel,
    current_state: str,
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Sonraki notaları tahmin eder.
    
    Args:
        model: Markov modeli
        current_state: Mevcut durum
        top_k: Döndürülecek tahmin sayısı
        
    Returns:
        [(durum, olasılık), ...] listesi
    """
    if current_state not in model.state_to_idx:
        return []
    
    idx = model.state_to_idx[current_state]
    probs = model.transition_matrix[idx]
    
    # En yüksek olasılıklıları seç
    top_indices = np.argsort(probs)[-top_k:][::-1]
    
    predictions = []
    for i in top_indices:
        if probs[i] > 0:
            predictions.append((model.states[i], float(probs[i])))
    
    return predictions


def analyze_markov_properties(events: List[NoteEvent], order: int = 1) -> MarkovAnalysis:
    """
    Kapsamlı Markov zinciri analizi.
    
    Args:
        events: Nota olayları
        order: Markov derecesi
        
    Returns:
        MarkovAnalysis sonuçları
    """
    # Her özellik için model oluştur
    pitch_model = analyze_pitch_transitions(events, order)
    duration_model = analyze_duration_transitions(events, order)
    interval_model = analyze_interval_transitions(events, order)
    
    # Tahmin edilebilirlik
    pitch_pred = calculate_predictability(pitch_model.transition_matrix) if pitch_model.transition_matrix.size > 0 else 0.0
    interval_pred = calculate_predictability(interval_model.transition_matrix) if interval_model.transition_matrix.size > 0 else 0.0
    
    # Genel tahmin edilebilirlik (ağırlıklı ortalama)
    predictability = (pitch_pred * 0.5 + interval_pred * 0.5)
    
    # Yaygın geçişler
    pitches = [e.pitch for e in events]
    common_transitions = find_common_transitions(pitches, min_frequency=2)
    
    # Nadir geçişler
    rare_transitions = find_rare_transitions(common_transitions, threshold=0.05)
    
    # Tekrar eden pattern'ler
    repeating_patterns = find_repeating_patterns(pitches, min_length=3, min_occurrences=2)
    
    return MarkovAnalysis(
        pitch_model=pitch_model,
        duration_model=duration_model,
        interval_model=interval_model,
        predictability=predictability,
        most_common_transitions=common_transitions[:20],
        rare_transitions=rare_transitions,
        repeating_patterns=repeating_patterns[:10]
    )


def compare_markov_models(model1: MarkovModel, model2: MarkovModel) -> Dict:
    """
    İki Markov modelini karşılaştırır.
    
    Args:
        model1: Birinci model
        model2: İkinci model
        
    Returns:
        Karşılaştırma metrikleri
    """
    # Entropy farkı
    entropy_diff = abs(model1.entropy - model2.entropy)
    
    # State uzayı boyutu
    state_diff = abs(len(model1.states) - len(model2.states))
    
    # Tahmin edilebilirlik
    pred1 = calculate_predictability(model1.transition_matrix) if model1.transition_matrix.size > 0 else 0.0
    pred2 = calculate_predictability(model2.transition_matrix) if model2.transition_matrix.size > 0 else 0.0
    pred_diff = abs(pred1 - pred2)
    
    return {
        "entropy_difference": round(entropy_diff, 3),
        "state_space_difference": state_diff,
        "predictability_difference": round(pred_diff, 3),
        "model1_entropy": round(model1.entropy, 3),
        "model2_entropy": round(model2.entropy, 3),
        "model1_predictability": round(pred1, 3),
        "model2_predictability": round(pred2, 3),
    }


def extract_markov_features(events: List[NoteEvent]) -> Dict[str, float]:
    """
    Feature extraction için Markov özellikleri.
    
    Args:
        events: Nota olayları
        
    Returns:
        Feature sözlüğü
    """
    analysis = analyze_markov_properties(events, order=1)
    
    features = {
        "markov_pitch_entropy": round(analysis.pitch_model.entropy, 3),
        "markov_interval_entropy": round(analysis.interval_model.entropy, 3),
        "markov_predictability": round(analysis.predictability, 3),
        "markov_pitch_states": len(analysis.pitch_model.states),
        "markov_common_transitions": len(analysis.most_common_transitions),
        "markov_repeating_patterns": len(analysis.repeating_patterns),
    }
    
    # En yaygın geçişin olasılığı
    if analysis.most_common_transitions:
        features["markov_top_transition_prob"] = round(analysis.most_common_transitions[0].probability, 3)
    else:
        features["markov_top_transition_prob"] = 0.0
    
    return features


__all__ = [
    "MarkovModel",
    "TransitionPattern",
    "MarkovAnalysis",
    "build_transition_matrix",
    "calculate_entropy",
    "calculate_predictability",
    "find_common_transitions",
    "find_rare_transitions",
    "find_repeating_patterns",
    "analyze_pitch_transitions",
    "analyze_duration_transitions",
    "analyze_interval_transitions",
    "predict_next_notes",
    "analyze_markov_properties",
    "compare_markov_models",
    "extract_markov_features",
]
