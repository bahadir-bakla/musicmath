"""
Müzik benzerlik algoritmaları modülü.

Desteklenen algoritmalar:
- DTW (Dynamic Time Warping): Zaman serisi benzerliği, farklı hızdaki melodileri karşılaştırma
- LCS (Longest Common Subsequence): Ortak alt dizi bulma
- Cosine Similarity: Feature vektör uzayında açısal benzerlik
- Pearson Correlation: İstatistiksel korelasyon
- LSH (Locality Sensitive Hashing): Büyük ölçekli yaklaşık benzerlik arama

Kullanım:
    from music_math.analysis.similarity import (
        dtw_similarity,
        lcs_similarity,
        cosine_vector_sim,
        pearson_correlation,
        lsh_similarity_search,
        find_similar_tracks
    )
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from collections import defaultdict
import hashlib
from dataclasses import dataclass

from music_math.core.types import NoteEvent
from music_math.data.loader import parse_midi_to_note_events


@dataclass
class SimilarityResult:
    """Benzerlik analizi sonuçları için veri sınıfı."""
    track_id: str
    track_name: str
    composer: str
    era: str
    similarity_score: float
    algorithm: str
    details: Dict


# =============================================================================
# 1. DYNAMIC TIME WARPING (DTW)
# =============================================================================

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Dynamic Time Warping mesafesi hesaplar.
    
    DTW, zaman serilerini hizalayarak farklı hızlardaki benzer pattern'leri bulur.
    Özellikle melodiler arası benzerlik için idealdir.
    
    Args:
        seq1: Birinci zaman serisi (pitch veya interval)
        seq2: İkinci zaman serisi
        
    Returns:
        (mesafe, warp_yolu) tuple'ı
        
    Example:
        >>> pitch1 = np.array([60, 62, 64, 65, 67])
        >>> pitch2 = np.array([60, 60, 62, 64, 65, 67])
        >>> dist, path = dtw_distance(pitch1, pitch2)
        >>> print(f"DTW mesafe: {dist:.2f}")
    """
    n, m = len(seq1), len(seq2)
    
    # Maliyet matrisi
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0
    
    # Dinamik programlama
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = abs(seq1[i-1] - seq2[j-1])
            cost[i, j] = dist + min(
                cost[i-1, j],     # insertion
                cost[i, j-1],     # deletion
                cost[i-1, j-1]    # match
            )
    
    # Warp yolunu geri izle
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i-1, j-1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            idx = np.argmin([cost[i-1, j-1], cost[i-1, j], cost[i, j-1]])
            if idx == 0:
                i -= 1
                j -= 1
            elif idx == 1:
                i -= 1
            else:
                j -= 1
    
    path.reverse()
    return cost[n, m], np.array(path)


def dtw_similarity(seq1: np.ndarray, seq2: np.ndarray, normalize: bool = True) -> float:
    """
    DTW benzerlik skoru (0-1 arası, 1=en benzer).
    
    Args:
        seq1: Birinci zaman serisi
        seq2: İkinci zaman serisi
        normalize: Normalize edilmiş skor döndür
        
    Returns:
        Benzerlik skoru [0, 1]
    """
    dist, _ = dtw_distance(seq1, seq2)
    
    if normalize:
        # Normalize: max mesafe ~ len(seq1) + len(seq2)
        max_dist = len(seq1) + len(seq2)
        similarity = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
        return max(0.0, min(1.0, similarity))
    
    return dist


def dtw_events_similarity(events1: List[NoteEvent], events2: List[NoteEvent], 
                          use_intervals: bool = True) -> float:
    """
    İki nota olay listesi arasında DTW benzerliği.
    
    Args:
        events1: Birinci parçanın nota olayları
        events2: İkinci parçanın nota olayları
        use_intervals: True ise interval, False ise pitch kullan
    """
    if use_intervals:
        pitches1 = np.array([e.pitch for e in events1])
        pitches2 = np.array([e.pitch for e in events2])
        if len(pitches1) < 2 or len(pitches2) < 2:
            return 0.0
        seq1 = np.diff(pitches1)
        seq2 = np.diff(pitches2)
    else:
        seq1 = np.array([e.pitch for e in events1])
        seq2 = np.array([e.pitch for e in events2])
    
    return dtw_similarity(seq1, seq2)


# =============================================================================
# 2. LONGEST COMMON SUBSEQUENCE (LCS)
# =============================================================================

def lcs_length(seq1: np.ndarray, seq2: np.ndarray, tolerance: float = 0.0) -> int:
    """
    En uzun ortak alt dizi uzunluğu.
    
    Args:
        seq1: Birinci dizi
        seq2: İkinci dizi
        tolerance: Sayısal tolerans (örn: 1.0 ise ±1 pitch farkı tolere edilir)
        
    Returns:
        LCS uzunluğu
    """
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if tolerance == 0:
                match = seq1[i-1] == seq2[j-1]
            else:
                match = abs(seq1[i-1] - seq2[j-1]) <= tolerance
            
            if match:
                dp[i, j] = dp[i-1, j-1] + 1
            else:
                dp[i, j] = max(dp[i-1, j], dp[i, j-1])
    
    return int(dp[n, m])


def lcs_similarity(seq1: np.ndarray, seq2: np.ndarray, tolerance: float = 0.0) -> float:
    """
    LCS benzerlik skoru [0, 1].
    
    Args:
        seq1: Birinci dizi
        seq2: İkinci dizi
        tolerance: Sayısal tolerans
        
    Returns:
        Benzerlik skoru
    """
    lcs_len = lcs_length(seq1, seq2, tolerance)
    max_len = max(len(seq1), len(seq2))
    return lcs_len / max_len if max_len > 0 else 0.0


def lcs_backtrack(seq1: np.ndarray, seq2: np.ndarray, tolerance: float = 0.0) -> List:
    """
    LCS dizisini geri izleyerek bulur.
    
    Returns:
        Ortak alt dizi elemanları listesi
    """
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    # DP tablosu oluştur
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if tolerance == 0:
                match = seq1[i-1] == seq2[j-1]
            else:
                match = abs(seq1[i-1] - seq2[j-1]) <= tolerance
            
            if match:
                dp[i, j] = dp[i-1, j-1] + 1
            else:
                dp[i, j] = max(dp[i-1, j], dp[i, j-1])
    
    # Geri izleme
    lcs = []
    i, j = n, m
    while i > 0 and j > 0:
        if tolerance == 0:
            match = seq1[i-1] == seq2[j-1]
        else:
            match = abs(seq1[i-1] - seq2[j-1]) <= tolerance
        
        if match:
            lcs.append(seq1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1, j] > dp[i, j-1]:
            i -= 1
        else:
            j -= 1
    
    lcs.reverse()
    return lcs


# =============================================================================
# 3. COSINE SIMILARITY (Feature Vectors)
# =============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    İki vektör arası kosinüs benzerliği.
    
    Args:
        vec1: Birinci vektör
        vec2: İkinci vektör
        
    Returns:
        Cosine benzerliği [-1, 1], normalde [0, 1]
    """
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (norm1 * norm2)


def cosine_vector_sim(features1: Dict[str, float], features2: Dict[str, float],
                      numeric_only: bool = True) -> float:
    """
    Feature sözlükleri arası kosinüs benzerliği.
    
    Args:
        features1: Birinci parçanın feature'ları
        features2: İkinci parçanın feature'ları
        numeric_only: Sadece sayısal değerleri kullan
        
    Returns:
        Benzerlik skoru [0, 1]
    """
    # Ortak key'leri bul
    common_keys = set(features1.keys()) & set(features2.keys())
    
    if numeric_only:
        common_keys = {k for k in common_keys 
                      if isinstance(features1[k], (int, float)) 
                      and isinstance(features2[k], (int, float))}
    
    if not common_keys:
        return 0.0
    
    vec1 = np.array([float(features1[k]) for k in common_keys])
    vec2 = np.array([float(features2[k]) for k in common_keys])
    
    return cosine_similarity(vec1, vec2)


# =============================================================================
# 4. PEARSON CORRELATION
# =============================================================================

def pearson_correlation(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Pearson korelasyon katsayısı.
    
    Args:
        seq1: Birinci zaman serisi
        seq2: İkinci zaman serisi
        
    Returns:
        Korelasyon [-1, 1]
    """
    if len(seq1) != len(seq2):
        # Farklı uzunluklar için interpolate veya truncate
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    if len(seq1) < 2:
        return 0.0
    
    mean1, mean2 = np.mean(seq1), np.mean(seq2)
    std1, std2 = np.std(seq1), np.std(seq2)
    
    if std1 == 0 or std2 == 0:
        return 0.0
    
    covariance = np.mean((seq1 - mean1) * (seq2 - mean2))
    return covariance / (std1 * std2)


def pearson_similarity(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Pearson korelasyonundan benzerlik skoru [0, 1].
    
    Args:
        seq1: Birinci zaman serisi
        seq2: İkinci zaman serisi
        
    Returns:
        Benzerlik skoru
    """
    corr = pearson_correlation(seq1, seq2)
    # [-1, 1] -> [0, 1]
    return (corr + 1) / 2


# =============================================================================
# 5. LOCALITY SENSITIVE HASHING (LSH)
# =============================================================================

class LSHIndex:
    """
    LSH indeksi için sınıf. Büyük datasetlerde yaklaşık benzerlik arama.
    """
    
    def __init__(self, num_bands: int = 20, rows_per_band: int = 5):
        """
        Args:
            num_bands: Bant sayısı
            rows_per_band: Her banttaki satır sayısı
        """
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.bands: List[Dict] = [defaultdict(list) for _ in range(num_bands)]
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
    
    def _minhash_signature(self, vector: np.ndarray, num_hash: int) -> List[int]:
        """
        MinHash imzası oluşturur.
        """
        np.random.seed(42)
        signature = []
        
        for i in range(num_hash):
            # Rastgele permütasyon fonksiyonu
            a = np.random.randint(1, 10000)
            b = np.random.randint(0, 10000)
            p = 999999937  # Büyük asal sayı
            
            min_hash = float('inf')
            for idx, val in enumerate(vector):
                if val != 0:
                    hash_val = ((a * idx + b) % p) % len(vector)
                    min_hash = min(min_hash, hash_val)
            
            signature.append(int(min_hash) if min_hash != float('inf') else 0)
        
        return signature
    
    def add_vector(self, track_id: str, vector: np.ndarray, metadata: Dict = None):
        """
        Vektörü LSH indeksine ekler.
        
        Args:
            track_id: Parça ID'si
            vector: Feature vektörü
            metadata: Ek bilgiler (besteci, dönem vb.)
        """
        self.vectors[track_id] = vector
        self.metadata[track_id] = metadata or {}
        
        # MinHash imzası
        num_hashes = self.num_bands * self.rows_per_band
        signature = self._minhash_signature(vector, num_hashes)
        
        # Bantlara böl
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_signature = tuple(signature[start:end])
            
            # Hash tablosuna ekle
            bucket_key = hashlib.md5(str(band_signature).encode()).hexdigest()
            self.bands[band_idx][bucket_key].append(track_id)
    
    def query(self, vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        En yakın komşuları bulur.
        
        Args:
            vector: Sorgu vektörü
            top_k: Döndürülecek sonuç sayısı
            
        Returns:
            (track_id, similarity) listesi
        """
        # Adayları topla
        candidates = set()
        num_hashes = self.num_bands * self.rows_per_band
        signature = self._minhash_signature(vector, num_hashes)
        
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_signature = tuple(signature[start:end])
            bucket_key = hashlib.md5(str(band_signature).encode()).hexdigest()
            candidates.update(self.bands[band_idx].get(bucket_key, []))
        
        # Gerçek benzerlikleri hesapla
        similarities = []
        for track_id in candidates:
            if track_id in self.vectors:
                sim = cosine_similarity(vector, self.vectors[track_id])
                similarities.append((track_id, sim))
        
        # Sırala ve top_k döndür
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def lsh_similarity_search(query_vector: np.ndarray, 
                         track_vectors: Dict[str, np.ndarray],
                         top_k: int = 10) -> List[Tuple[str, float]]:
    """
    LSH kullanarak hızlı benzerlik araması.
    
    Args:
        query_vector: Sorgu vektörü
        track_vectors: Tüm parçaların vektörleri {track_id: vector}
        top_k: Sonuç sayısı
        
    Returns:
        (track_id, similarity) listesi
    """
    lsh = LSHIndex()
    
    # Tüm vektörleri indeksle
    for track_id, vector in track_vectors.items():
        lsh.add_vector(track_id, vector)
    
    # Sorgu
    return lsh.query(query_vector, top_k)


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def normalize_sequence(seq: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Zaman serisini normalize eder.
    
    Args:
        seq: Girdi serisi
        method: "minmax", "zscore", "max"
        
    Returns:
        Normalize edilmiş seri
    """
    if method == "minmax":
        min_val, max_val = np.min(seq), np.max(seq)
        if max_val - min_val == 0:
            return np.zeros_like(seq)
        return (seq - min_val) / (max_val - min_val)
    
    elif method == "zscore":
        mean, std = np.mean(seq), np.std(seq)
        if std == 0:
            return np.zeros_like(seq)
        return (seq - mean) / std
    
    elif method == "max":
        max_val = np.max(np.abs(seq))
        if max_val == 0:
            return seq
        return seq / max_val
    
    return seq


def events_to_sequence(events: List[NoteEvent], 
                       feature: str = "pitch") -> np.ndarray:
    """
    Nota olaylarını sayısal diziye dönüştürür.
    
    Args:
        events: Nota olayları
        feature: "pitch", "interval", "duration", "velocity"
        
    Returns:
        Numpy dizisi
    """
    if feature == "pitch":
        return np.array([e.pitch for e in events])
    elif feature == "duration":
        return np.array([e.duration for e in events])
    elif feature == "interval":
        pitches = [e.pitch for e in events]
        if len(pitches) < 2:
            return np.array([])
        return np.diff(pitches)
    elif feature == "velocity":
        return np.array([getattr(e, 'velocity', 64) for e in events])
    else:
        return np.array([e.pitch for e in events])


# =============================================================================
# ANA FONKSİYON: Benzer Parça Bulma
# =============================================================================

def find_similar_tracks(
    query_events: List[NoteEvent],
    candidate_events: Dict[str, List[NoteEvent]],
    algorithm: str = "dtw",
    top_k: int = 10,
    **kwargs
) -> List[SimilarityResult]:
    """
    Bir parçaya en benzer parçaları bulur.
    
    Args:
        query_events: Sorgu parçasının nota olayları
        candidate_events: Aday parçalar {track_id: events}
        algorithm: "dtw", "lcs", "cosine", "pearson"
        top_k: Sonuç sayısı
        **kwargs: Algoritma özel parametreler
        
    Returns:
        SimilarityResult listesi
        
    Example:
        >>> query = parse_midi_to_note_events("query.mid")
        >>> candidates = {
        ...     "bach_1": parse_midi_to_note_events("bach1.mid"),
        ...     "mozart_1": parse_midi_to_note_events("mozart1.mid"),
        ... }
        >>> results = find_similar_tracks(query, candidates, algorithm="dtw", top_k=5)
        >>> for r in results:
        ...     print(f"{r.track_id}: {r.similarity_score:.3f}")
    """
    results = []
    
    # Sorgu dizisi
    if algorithm in ["dtw", "lcs", "pearson"]:
        query_seq = events_to_sequence(query_events, kwargs.get("feature", "interval"))
    else:
        # Cosine için feature vektörü lazım
        query_seq = events_to_sequence(query_events, "pitch")
    
    for track_id, events in candidate_events.items():
        if len(events) < 10:  # Çok kısa parçaları atla
            continue
        
        if algorithm in ["dtw", "lcs", "pearson"]:
            candidate_seq = events_to_sequence(events, kwargs.get("feature", "interval"))
        else:
            candidate_seq = events_to_sequence(events, "pitch")
        
        # Benzerlik hesapla
        if algorithm == "dtw":
            score = dtw_similarity(query_seq, candidate_seq)
        elif algorithm == "lcs":
            tolerance = kwargs.get("tolerance", 0.0)
            score = lcs_similarity(query_seq, candidate_seq, tolerance)
        elif algorithm == "cosine":
            # Basit: pitch histogram kullan
            query_hist = np.histogram(query_seq, bins=12, range=(0, 127))[0].astype(float)
            cand_hist = np.histogram(candidate_seq, bins=12, range=(0, 127))[0].astype(float)
            score = cosine_similarity(query_hist, cand_hist)
        elif algorithm == "pearson":
            score = pearson_similarity(query_seq, candidate_seq)
        else:
            score = dtw_similarity(query_seq, candidate_seq)
        
        results.append(SimilarityResult(
            track_id=track_id,
            track_name=kwargs.get("metadata", {}).get(track_id, {}).get("name", track_id),
            composer=kwargs.get("metadata", {}).get(track_id, {}).get("composer", "Unknown"),
            era=kwargs.get("metadata", {}).get(track_id, {}).get("era", "Unknown"),
            similarity_score=score,
            algorithm=algorithm,
            details={}
        ))
    
    # Sırala
    results.sort(key=lambda x: x.similarity_score, reverse=True)
    return results[:top_k]


def compare_two_pieces(
    filepath1: str,
    filepath2: str,
    algorithm: str = "dtw"
) -> Dict:
    """
    İki MIDI dosyasını karşılaştırır.
    
    Args:
        filepath1: Birinci dosya yolu
        filepath2: İkinci dosya yolu
        algorithm: Kullanılacak algoritma
        
    Returns:
        Karşılaştırma sonuçları sözlüğü
    """
    events1 = parse_midi_to_note_events(filepath1)
    events2 = parse_midi_to_note_events(filepath2)
    
    if algorithm == "dtw":
        seq1 = events_to_sequence(events1, "interval")
        seq2 = events_to_sequence(events2, "interval")
        distance, path = dtw_distance(seq1, seq2)
        similarity = dtw_similarity(seq1, seq2)
        return {
            "algorithm": "dtw",
            "distance": float(distance),
            "similarity": float(similarity),
            "warp_path": path.tolist(),
            "seq1_length": len(seq1),
            "seq2_length": len(seq2),
        }
    
    elif algorithm == "lcs":
        seq1 = events_to_sequence(events1, "pitch")
        seq2 = events_to_sequence(events2, "pitch")
        lcs_len = lcs_length(seq1, seq2, tolerance=1)
        lcs_seq = lcs_backtrack(seq1, seq2, tolerance=1)
        similarity = lcs_similarity(seq1, seq2, tolerance=1)
        return {
            "algorithm": "lcs",
            "lcs_length": lcs_len,
            "similarity": float(similarity),
            "lcs_sequence": [int(x) for x in lcs_seq],
            "seq1_length": len(seq1),
            "seq2_length": len(seq2),
        }
    
    elif algorithm == "pearson":
        seq1 = events_to_sequence(events1, "pitch")
        seq2 = events_to_sequence(events2, "pitch")
        corr = pearson_correlation(seq1, seq2)
        similarity = pearson_similarity(seq1, seq2)
        return {
            "algorithm": "pearson",
            "correlation": float(corr),
            "similarity": float(similarity),
        }
    
    else:
        raise ValueError(f"Bilinmeyen algoritma: {algorithm}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # DTW
    "dtw_distance",
    "dtw_similarity",
    "dtw_events_similarity",
    
    # LCS
    "lcs_length",
    "lcs_similarity",
    "lcs_backtrack",
    
    # Cosine
    "cosine_similarity",
    "cosine_vector_sim",
    
    # Pearson
    "pearson_correlation",
    "pearson_similarity",
    
    # LSH
    "LSHIndex",
    "lsh_similarity_search",
    
    # Yardımcılar
    "normalize_sequence",
    "events_to_sequence",
    
    # Ana fonksiyonlar
    "SimilarityResult",
    "find_similar_tracks",
    "compare_two_pieces",
]
