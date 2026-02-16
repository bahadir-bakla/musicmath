"""
Mathematical Pattern Discovery in Audio

CQT ve chroma özelliklerinden matematiksel pattern'ler keşfeder.
MIDI analizi ile cross-validation yapar.

Keşfedilen pattern'ler:
- Key progressions and modulations
- Harmonic entropy and information theory
- Chroma geometry (12-tone space)
- Tonal gravity and attraction

Usage:
    from music_math.analysis.chroma_patterns import ChromaPatternAnalyzer
    analyzer = ChromaPatternAnalyzer()
    patterns = analyzer.analyze(chroma_matrix, sr=22050)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import math

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


@dataclass
class KeySegment:
    """Bir tonal segment (sürekli aynı ton)."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    key: str              # e.g., "C major"
    chroma_profile: np.ndarray  # (12,) average chroma
    strength: float       # Tonal clarity (0-1)


@dataclass
class Modulation:
    """Ton değiştirme (modulation)."""
    time: float
    from_key: str
    to_key: str
    distance: float       # Circle of fifths distance
    type: str            # "diatonic", "chromatic", "distant"
    confidence: float


@dataclass
class HarmonicEntropy:
    """Harmonik entropi metrikleri (Information Theory)."""
    global_entropy: float         # Overall tonal complexity
    local_entropy_mean: float     # Average frame entropy
    local_entropy_std: float      # Variability
    information_rate: float       # Bits per second
    complexity_score: float       # 0-1 arası


@dataclass
class ChromaGeometry:
    """12-tone uzayında geometrik özellikler."""
    centroid: np.ndarray          # (12,) Average position in chroma space
    variance: float               # Spread in chroma space
    principal_axes: np.ndarray    # (2, 12) PCA axes
    eccentricity: float           # How elongated is the distribution
    circularity: float            # How circular (0-1)


@dataclass
class TonalGravity:
    """Tonal attraction patterns (Lerdahl's Tonal Pitch Space)."""
    attraction_matrix: np.ndarray  # (12, 12) Pitch class attractions
    stability_ranking: List[Tuple[int, float]]  # Most stable pitch classes
    tension_curve: np.ndarray      # Tension over time


@dataclass
class MathematicalPatterns:
    """Keşfedilen tüm matematiksel pattern'ler."""
    key_segments: List[KeySegment]
    modulations: List[Modulation]
    harmonic_entropy: HarmonicEntropy
    chroma_geometry: ChromaGeometry
    tonal_gravity: TonalGravity
    fibonacci_patterns: List[Dict]
    prime_patterns: List[Dict]
    golden_ratio_moments: List[float]


class ChromaPatternAnalyzer:
    """Analyze chroma features for mathematical patterns."""
    
    def __init__(self, hop_length: int = 512, sr: int = 22050):
        """
        Initialize analyzer.
        
        Args:
            hop_length: Frame hop length
            sr: Sample rate
        """
        self.hop_length = hop_length
        self.sr = sr
        self.frame_duration = hop_length / sr
        
        # 12-tone note names
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Circle of fifths order
        self.circle_of_fifths = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]  # C, G, D, A, E, B, F#, C#, etc.
    
    def analyze(self, chroma: np.ndarray) -> MathematicalPatterns:
        """
        Comprehensive mathematical analysis of chromagram.
        
        Args:
            chroma: Chroma matrix (12 × n_frames)
            
        Returns:
            MathematicalPatterns
        """
        logger.info("Starting mathematical pattern analysis...")
        
        # Key segments and modulations
        key_segments = self.find_key_segments(chroma)
        modulations = self.detect_modulations(key_segments)
        
        # Information theory
        harmonic_entropy = self.compute_harmonic_entropy(chroma)
        
        # Geometry
        chroma_geometry = self.analyze_chroma_geometry(chroma)
        
        # Tonal gravity
        tonal_gravity = self.compute_tonal_gravity(chroma)
        
        # Mathematical patterns
        fibonacci_patterns = self.find_fibonacci_patterns(chroma)
        prime_patterns = self.find_prime_patterns(chroma)
        golden_ratio_moments = self.find_golden_ratio_moments(chroma)
        
        logger.info(f"Analysis complete: {len(key_segments)} key segments, {len(modulations)} modulations")
        
        return MathematicalPatterns(
            key_segments=key_segments,
            modulations=modulations,
            harmonic_entropy=harmonic_entropy,
            chroma_geometry=chroma_geometry,
            tonal_gravity=tonal_gravity,
            fibonacci_patterns=fibonacci_patterns,
            prime_patterns=prime_patterns,
            golden_ratio_moments=golden_ratio_moments
        )
    
    def find_key_segments(self, chroma: np.ndarray, window_size: int = 50) -> List[KeySegment]:
        """
        Find continuous key segments.
        
        Args:
            chroma: Chroma matrix (12 × n_frames)
            window_size: Analysis window size in frames (~1-2 seconds)
            
        Returns:
            List of KeySegment
        """
        n_frames = chroma.shape[1]
        segments = []
        
        i = 0
        while i < n_frames:
            # Get window
            end = min(i + window_size, n_frames)
            window_chroma = np.mean(chroma[:, i:end], axis=1)
            
            # Estimate key
            key, strength = self._estimate_key_from_chroma(window_chroma)
            
            # Extend segment while key is stable
            segment_start = i
            segment_end = end
            
            j = end
            while j < n_frames:
                next_window = np.mean(chroma[:, j:min(j+10, n_frames)], axis=1)
                next_key, next_strength = self._estimate_key_from_chroma(next_window)
                
                if next_key == key and next_strength > 0.3:
                    segment_end = min(j + 10, n_frames)
                    j += 10
                else:
                    break
            
            # Create segment
            if segment_end - segment_start >= 10:  # Minimum 10 frames
                segment = KeySegment(
                    start_frame=segment_start,
                    end_frame=segment_end,
                    start_time=segment_start * self.frame_duration,
                    end_time=segment_end * self.frame_duration,
                    key=key,
                    chroma_profile=np.mean(chroma[:, segment_start:segment_end], axis=1),
                    strength=strength
                )
                segments.append(segment)
            
            i = segment_end
        
        return segments
    
    def _estimate_key_from_chroma(self, chroma_mean: np.ndarray) -> Tuple[str, float]:
        """Estimate key from average chroma."""
        # Krumhansl-Kessler profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        major_profile /= np.sum(major_profile)
        minor_profile /= np.sum(minor_profile)
        chroma_normalized = chroma_mean / (np.sum(chroma_mean) + 1e-10)
        
        best_strength = -1
        best_key = "C major"
        
        for i in range(12):
            rot_major = np.roll(major_profile, i)
            rot_minor = np.roll(minor_profile, i)
            
            corr_major = np.corrcoef(chroma_normalized, rot_major)[0, 1]
            corr_minor = np.corrcoef(chroma_normalized, rot_minor)[0, 1]
            
            if corr_major > best_strength:
                best_strength = corr_major
                best_key = f"{self.note_names[i]} major"
            
            if corr_minor > best_strength:
                best_strength = corr_minor
                best_key = f"{self.note_names[i]} minor"
        
        return best_key, max(0, best_strength)
    
    def detect_modulations(self, segments: List[KeySegment]) -> List[Modulation]:
        """Detect key changes between segments."""
        modulations = []
        
        for i in range(len(segments) - 1):
            from_key = segments[i].key
            to_key = segments[i + 1].key
            
            if from_key != to_key:
                distance = self._circle_of_fifths_distance(from_key, to_key)
                
                modulation_type = "distant"
                if distance <= 1:
                    modulation_type = "diatonic"
                elif distance <= 3:
                    modulation_type = "chromatic"
                
                mod = Modulation(
                    time=segments[i + 1].start_time,
                    from_key=from_key,
                    to_key=to_key,
                    distance=distance,
                    type=modulation_type,
                    confidence=(segments[i].strength + segments[i + 1].strength) / 2
                )
                modulations.append(mod)
        
        return modulations
    
    def _circle_of_fifths_distance(self, key1: str, key2: str) -> int:
        """Calculate distance on circle of fifths."""
        # Extract root notes
        root1 = key1.split()[0]
        root2 = key2.split()[0]
        
        idx1 = self.note_names.index(root1)
        idx2 = self.note_names.index(root2)
        
        # Position on circle of fifths
        pos1 = self.circle_of_fifths.index(idx1)
        pos2 = self.circle_of_fifths.index(idx2)
        
        # Minimum distance (circular)
        distance = min(abs(pos1 - pos2), 12 - abs(pos1 - pos2))
        
        return distance
    
    def compute_harmonic_entropy(self, chroma: np.ndarray) -> HarmonicEntropy:
        """Compute information-theoretic entropy measures."""
        # Global entropy (average distribution)
        global_mean = np.mean(chroma, axis=1)
        global_mean /= np.sum(global_mean) + 1e-10
        global_entropy = -np.sum(global_mean * np.log2(global_mean + 1e-10))
        
        # Local entropy (frame by frame)
        local_entropies = []
        for i in range(chroma.shape[1]):
            frame = chroma[:, i]
            frame_norm = frame / (np.sum(frame) + 1e-10)
            entropy = -np.sum(frame_norm * np.log2(frame_norm + 1e-10))
            local_entropies.append(entropy)
        
        local_entropy_mean = np.mean(local_entropies)
        local_entropy_std = np.std(local_entropies)
        
        # Information rate (bits per second)
        information_rate = local_entropy_mean / self.frame_duration
        
        # Complexity score (normalized)
        max_entropy = np.log2(12)  # Maximum possible entropy
        complexity_score = local_entropy_mean / max_entropy
        
        return HarmonicEntropy(
            global_entropy=float(global_entropy),
            local_entropy_mean=float(local_entropy_mean),
            local_entropy_std=float(local_entropy_std),
            information_rate=float(information_rate),
            complexity_score=float(complexity_score)
        )
    
    def analyze_chroma_geometry(self, chroma: np.ndarray) -> ChromaGeometry:
        """Analyze geometric properties in 12-tone space."""
        from sklearn.decomposition import PCA
        
        # Centroid
        centroid = np.mean(chroma, axis=1)
        
        # Variance
        variance = np.mean(np.var(chroma, axis=1))
        
        # PCA for principal axes
        pca = PCA(n_components=2)
        pca.fit(chroma.T)
        principal_axes = pca.components_  # (2, 12)
        
        # Eccentricity (how elongated)
        eigenvalues = pca.explained_variance_
        if len(eigenvalues) >= 2 and eigenvalues[0] > 0:
            eccentricity = np.sqrt(1 - (eigenvalues[1] / eigenvalues[0])**2)
        else:
            eccentricity = 0.0
        
        # Circularity (inverse of eccentricity)
        circularity = 1.0 - eccentricity
        
        return ChromaGeometry(
            centroid=centroid,
            variance=float(variance),
            principal_axes=principal_axes,
            eccentricity=float(eccentricity),
            circularity=float(circularity)
        )
    
    def compute_tonal_gravity(self, chroma: np.ndarray) -> TonalGravity:
        """Compute tonal attraction patterns."""
        # Calculate attraction based on proximity in chroma space
        n_frames = chroma.shape[1]
        attraction_matrix = np.zeros((12, 12))
        
        for i in range(n_frames - 1):
            current = chroma[:, i]
            next_frame = chroma[:, i + 1]
            
            # Find strongest pitch classes
            current_pc = np.argmax(current)
            next_pc = np.argmax(next_frame)
            
            attraction_matrix[current_pc, next_pc] += 1
        
        # Normalize
        row_sums = attraction_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        attraction_matrix = attraction_matrix / row_sums
        
        # Stability ranking (most stable pitch classes)
        stability = attraction_matrix.sum(axis=0)
        stability_ranking = sorted(
            [(i, stability[i]) for i in range(12)],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Tension curve (over time)
        tension_curve = np.zeros(n_frames)
        for i in range(n_frames):
            # Tension as deviation from centroid
            tension_curve[i] = np.linalg.norm(chroma[:, i] - np.mean(chroma, axis=1))
        
        return TonalGravity(
            attraction_matrix=attraction_matrix,
            stability_ranking=stability_ranking,
            tension_curve=tension_curve
        )
    
    def find_fibonacci_patterns(self, chroma: np.ndarray) -> List[Dict]:
        """Find Fibonacci-related patterns in chroma."""
        patterns = []
        fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34]
        
        # Look for Fibonacci-length cycles
        for fib_len in fib_numbers:
            if fib_len >= chroma.shape[1]:
                continue
            
            # Check for repeating patterns of Fibonacci length
            for start in range(chroma.shape[1] - fib_len * 2):
                pattern1 = chroma[:, start:start+fib_len]
                pattern2 = chroma[:, start+fib_len:start+2*fib_len]
                
                # Similarity
                sim = 1 - cosine(pattern1.flatten(), pattern2.flatten())
                
                if sim > 0.8:  # High similarity
                    patterns.append({
                        'type': 'fibonacci_repetition',
                        'fibonacci_number': fib_len,
                        'start_frame': start,
                        'similarity': float(sim),
                        'time': start * self.frame_duration
                    })
        
        return patterns[:10]  # Top 10
    
    def find_prime_patterns(self, chroma: np.ndarray) -> List[Dict]:
        """Find prime number related patterns."""
        patterns = []
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        
        # Check if interval lengths are prime
        peak_frames = []
        for i in range(chroma.shape[1]):
            if np.max(chroma[:, i]) > 0.5:  # Significant peak
                peak_frames.append(i)
        
        if len(peak_frames) > 1:
            intervals = np.diff(peak_frames)
            
            for i, interval in enumerate(intervals):
                if interval in primes:
                    patterns.append({
                        'type': 'prime_interval',
                        'prime': int(interval),
                        'frame': int(peak_frames[i]),
                        'time': peak_frames[i] * self.frame_duration
                    })
        
        return patterns[:10]
    
    def find_golden_ratio_moments(self, chroma: np.ndarray) -> List[float]:
        """Find moments related to golden ratio (phi)."""
        phi = (1 + np.sqrt(5)) / 2
        moments = []
        n_frames = chroma.shape[1]
        
        # Check for golden ratio divisions
        total_duration = n_frames * self.frame_duration
        
        # Check if climaxes occur at phi positions
        # Climax = high chroma entropy or centroid change
        chroma_diff = np.diff(np.mean(chroma, axis=0))
        significant_changes = np.where(np.abs(chroma_diff) > np.std(chroma_diff))[0]
        
        for frame in significant_changes:
            time = frame * self.frame_duration
            ratio = time / total_duration if total_duration > 0 else 0
            
            # Check if close to phi-related ratios
            phi_ratios = [1/phi, 1/phi**2, phi-1, 2-phi]
            
            for pr in phi_ratios:
                if abs(ratio - pr) < 0.05:  # Within 5%
                    moments.append(float(time))
                    break
        
        return list(set(moments))[:5]  # Unique, max 5


def analyze_mathematical_patterns(
    chroma: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> MathematicalPatterns:
    """
    Convenience function for mathematical pattern analysis.
    
    Args:
        chroma: Chroma matrix (12 × n_frames)
        sr: Sample rate
        hop_length: Frame hop
        
    Returns:
        MathematicalPatterns
    """
    analyzer = ChromaPatternAnalyzer(hop_length=hop_length, sr=sr)
    return analyzer.analyze(chroma)


__all__ = [
    'KeySegment',
    'Modulation',
    'HarmonicEntropy',
    'ChromaGeometry',
    'TonalGravity',
    'MathematicalPatterns',
    'ChromaPatternAnalyzer',
    'analyze_mathematical_patterns',
]
