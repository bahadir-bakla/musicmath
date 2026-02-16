"""
Audio Feature Extraction Module

CQT ve chroma özellikleri ile müzik matematiği analizi.
MIDI ile cross-validation yapılabilir özellikler.

Usage:
    from music_math.audio.features import AudioFeatureExtractor
    extractor = AudioFeatureExtractor()
    features = extractor.extract_all(audio, sr=22050)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import json

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChromaFeatures:
    """12-tone chroma representation features."""
    mean: np.ndarray           # (12,) - Average pitch class energy
    std: np.ndarray            # (12,) - Variability
    entropy: float             # Tonal complexity
    key_estimate: str          # Estimated key (e.g., "C major")
    modality: str              # "major" or "minor"
    tonality_strength: float   # How clear is the key (0-1)


@dataclass
class SpectralFeatures:
    """Spectral (frequency-domain) features."""
    centroid_mean: float       # Brightness
    centroid_std: float
    rolloff_mean: float        # Frequency balance
    rolloff_std: float
    flux_mean: float           # Timbre change rate
    flux_std: float
    contrast_mean: np.ndarray  # (6,) Octave-based contrast


@dataclass
class RhythmicFeatures:
    """Rhythm and tempo features."""
    tempo: float               # BPM
    tempo_confidence: float    # Tempo detection confidence
    beat_frames: np.ndarray    # Beat positions
    onset_strength_mean: float # Rhythmic activity
    onset_strength_std: float
    rhythm_regularity: float   # How regular are the beats (0-1)


@dataclass
class AudioFeatures:
    """Complete audio feature set."""
    chroma: ChromaFeatures
    spectral: SpectralFeatures
    rhythmic: RhythmicFeatures
    duration: float            # Audio duration in seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for JSON serialization)."""
        return {
            'chroma': {
                'mean': self.chroma.mean.tolist(),
                'std': self.chroma.std.tolist(),
                'entropy': float(self.chroma.entropy),
                'key_estimate': self.chroma.key_estimate,
                'modality': self.chroma.modality,
                'tonality_strength': float(self.chroma.tonality_strength),
            },
            'spectral': {
                'centroid_mean': float(self.spectral.centroid_mean),
                'centroid_std': float(self.spectral.centroid_std),
                'rolloff_mean': float(self.spectral.rolloff_mean),
                'rolloff_std': float(self.spectral.rolloff_std),
                'flux_mean': float(self.spectral.flux_mean),
                'flux_std': float(self.spectral.flux_std),
                'contrast_mean': self.spectral.contrast_mean.tolist(),
            },
            'rhythmic': {
                'tempo': float(self.rhythmic.tempo),
                'tempo_confidence': float(self.rhythmic.tempo_confidence),
                'beat_frames': self.rhythmic.beat_frames.tolist(),
                'onset_strength_mean': float(self.rhythmic.onset_strength_mean),
                'onset_strength_std': float(self.rhythmic.onset_strength_std),
                'rhythm_regularity': float(self.rhythmic.rhythm_regularity),
            },
            'duration': float(self.duration),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AudioFeatures':
        """Create from dictionary."""
        return cls(
            chroma=ChromaFeatures(
                mean=np.array(data['chroma']['mean']),
                std=np.array(data['chroma']['std']),
                entropy=data['chroma']['entropy'],
                key_estimate=data['chroma']['key_estimate'],
                modality=data['chroma']['modality'],
                tonality_strength=data['chroma']['tonality_strength'],
            ),
            spectral=SpectralFeatures(
                centroid_mean=data['spectral']['centroid_mean'],
                centroid_std=data['spectral']['centroid_std'],
                rolloff_mean=data['spectral']['rolloff_mean'],
                rolloff_std=data['spectral']['rolloff_std'],
                flux_mean=data['spectral']['flux_mean'],
                flux_std=data['spectral']['flux_std'],
                contrast_mean=np.array(data['spectral']['contrast_mean']),
            ),
            rhythmic=RhythmicFeatures(
                tempo=data['rhythmic']['tempo'],
                tempo_confidence=data['rhythmic']['tempo_confidence'],
                beat_frames=np.array(data['rhythmic']['beat_frames']),
                onset_strength_mean=data['rhythmic']['onset_strength_mean'],
                onset_strength_std=data['rhythmic']['onset_strength_std'],
                rhythm_regularity=data['rhythmic']['rhythm_regularity'],
            ),
            duration=data['duration'],
        )


class AudioFeatureExtractor:
    """Extract comprehensive audio features for mathematical analysis."""
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize extractor.
        
        Args:
            use_gpu: Use GPU acceleration if available (RTX 3070)
        """
        self.use_gpu = use_gpu
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check required libraries."""
        try:
            import librosa
            self.has_librosa = True
        except ImportError:
            logger.error("librosa not installed: pip install librosa")
            self.has_librosa = False
        
        try:
            import torch
            self.has_torch = torch.cuda.is_available()
            if self.has_torch:
                logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        except ImportError:
            self.has_torch = False
    
    def extract_chroma_features(
        self,
        audio: np.ndarray,
        sr: int = 22050
    ) -> ChromaFeatures:
        """
        Extract 12-tone chroma features.
        
        Args:
            audio: Audio time series
            sr: Sample rate
            
        Returns:
            ChromaFeatures
        """
        import librosa
        
        # Compute chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
        
        # Basic statistics
        chroma_mean = np.mean(chroma, axis=1)  # (12,)
        chroma_std = np.std(chroma, axis=1)    # (12,)
        
        # Entropy (tonal complexity)
        # Higher entropy = more complex/chromatic harmony
        chroma_normalized = chroma_mean / (np.sum(chroma_mean) + 1e-10)
        entropy = -np.sum(chroma_normalized * np.log2(chroma_normalized + 1e-10))
        
        # Key estimation using Krumhansl-Schmuckler algorithm
        key, strength = self._estimate_key(chroma_mean)
        modality = "major" if key.endswith("major") else "minor"
        
        return ChromaFeatures(
            mean=chroma_mean,
            std=chroma_std,
            entropy=float(entropy),
            key_estimate=key,
            modality=modality,
            tonality_strength=float(strength)
        )
    
    def _estimate_key(self, chroma_mean: np.ndarray) -> Tuple[str, float]:
        """
        Estimate musical key from chroma profile.
        
        Uses Krumhansl-Schmuckler key-finding algorithm.
        
        Returns:
            (key_name, strength) tuple
        """
        import librosa
        
        # Use librosa's key detection
        # This uses Krumhansl-Kessler probe tone profiles
        key = librosa.feature.key.chroma_key(chroma_mean)
        
        # Calculate strength (correlation with best matching profile)
        # Higher = more confident
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalize
        major_profile /= np.sum(major_profile)
        minor_profile /= np.sum(minor_profile)
        chroma_normalized = chroma_mean / (np.sum(chroma_mean) + 1e-10)
        
        # Find best match
        best_strength = 0
        best_key = "C major"
        
        for i in range(12):
            # Rotate profiles
            rot_major = np.roll(major_profile, i)
            rot_minor = np.roll(minor_profile, i)
            
            # Correlation
            corr_major = np.corrcoef(chroma_normalized, rot_major)[0, 1]
            corr_minor = np.corrcoef(chroma_normalized, rot_minor)[0, 1]
            
            if corr_major > best_strength:
                best_strength = corr_major
                note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                best_key = f"{note_names[i]} major"
            
            if corr_minor > best_strength:
                best_strength = corr_minor
                note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                best_key = f"{note_names[i]} minor"
        
        return best_key, best_strength
    
    def extract_spectral_features(
        self,
        audio: np.ndarray,
        sr: int = 22050
    ) -> SpectralFeatures:
        """Extract spectral features."""
        import librosa
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        
        # Spectral rolloff (frequency balance)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        
        # Spectral flux (timbre changes)
        S = np.abs(librosa.stft(audio))
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
        
        # Spectral contrast (octave-based)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        
        return SpectralFeatures(
            centroid_mean=float(np.mean(centroid)),
            centroid_std=float(np.std(centroid)),
            rolloff_mean=float(np.mean(rolloff)),
            rolloff_std=float(np.std(rolloff)),
            flux_mean=float(np.mean(flux)),
            flux_std=float(np.std(flux)),
            contrast_mean=np.mean(contrast, axis=1),  # (6,)
        )
    
    def extract_rhythmic_features(
        self,
        audio: np.ndarray,
        sr: int = 22050
    ) -> RhythmicFeatures:
        """Extract rhythm and tempo features."""
        import librosa
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        
        # Tempo and beats
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Rhythm regularity (beat interval consistency)
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            regularity = 1.0 - (np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-10))
            regularity = max(0, min(1, regularity))  # Clamp to [0, 1]
        else:
            regularity = 0.0
        
        return RhythmicFeatures(
            tempo=float(tempo),
            tempo_confidence=regularity,  # Use regularity as proxy
            beat_frames=beats,
            onset_strength_mean=float(np.mean(onset_env)),
            onset_strength_std=float(np.std(onset_env)),
            rhythm_regularity=float(regularity),
        )
    
    def extract_all(
        self,
        audio: np.ndarray,
        sr: int = 22050
    ) -> AudioFeatures:
        """
        Extract all audio features.
        
        Args:
            audio: Audio time series
            sr: Sample rate
            
        Returns:
            AudioFeatures
        """
        logger.info("Extracting audio features...")
        
        duration = len(audio) / sr
        
        chroma = self.extract_chroma_features(audio, sr)
        spectral = self.extract_spectral_features(audio, sr)
        rhythmic = self.extract_rhythmic_features(audio, sr)
        
        logger.info(f"Features extracted: key={chroma.key_estimate}, tempo={rhythmic.tempo:.1f}bpm")
        
        return AudioFeatures(
            chroma=chroma,
            spectral=spectral,
            rhythmic=rhythmic,
            duration=duration
        )
    
    def extract_from_midi(
        self,
        midi_path: Path,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None
    ) -> AudioFeatures:
        """
        Extract features directly from MIDI (renders first).
        
        Args:
            midi_path: Path to MIDI file
            use_cache: Use feature cache
            cache_dir: Cache directory
            
        Returns:
            AudioFeatures
        """
        from .synthesis import midi_to_audio, get_audio_hash
        
        midi_path = Path(midi_path)
        
        # Check cache
        if use_cache and cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            audio_hash = get_audio_hash(midi_path)
            cache_file = cache_dir / f"{audio_hash}_features.json"
            
            if cache_file.exists():
                logger.info(f"Loading cached features: {cache_file.name}")
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return AudioFeatures.from_dict(data)
        
        # Render and extract
        logger.info(f"Processing MIDI: {midi_path.name}")
        audio, sr = midi_to_audio(midi_path)
        features = self.extract_all(audio, sr)
        
        # Cache
        if use_cache and cache_dir:
            with open(cache_file, 'w') as f:
                json.dump(features.to_dict(), f, indent=2)
            logger.info(f"Cached features: {cache_file.name}")
        
        return features


def extract_features_for_dataset(
    midi_files: List[Path],
    output_csv: Path,
    use_gpu: bool = True
):
    """
    Batch extract features for multiple MIDI files.
    
    Args:
        midi_files: List of MIDI file paths
        output_csv: Output CSV path
        use_gpu: Use GPU acceleration
    """
    import pandas as pd
    
    extractor = AudioFeatureExtractor(use_gpu=use_gpu)
    records = []
    
    for i, midi_file in enumerate(midi_files):
        logger.info(f"Processing {i+1}/{len(midi_files)}: {midi_file.name}")
        
        try:
            features = extractor.extract_from_midi(midi_file)
            
            record = {
                'filepath': str(midi_file),
                'duration': features.duration,
                'key': features.chroma.key_estimate,
                'modality': features.chroma.modality,
                'tonality_strength': features.chroma.tonality_strength,
                'chroma_entropy': features.chroma.entropy,
                'tempo': features.rhythmic.tempo,
                'tempo_confidence': features.rhythmic.tempo_confidence,
                'rhythm_regularity': features.rhythmic.rhythm_regularity,
                'spectral_centroid': features.spectral.centroid_mean,
                'spectral_flux': features.spectral.flux_mean,
            }
            
            # Add chroma vector
            for j, val in enumerate(features.chroma.mean):
                record[f'chroma_{j}'] = val
            
            records.append(record)
            
        except Exception as e:
            logger.error(f"Error processing {midi_file}: {e}")
    
    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved features to: {output_csv}")


__all__ = [
    'AudioFeatureExtractor',
    'AudioFeatures',
    'ChromaFeatures',
    'SpectralFeatures',
    'RhythmicFeatures',
    'extract_features_for_dataset',
]
