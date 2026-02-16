"""
MusicMath Audio Analysis Module

MIDI dosyalarını audio'ya render eder ve matematiksel analiz için
feature extraction yapar. ASUS ROG Strix RTX 3070 için optimize edilmiştir.

Modules:
    synthesis: MIDI-to-Audio rendering (FluidSynth)
    cqt: Constant-Q Transform with GPU acceleration
    features: Audio feature extraction (chroma, spectral, rhythmic)

Usage:
    from music_math.audio import midi_to_audio, AudioFeatureExtractor
    
    # MIDI'den audio render et
    audio, sr = midi_to_audio("piece.mid")
    
    # Feature extraction
    extractor = AudioFeatureExtractor(use_gpu=True)
    features = extractor.extract_all(audio, sr)
    
    print(f"Key: {features.chroma.key_estimate}")
    print(f"Tempo: {features.rhythmic.tempo} BPM")
"""

from .synthesis import (
    FluidSynthWrapper,
    get_synth,
    midi_to_audio,
    render_midi_bytes,
    get_audio_hash,
    DEFAULT_SOUNDFONT,
    SAMPLE_RATE,
)

from .cqt import (
    compute_cqt,
    compute_cqt_gpu,
    compute_cqt_for_midi,
    cqt_to_chroma,
    get_cqt_params,
    get_cqt_frequencies,
    DEFAULT_CQT_PARAMS,
    GPU_CONFIG,
)

from .features import (
    AudioFeatureExtractor,
    AudioFeatures,
    ChromaFeatures,
    SpectralFeatures,
    RhythmicFeatures,
    extract_features_for_dataset,
)

__version__ = "1.0.0"

__all__ = [
    # Synthesis
    'FluidSynthWrapper',
    'get_synth',
    'midi_to_audio',
    'render_midi_bytes',
    'get_audio_hash',
    'DEFAULT_SOUNDFONT',
    'SAMPLE_RATE',
    
    # CQT
    'compute_cqt',
    'compute_cqt_gpu',
    'compute_cqt_for_midi',
    'cqt_to_chroma',
    'get_cqt_params',
    'get_cqt_frequencies',
    'DEFAULT_CQT_PARAMS',
    'GPU_CONFIG',
    
    # Features
    'AudioFeatureExtractor',
    'AudioFeatures',
    'ChromaFeatures',
    'SpectralFeatures',
    'RhythmicFeatures',
    'extract_features_for_dataset',
]
