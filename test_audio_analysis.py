"""
Audio Analysis Test/Demo Script

Yeni audio analiz modüllerini test eder.
Usage:
    python test_audio_analysis.py <midi_file>
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_midi_to_audio(midi_path: str):
    """Test MIDI to audio rendering."""
    logger.info("=" * 60)
    logger.info("TEST 1: MIDI to Audio Rendering")
    logger.info("=" * 60)
    
    try:
        from music_math.audio import midi_to_audio, SAMPLE_RATE
        
        logger.info(f"Rendering: {midi_path}")
        start_time = time.time()
        
        audio, sr = midi_to_audio(midi_path, duration_limit=30)  # First 30 seconds
        
        render_time = time.time() - start_time
        audio_duration = len(audio) / sr
        
        logger.info(f"✓ Rendered {audio_duration:.2f}s of audio")
        logger.info(f"✓ Sample rate: {sr} Hz")
        logger.info(f"✓ Render time: {render_time:.2f}s")
        logger.info(f"✓ Audio shape: {audio.shape}")
        logger.info(f"✓ Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
        
        return audio, sr
        
    except Exception as e:
        logger.error(f"✗ MIDI to audio failed: {e}")
        return None, None


def test_cqt(audio: np.ndarray, sr: int):
    """Test CQT computation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: CQT (Constant-Q Transform)")
    logger.info("=" * 60)
    
    try:
        from music_math.audio import compute_cqt, compute_cqt_gpu, cqt_to_chroma
        
        # Test CPU version
        logger.info("Computing CQT (CPU)...")
        start_time = time.time()
        C_cpu = compute_cqt(audio, sr=sr, n_bins=84)
        cpu_time = time.time() - start_time
        
        logger.info(f"✓ CPU CQT shape: {C_cpu.shape}")
        logger.info(f"✓ CPU CQT time: {cpu_time:.2f}s")
        
        # Test GPU version if available
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Computing CQT (GPU - RTX 3070)...")
                start_time = time.time()
                C_gpu = compute_cqt_gpu(audio, sr=sr, n_bins=84)
                gpu_time = time.time() - start_time
                
                logger.info(f"✓ GPU CQT shape: {C_gpu.shape}")
                logger.info(f"✓ GPU CQT time: {gpu_time:.2f}s")
                logger.info(f"✓ Speedup: {cpu_time/gpu_time:.1f}x")
                
                C = C_gpu
            else:
                logger.info("⚠ GPU not available, using CPU")
                C = C_cpu
        except ImportError:
            logger.info("⚠ PyTorch not installed, using CPU")
            C = C_cpu
        
        # Test chroma conversion
        logger.info("Converting to chroma...")
        chroma = cqt_to_chroma(C)
        logger.info(f"✓ Chroma shape: {chroma.shape}")
        logger.info(f"✓ Chroma range: [{chroma.min():.3f}, {chroma.max():.3f}]")
        
        return C, chroma
        
    except Exception as e:
        logger.error(f"✗ CQT computation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_feature_extraction(audio: np.ndarray, sr: int):
    """Test audio feature extraction."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Audio Feature Extraction")
    logger.info("=" * 60)
    
    try:
        from music_math.audio import AudioFeatureExtractor
        
        extractor = AudioFeatureExtractor(use_gpu=True)
        
        logger.info("Extracting features...")
        start_time = time.time()
        features = extractor.extract_all(audio, sr)
        extraction_time = time.time() - start_time
        
        # Chroma features
        logger.info("\n--- Chroma Features ---")
        logger.info(f"✓ Key estimate: {features.chroma.key_estimate}")
        logger.info(f"✓ Modality: {features.chroma.modality}")
        logger.info(f"✓ Tonality strength: {features.chroma.tonality_strength:.3f}")
        logger.info(f"✓ Chroma entropy: {features.chroma.entropy:.3f}")
        
        # Rhythmic features
        logger.info("\n--- Rhythmic Features ---")
        logger.info(f"✓ Tempo: {features.rhythmic.tempo:.1f} BPM")
        logger.info(f"✓ Tempo confidence: {features.rhythmic.tempo_confidence:.3f}")
        logger.info(f"✓ Rhythm regularity: {features.rhythmic.rhythm_regularity:.3f}")
        
        # Spectral features
        logger.info("\n--- Spectral Features ---")
        logger.info(f"✓ Spectral centroid: {features.spectral.centroid_mean:.1f} Hz")
        logger.info(f"✓ Spectral rolloff: {features.spectral.rolloff_mean:.1f} Hz")
        logger.info(f"✓ Spectral flux: {features.spectral.flux_mean:.3f}")
        
        logger.info(f"\n✓ Extraction time: {extraction_time:.2f}s")
        
        return features
        
    except Exception as e:
        logger.error(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_mathematical_patterns(chroma: np.ndarray):
    """Test mathematical pattern discovery."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Mathematical Pattern Discovery")
    logger.info("=" * 60)
    
    try:
        from music_math.analysis.chroma_patterns import (
            ChromaPatternAnalyzer,
            analyze_mathematical_patterns
        )
        
        logger.info("Analyzing mathematical patterns...")
        start_time = time.time()
        patterns = analyze_mathematical_patterns(chroma)
        analysis_time = time.time() - start_time
        
        # Key segments
        logger.info("\n--- Key Segments ---")
        logger.info(f"✓ Found {len(patterns.key_segments)} key segments")
        for i, seg in enumerate(patterns.key_segments[:3]):
            logger.info(f"  {i+1}. {seg.key} ({seg.start_time:.1f}s - {seg.end_time:.1f}s, strength: {seg.strength:.3f})")
        
        # Modulations
        logger.info("\n--- Modulations ---")
        logger.info(f"✓ Found {len(patterns.modulations)} modulations")
        for i, mod in enumerate(patterns.modulations[:3]):
            logger.info(f"  {i+1}. {mod.from_key} → {mod.to_key} at {mod.time:.1f}s ({mod.type}, {mod.distance} steps)")
        
        # Harmonic entropy
        logger.info("\n--- Harmonic Entropy ---")
        logger.info(f"✓ Global entropy: {patterns.harmonic_entropy.global_entropy:.3f}")
        logger.info(f"✓ Local entropy: {patterns.harmonic_entropy.local_entropy_mean:.3f} ± {patterns.harmonic_entropy.local_entropy_std:.3f}")
        logger.info(f"✓ Information rate: {patterns.harmonic_entropy.information_rate:.1f} bits/s")
        logger.info(f"✓ Complexity score: {patterns.harmonic_entropy.complexity_score:.3f}")
        
        # Chroma geometry
        logger.info("\n--- Chroma Geometry ---")
        logger.info(f"✓ Variance: {patterns.chroma_geometry.variance:.3f}")
        logger.info(f"✓ Eccentricity: {patterns.chroma_geometry.eccentricity:.3f}")
        logger.info(f"✓ Circularity: {patterns.chroma_geometry.circularity:.3f}")
        
        # Tonal gravity
        logger.info("\n--- Tonal Gravity ---")
        top_stable = patterns.tonal_gravity.stability_ranking[:3]
        from music_math.audio.cqt import get_cqt_frequencies
        freqs = get_cqt_frequencies(n_bins=12)
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        logger.info(f"✓ Most stable pitch classes:")
        for pc, strength in top_stable:
            logger.info(f"    {note_names[pc]}: {strength:.3f}")
        
        # Fibonacci patterns
        if patterns.fibonacci_patterns:
            logger.info("\n--- Fibonacci Patterns ---")
            logger.info(f"✓ Found {len(patterns.fibonacci_patterns)} Fibonacci patterns")
            for pat in patterns.fibonacci_patterns[:2]:
                logger.info(f"  • {pat['type']}: Fibonacci({pat['fibonacci_number']}) at {pat['time']:.1f}s")
        
        # Golden ratio
        if patterns.golden_ratio_moments:
            logger.info("\n--- Golden Ratio Moments ---")
            logger.info(f"✓ Found {len(patterns.golden_ratio_moments)} φ-related moments")
            for t in patterns.golden_ratio_moments[:3]:
                logger.info(f"  • {t:.1f}s")
        
        logger.info(f"\n✓ Pattern analysis time: {analysis_time:.2f}s")
        
        return patterns
        
    except Exception as e:
        logger.error(f"✗ Pattern analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    if len(sys.argv) < 2:
        print("Usage: python test_audio_analysis.py <midi_file>")
        print("\nExample:")
        print("  python test_audio_analysis.py data/raw/piano_midi/chpn-p4.mid")
        sys.exit(1)
    
    midi_path = sys.argv[1]
    
    if not Path(midi_path).exists():
        logger.error(f"File not found: {midi_path}")
        sys.exit(1)
    
    logger.info("\n" + "=" * 60)
    logger.info("MUSIC MATH AUDIO ANALYSIS TEST")
    logger.info("=" * 60)
    logger.info(f"File: {midi_path}")
    logger.info(f"RTX 3070 GPU acceleration enabled")
    logger.info("=" * 60)
    
    # Test 1: MIDI to Audio
    audio, sr = test_midi_to_audio(midi_path)
    if audio is None:
        logger.error("Cannot continue without audio")
        sys.exit(1)
    
    # Test 2: CQT
    C, chroma = test_cqt(audio, sr)
    if chroma is None:
        logger.error("Cannot continue without chroma")
        sys.exit(1)
    
    # Test 3: Feature Extraction
    features = test_feature_extraction(audio, sr)
    if features is None:
        logger.error("Feature extraction failed")
    
    # Test 4: Mathematical Patterns
    patterns = test_mathematical_patterns(chroma)
    if patterns is None:
        logger.error("Pattern analysis failed")
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
