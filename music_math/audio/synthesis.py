"""
MIDI-to-Audio Synthesis Module

FluidSynth kullanarak MIDI dosyalarını audio'ya çevirir.
ASUS ROG Strix RTX 3070 için optimize edilmiştir.

Usage:
    from music_math.audio.synthesis import midi_to_audio, get_synth
    audio, sr = midi_to_audio("piece.mid")
"""

from __future__ import annotations

import os
import io
import tempfile
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Union
import logging

import numpy as np

logger = logging.getLogger(__name__)

# SoundFont yolu (proje kök dizininde olacak)
DEFAULT_SOUNDFONT = Path(__file__).parent.parent.parent / "data" / "soundfonts" / "GeneralUser_GS_SoftSynth.sf2"
SAMPLE_RATE = 22050  # CQT için ideal sample rate


class FluidSynthWrapper:
    """FluidSynth wrapper with caching and optimization."""
    
    def __init__(self, soundfont_path: Optional[Union[str, Path]] = None):
        """
        Initialize FluidSynth.
        
        Args:
            soundfont_path: Path to SF2 soundfont file. Uses default if None.
        """
        self.soundfont_path = Path(soundfont_path) if soundfont_path else DEFAULT_SOUNDFONT
        self._synth = None
        self._driver = None
        
        # Check if soundfont exists
        if not self.soundfont_path.exists():
            logger.warning(f"SoundFont not found: {self.soundfont_path}")
            logger.info("Please download GeneralUser_GS_SoftSynth.sf2 and place it in data/soundfonts/")
            self.soundfont_path = None
    
    def _init_synth(self, sample_rate: int = SAMPLE_RATE) -> bool:
        """Initialize FluidSynth instance."""
        try:
            import fluidsynth
            
            # Create synthesizer
            self._synth = fluidsynth.Synth(samplerate=sample_rate)
            
            # Load soundfont if available
            if self.soundfont_path and self.soundfont_path.exists():
                sfid = self._synth.sfload(str(self.soundfont_path))
                if sfid < 0:
                    logger.error("Failed to load SoundFont")
                    return False
                logger.info(f"Loaded SoundFont: {self.soundfont_path.name}")
            else:
                logger.warning("No SoundFont loaded, using default")
            
            return True
            
        except ImportError:
            logger.error("pyfluidsynth not installed. Run: pip install pyfluidsynth")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize FluidSynth: {e}")
            return False
    
    def render_midi(
        self,
        midi_path: Union[str, Path],
        sample_rate: int = SAMPLE_RATE,
        duration_limit: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Render MIDI file to audio.
        
        Args:
            midi_path: Path to MIDI file
            sample_rate: Output sample rate
            duration_limit: Maximum duration in seconds (None for full file)
            
        Returns:
            (audio_array, sample_rate) tuple
        """
        midi_path = Path(midi_path)
        
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        
        # Initialize synth
        if not self._init_synth(sample_rate):
            raise RuntimeError("Failed to initialize FluidSynth")
        
        try:
            import fluidsynth
            
            # Load MIDI file
            player = fluidsynth.Player(self._synth)
            player.add(str(midi_path))
            
            # Get MIDI duration (approximate)
            if duration_limit:
                total_samples = int(sample_rate * duration_limit)
            else:
                # Estimate duration from MIDI (rough estimate)
                total_samples = int(sample_rate * 300)  # 5 minutes max default
            
            # Render audio
            logger.info(f"Rendering MIDI: {midi_path.name} (sr={sample_rate})")
            
            # Process in chunks to save memory
            chunk_size = sample_rate * 10  # 10 seconds chunks
            audio_chunks = []
            
            player.play()
            
            while True:
                # Get audio chunk
                chunk = self._synth.get_samples(chunk_size)
                
                if not chunk or len(chunk) == 0:
                    break
                
                # Convert to numpy
                chunk_arr = np.array(chunk, dtype=np.float32)
                
                # Stereo to mono (average channels)
                if len(chunk_arr.shape) > 1:
                    chunk_arr = np.mean(chunk_arr, axis=1)
                
                audio_chunks.append(chunk_arr)
                
                # Check duration limit
                total_rendered = sum(len(c) for c in audio_chunks)
                if duration_limit and total_rendered >= total_samples:
                    break
            
            player.stop()
            
            # Concatenate chunks
            if audio_chunks:
                audio = np.concatenate(audio_chunks)
            else:
                audio = np.array([], dtype=np.float32)
            
            logger.info(f"Rendered {len(audio)/sample_rate:.2f}s of audio")
            
            return audio, sample_rate
            
        except Exception as e:
            logger.error(f"Error rendering MIDI: {e}")
            raise
        finally:
            # Cleanup
            if self._synth:
                self._synth.delete()
                self._synth = None
    
    def __del__(self):
        """Cleanup."""
        if self._synth:
            try:
                self._synth.delete()
            except:
                pass


# Global synth instance (singleton)
_synth_instance: Optional[FluidSynthWrapper] = None


def get_synth(soundfont_path: Optional[Union[str, Path]] = None) -> FluidSynthWrapper:
    """Get or create FluidSynth instance."""
    global _synth_instance
    if _synth_instance is None:
        _synth_instance = FluidSynthWrapper(soundfont_path)
    return _synth_instance


def midi_to_audio(
    midi_path: Union[str, Path],
    sample_rate: int = SAMPLE_RATE,
    duration_limit: Optional[float] = None,
    soundfont_path: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, int]:
    """
    Convert MIDI file to audio using FluidSynth.
    
    Args:
        midi_path: Path to MIDI file
        sample_rate: Output sample rate (default 22050 for CQT)
        duration_limit: Maximum duration in seconds
        soundfont_path: Custom SoundFont path
        
    Returns:
        (audio_array, sample_rate) tuple
        
    Example:
        >>> audio, sr = midi_to_audio("bach.mid")
        >>> print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
    """
    synth = get_synth(soundfont_path)
    return synth.render_midi(midi_path, sample_rate, duration_limit)


def render_midi_bytes(
    midi_data: bytes,
    sample_rate: int = SAMPLE_RATE,
    duration_limit: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Render MIDI from bytes (for web upload scenarios).
    
    Args:
        midi_data: MIDI file bytes
        sample_rate: Output sample rate
        duration_limit: Maximum duration
        
    Returns:
        (audio_array, sample_rate) tuple
    """
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        f.write(midi_data)
        temp_path = f.name
    
    try:
        audio, sr = midi_to_audio(temp_path, sample_rate, duration_limit)
        return audio, sr
    finally:
        # Cleanup temp file
        os.unlink(temp_path)


def get_audio_hash(midi_path: Union[str, Path], soundfont_path: Optional[Path] = None) -> str:
    """
    Generate unique hash for MIDI + SoundFont combination.
    Used for caching.
    
    Args:
        midi_path: Path to MIDI file
        soundfont_path: Path to SoundFont
        
    Returns:
        MD5 hash string
    """
    midi_path = Path(midi_path)
    
    # Hash MIDI content + SoundFont path
    hasher = hashlib.md5()
    
    # Add MIDI file hash
    with open(midi_path, 'rb') as f:
        hasher.update(f.read())
    
    # Add SoundFont identifier
    sf_path = soundfont_path or DEFAULT_SOUNDFONT
    hasher.update(str(sf_path).encode())
    
    return hasher.hexdigest()


__all__ = [
    'FluidSynthWrapper',
    'get_synth',
    'midi_to_audio',
    'render_midi_bytes',
    'get_audio_hash',
    'DEFAULT_SOUNDFONT',
    'SAMPLE_RATE',
]
