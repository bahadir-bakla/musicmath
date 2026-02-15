"""Markov ve kısıt tabanlı temel generatif müzik üreticisi."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from music21 import chord, key, note, stream, tempo

from music_math.core.types import NoteEvent
from music_math.model.markov import MusicMarkovModel
from music_math.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClassicalMusicGenerator:
    """Matematiksel model tabanlı basit klasik müzik üreticisi."""

    markov_model: MusicMarkovModel

    def _choose_octave(self, pitch_class: int, prev_pitch: int) -> int:
        prev_octave = prev_pitch // 12
        min_oct, max_oct = 3, 6
        candidate_octaves = range(min_oct, max_oct + 1)
        return min(candidate_octaves, key=lambda o: abs(pitch_class + o * 12 - prev_pitch))

    def _sample_duration(self, style: str) -> float:
        if style == "baroque":
            options = [0.5, 1.0, 1.5, 2.0]
            weights = [0.3, 0.4, 0.15, 0.15]
        elif style == "romantic":
            options = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
            weights = [0.1, 0.2, 0.1, 0.25, 0.15, 0.1, 0.05, 0.05]
        else:
            options = [0.5, 1.0, 1.5, 2.0]
            weights = [0.25, 0.4, 0.2, 0.15]
        return float(np.random.choice(options, p=weights))

    def _get_tempo(self, style: str) -> int:
        rng = np.random.default_rng()
        if style == "baroque":
            return int(rng.integers(60, 101))
        if style == "classical":
            return int(rng.integers(80, 131))
        if style == "romantic":
            return int(rng.integers(50, 111))
        if style == "late_romantic":
            return int(rng.integers(55, 101))
        return 90

    @staticmethod
    def _midi_to_key(midi_note: int) -> str:
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return key_names[midi_note % 12]

    def generate(
        self,
        style: str = "classical",
        length_bars: int = 32,
        tonic: int = 60,
        time_signature: int = 4,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> Tuple[stream.Score, List[NoteEvent]]:
        """Ana üretim fonksiyonu."""
        if seed is not None:
            np.random.seed(seed)

        score = stream.Score()
        part = stream.Part()

        part.append(tempo.MetronomeMark(number=self._get_tempo(style)))
        part.append(key.Key(self._midi_to_key(tonic)))

        notes: List[NoteEvent] = []
        current_pitch = tonic
        beats_per_bar = time_signature
        total_beats = float(length_bars * beats_per_bar)
        current_beat = 0.0

        max_attempts = int(total_beats * 3)
        attempts = 0

        while current_beat < total_beats and attempts < max_attempts:
            attempts += 1

            # Pitch-class Markov'tan örnekle
            current_pc = current_pitch % 12
            probs = self.markov_model.pitch_transitions[current_pc].astype(float)
            if temperature != 1.0:
                probs = np.power(probs + 1e-8, 1.0 / temperature)
            probs = probs / probs.sum()
            next_pc = int(np.random.choice(12, p=probs))

            octave = self._choose_octave(next_pc, current_pitch)
            midi_pitch = next_pc + octave * 12

            duration = self._sample_duration(style)
            if current_beat + duration > total_beats:
                duration = total_beats - current_beat
            if duration <= 0:
                break

            n = note.Note(midi=midi_pitch)
            n.duration.quarterLength = duration
            part.append(n)

            notes.append(
                NoteEvent(
                    pitch=int(midi_pitch),
                    duration=float(duration),
                    start=float(current_beat),
                )
            )

            current_pitch = midi_pitch
            current_beat += duration

        score.append(part)
        return score, notes


def save_score_as_midi(score: stream.Score, filepath: str | Path) -> Path:
    """Bir music21 Score nesnesini MIDI olarak kaydet."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(path))
    logger.info("MIDI dosyası kaydedildi: %s", path)
    return path


__all__ = ["ClassicalMusicGenerator", "save_score_as_midi"]

