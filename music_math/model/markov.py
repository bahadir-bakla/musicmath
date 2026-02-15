"""Pitch ve süre için çok katmanlı Markov modeli."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pickle

from music_math.core.types import NoteEvent
from music_math.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MusicMarkovModel:
    """Pitch-class ve süre geçişlerini modelleyen basit 1. dereceden Markov modeli."""

    order: int = 1
    pitch_transitions: np.ndarray = field(default_factory=lambda: np.zeros((12, 12), dtype=float))
    duration_transitions: np.ndarray = field(default_factory=lambda: np.zeros((8, 8), dtype=float))

    def fit(self, notes_data_list: Iterable[List[NoteEvent]]) -> "MusicMarkovModel":
        """Birden fazla eserin nota dizilerinden geçiş matrislerini öğren."""
        pc_matrix = np.zeros((12, 12), dtype=float)
        dur_matrix = np.zeros((8, 8), dtype=float)

        for events in notes_data_list:
            if len(events) < 2:
                continue
            pitches = [e.pitch % 12 for e in events]
            durations = [self._quantize_duration(e.duration) for e in events]

            for i in range(len(pitches) - self.order):
                pc_matrix[pitches[i], pitches[i + 1]] += 1.0

            for i in range(len(durations) - self.order):
                d1, d2 = durations[i], durations[i + 1]
                if 0 <= d1 < 8 and 0 <= d2 < 8:
                    dur_matrix[d1, d2] += 1.0

        self.pitch_transitions = self._normalize_matrix(pc_matrix)
        self.duration_transitions = self._normalize_matrix(dur_matrix)
        return self

    @staticmethod
    def _quantize_duration(dur: float, levels: int = 8) -> int:
        """Süreyi kaba 1/8'lik birimlere kuantize et."""
        return int(min(max(dur * 2, 0), levels - 1))

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return matrix / row_sums

    def generate_pitch_sequence(
        self,
        length: int = 64,
        start_pitch: int = 60,
        temperature: float = 1.0,
    ) -> List[int]:
        """Pitch-class Markov matrisinden nota dizisi üret."""
        sequence = [start_pitch % 12]
        for _ in range(length - 1):
            current_pc = sequence[-1]
            probs = self.pitch_transitions[current_pc].astype(float)
            if temperature != 1.0:
                probs = np.power(probs + 1e-8, 1.0 / temperature)
            probs = probs / probs.sum()
            next_pc = int(np.random.choice(12, p=probs))
            sequence.append(next_pc)
        return sequence

    def transition_entropy(self) -> float:
        """Geçiş matrisinin ortalama satır entropisi."""
        entropies = []
        for row in self.pitch_transitions:
            row = row[row > 0]
            if row.size == 0:
                continue
            ent = float(-(row * np.log2(row)).sum())
            entropies.append(ent)
        return float(np.mean(entropies)) if entropies else 0.0

    def similarity(self, other: "MusicMarkovModel") -> float:
        """İki Markov modelini kosinüs benzerliği ile karşılaştır."""
        v1 = self.pitch_transitions.flatten()
        v2 = other.pitch_transitions.flatten()
        dot = float(np.dot(v1, v2))
        norm = float(np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
        return dot / norm

    def save(self, filepath: str | Path) -> None:
        """Modeli pickle ile kaydet."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        logger.info("MusicMarkovModel kaydedildi: %s", path)

    @classmethod
    def load(cls, filepath: str | Path) -> "MusicMarkovModel":
        """Pickle'dan modeli yükle."""
        path = Path(filepath)
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Dosyadaki nesne MusicMarkovModel değil: {type(obj)}")
        logger.info("MusicMarkovModel yüklendi: %s", path)
        return obj


__all__ = ["MusicMarkovModel"]

