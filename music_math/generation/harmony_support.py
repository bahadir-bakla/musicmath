"""Temel tonalite ve kadans kuralları için yardımcılar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


DIATONIC_SCALE = {
    "C_major": [0, 2, 4, 5, 7, 9, 11],
    "G_major": [7, 9, 11, 0, 2, 4, 6],
    "F_major": [5, 7, 9, 10, 0, 2, 4],
    "A_minor": [9, 11, 0, 2, 4, 5, 7],
    "D_minor": [2, 4, 5, 7, 9, 10, 0],
}


@dataclass
class HarmonySupport:
    """Basit tonalite ve çözülme kuralları."""

    key: str = "C_major"

    def __post_init__(self) -> None:
        self.scale: List[int] = DIATONIC_SCALE.get(self.key, DIATONIC_SCALE["C_major"])

    def is_diatonic(self, pitch_class: int) -> bool:
        return pitch_class in self.scale

    def diatonic_probability_boost(self, probs, boost_factor: float = 1.5):
        """Diyatonik notalara ekstra ağırlık verir."""
        boosted = probs.copy()
        for pc in self.scale:
            boosted[pc] *= boost_factor
        boosted = boosted / boosted.sum()
        return boosted

    def phrase_ending_note(self) -> int:
        """Basit fraz sonu için tonic pitch-class."""
        return self.scale[0]


__all__ = ["HarmonySupport", "DIATONIC_SCALE"]

