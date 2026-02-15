"""Tip tanımları ve ortak veri modelleri."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class NoteEvent:
    """
    MIDI benzeri tek bir nota olayı.

    Attributes:
        pitch: MIDI nota numarası (0–127)
        duration: Çeyrek nota cinsinden süre (quarterLength)
        start: Eser başlangıcına göre offset (quarterLength)
    """

    pitch: int
    duration: float
    start: float


class FeatureExtractor(Protocol):
    """Feature extraction fonksiyonları için basit bir protokol."""

    def __call__(self, *args, **kwargs) -> dict:
        ...


__all__ = ["NoteEvent", "FeatureExtractor"]

