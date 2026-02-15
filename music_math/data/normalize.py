"""
Tempo, süre ve benzeri özellikleri normalize etmek için yardımcı fonksiyonlar.

Şimdilik minimal; ileride FAZ 2–3 bulgularına göre genişletilebilir.
"""

from __future__ import annotations

from pathlib import Path

from music21 import converter

from music_math.core.logging import get_logger

logger = get_logger(__name__)


def normalize_tempo_to_score(
    filepath: str | Path,
    target_bpm: int = 120,
):
    """
    Bir MIDI dosyasını okuyup tempo işaretlerini normalize eden basit yardımcı.

    Not:
        - Şimdilik music21'deki MetronomeMark öğelerini tek bir değere çekiyor.
        - Pitch ilişkilerini değiştirmez, sadece zaman ölçeğini etkiler.
    """
    score = converter.parse(str(filepath))
    # MetronomeMark öğelerini sadeleştir
    mms = score.flat.getElementsByClass("MetronomeMark")
    for mm in mms:
        mm.number = target_bpm

    return score


__all__ = ["normalize_tempo_to_score"]

