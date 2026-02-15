"""
Metadata ve MIDI dosyalarını yüklemek için yardımcı fonksiyonlar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List

import pandas as pd
from music21 import converter, note, chord

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger
from music_math.core.types import NoteEvent

logger = get_logger(__name__)


def load_metadata(path: str | Path | None = None) -> pd.DataFrame:
    """
    Metadata CSV'yi yükle.

    Varsayılan: proje kökünde `metadata_clean.csv` varsa onu, yoksa `metadata.csv`.
    """
    root = CONFIG.paths.root
    if path is not None:
        p = Path(path)
    else:
        clean = root / "metadata_clean.csv"
        base = root / "metadata.csv"
        p = clean if clean.exists() else base

    if not p.exists():
        logger.warning("Metadata dosyası bulunamadı: %s", p)
        return pd.DataFrame()

    return pd.read_csv(p)


def iter_midi_files(metadata_df: pd.DataFrame) -> Iterator[Path]:
    """
    Metadata DataFrame'inden MIDI yol bilgilerini döndür.
    """
    root = CONFIG.paths.root
    for _, row in metadata_df.iterrows():
        rel = row["file_path"]
        yield root / rel


def parse_midi_to_note_events(filepath: str | Path) -> List[NoteEvent]:
    """
    Tek bir MIDI dosyasını `NoteEvent` listesine çevir.
    """
    score = converter.parse(str(filepath))
    events: List[NoteEvent] = []

    for el in score.flatten().notes:
        if isinstance(el, note.Note):
            events.append(
                NoteEvent(
                    pitch=int(el.pitch.midi),
                    duration=float(el.duration.quarterLength),
                    start=float(el.offset),
                )
            )
        elif isinstance(el, chord.Chord):
            for n in el.notes:
                events.append(
                    NoteEvent(
                        pitch=int(n.pitch.midi),
                        duration=float(el.duration.quarterLength),
                        start=float(el.offset),
                    )
                )

    return events


__all__ = ["load_metadata", "iter_midi_files", "parse_midi_to_note_events"]

