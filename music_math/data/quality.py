"""
MIDI dosyaları için kalite kontrol fonksiyonları.

FAZ 1'deki `check_midi_quality` taslağını temel alır.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from music21 import converter

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityConfig:
    min_notes: int = 50
    max_duration_quarter_length: float = 1200.0  # yaklaşık 20 dakika


def check_midi_quality(
    filepath: str | Path,
    min_notes: int | None = None,
    max_duration: float | None = None,
) -> Tuple[bool, str]:
    """
    Basit kalite kontrolü.

    Returns:
        (is_valid, reason)
    """
    cfg = QualityConfig()
    min_notes = min_notes if min_notes is not None else cfg.min_notes
    max_duration = max_duration if max_duration is not None else cfg.max_duration_quarter_length

    try:
        score = converter.parse(str(filepath))

        notes = list(score.flatten().notes)
        if len(notes) < min_notes:
            return False, f"Too few notes: {len(notes)}"

        duration = score.duration.quarterLength
        if duration > max_duration:
            return False, f"Too long: {duration:.0f} quarterLength"

        return True, "OK"

    except Exception as exc:  # pragma: no cover - hata mesajı ortamdan bağımsız
        return False, f"Parse error: {exc}"


def apply_quality_filter(
    metadata_csv: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Metadata CSV'deki tüm dosyalara kalite filtresi uygula.

    `quality_flag` ve `quality_note` alanlarını günceller, geçerli
    kayıtları içeren yeni bir CSV yazabilir.
    """
    input_path = Path(metadata_csv)
    if output_csv is None:
        output_path = input_path.with_name(input_path.stem + "_clean.csv")
    else:
        output_path = Path(output_csv)

    df = pd.read_csv(input_path)
    updated_rows = []

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = Path(CONFIG.paths.root) / rel_path
        is_valid, reason = check_midi_quality(full_path)
        row["quality_flag"] = 1 if is_valid else 0
        row["quality_note"] = reason
        updated_rows.append(row)

    df_updated = pd.DataFrame(updated_rows)
    df_clean = df_updated[df_updated["quality_flag"] == 1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    logger.info(
        "Kalite filtresi: toplam=%d | temiz=%d | elenen=%d | çıktı=%s",
        len(df_updated),
        len(df_clean),
        len(df_updated) - len(df_clean),
        output_path,
    )

    return df_clean


__all__ = ["QualityConfig", "check_midi_quality", "apply_quality_filter"]

