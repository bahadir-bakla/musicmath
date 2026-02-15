"""
MIDI dosyaları için metadata şablonu oluşturma ve yönetme aracı.

FAZ 1 planındaki `metadata.csv` yapısını baz alır.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger

logger = get_logger(__name__)


METADATA_COLUMNS: Dict[str, type] = {
    "file_path": str,
    "composer": str,
    "full_name": str,
    "birth_year": int,
    "death_year": int,
    "era": str,
    "composition_year": int,
    "form": str,
    "key": str,
    "instrumentation": str,
    "tempo_marking": str,
    "duration_seconds": float,
    "total_notes": int,
    "source": str,
    "quality_flag": int,
    "quality_note": str,
}


@dataclass
class MetadataConfig:
    midi_root: Path = CONFIG.paths.data_raw
    output_csv: Path = CONFIG.paths.root / "metadata.csv"


def create_metadata_template(
    midi_dir: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Tüm MIDI dosyalarını tarayıp boş metadata satırları oluştur.

    Manüel olarak doldurulacak alanlar boş bırakılır.
    """
    cfg = MetadataConfig()
    midi_root = Path(midi_dir) if midi_dir is not None else cfg.midi_root
    output_path = Path(output_csv) if output_csv is not None else cfg.output_csv

    records: List[dict] = []

    if not midi_root.exists():
        logger.warning("MIDI dizini bulunamadı: %s", midi_root)
        return pd.DataFrame(columns=METADATA_COLUMNS.keys())

    for path in midi_root.rglob("*"):
        if path.suffix.lower() not in {".mid", ".midi"}:
            continue

        # Ensure both paths are absolute for proper relative_to calculation
        abs_path = path.resolve()
        abs_root = CONFIG.paths.root.resolve()
        
        records.append(
            {
                "file_path": str(abs_path.relative_to(abs_root)),
                "composer": "",
                "full_name": "",
                "birth_year": 0,
                "death_year": 0,
                "era": "",
                "composition_year": 0,
                "form": "",
                "key": "",
                "instrumentation": "",
                "tempo_marking": "",
                "duration_seconds": 0.0,
                "total_notes": 0,
                "source": "",
                "quality_flag": 1,
                "quality_note": "",
            }
        )

    df = pd.DataFrame(records, columns=list(METADATA_COLUMNS.keys()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("%d MIDI dosyası bulundu. Metadata CSV: %s", len(df), output_path)
    return df


def load_metadata(path: str | Path | None = None) -> pd.DataFrame:
    """Metadata CSV'yi oku; yoksa boş DataFrame döndür."""
    meta_path = Path(path) if path is not None else MetadataConfig().output_csv
    if not meta_path.exists():
        logger.warning("Metadata CSV bulunamadı: %s", meta_path)
        return pd.DataFrame(columns=METADATA_COLUMNS.keys())
    return pd.read_csv(meta_path)


__all__ = ["METADATA_COLUMNS", "MetadataConfig", "create_metadata_template", "load_metadata"]

