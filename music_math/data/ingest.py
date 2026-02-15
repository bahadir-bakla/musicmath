"""
Veri ingestion yardımcıları.

Bu modül, harici dataset'lerin (ör. MAESTRO) ve yerel MIDI arşivlerinin
proje dizin yapısına yerleştirilmesi için temel fonksiyonları tanımlar.

Not: Ağ erişimi bu ortamda kısıtlı olabileceği için, indirme fonksiyonları
çoğunlukla kullanıcı tarafından çalıştırılmak üzere tasarlanmıştır.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger

logger = get_logger(__name__)


def copy_local_midis(source_dirs: Iterable[str | Path]) -> None:
    """
    Yerel klasörlerdeki MIDI dosyalarını `data/raw/` altına kopyala.

    Args:
        source_dirs: MIDI dosyalarının bulunduğu klasörler.
    """
    raw_root = CONFIG.paths.data_raw
    raw_root.mkdir(parents=True, exist_ok=True)

    total_copied = 0

    for directory in source_dirs:
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.warning("Kaynak dizin bulunamadı: %s", directory_path)
            continue

        for midi_path in directory_path.rglob("*.mid"):
            rel = midi_path.name
            dest = raw_root / rel
            dest = _avoid_overwrite(dest)
            shutil.copy2(midi_path, dest)
            total_copied += 1

        for midi_path in directory_path.rglob("*.midi"):
            rel = midi_path.name
            dest = raw_root / rel
            dest = _avoid_overwrite(dest)
            shutil.copy2(midi_path, dest)
            total_copied += 1

    logger.info("Toplam %d MIDI dosyası data/raw/ altına kopyalandı.", total_copied)


def _avoid_overwrite(path: Path) -> Path:
    """
    Aynı isimli dosyaların üzerine yazmamak için basit bir isimlendirme stratejisi.
    """
    candidate = path
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1
    return candidate


__all__ = ["copy_local_midis"]

