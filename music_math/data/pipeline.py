"""
FAZ 1 veri pipeline'ı: metadata oluşturma, kalite filtresi ve temiz dizine kopyalama.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger
from music_math.data.metadata import create_metadata_template
from music_math.data.quality import apply_quality_filter

logger = get_logger(__name__)


def build_clean_dataset(
    midi_dir: str | Path | None = None,
    metadata_csv: str | Path | None = None,
    clean_metadata_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    FAZ 1 için uçtan uca pipeline:

    1. `data/raw/` altındaki MIDI dosyalarından metadata şablonu oluştur.
    2. Kalite filtresi uygula, `*_clean.csv` üret.
    3. Geçerli dosyaları `data/clean/` altına kopyala.
    """
    root = CONFIG.paths.root
    midi_root = Path(midi_dir) if midi_dir is not None else CONFIG.paths.data_raw

    # 1) Metadata şablonu (varsa oku, yoksa oluştur)
    if metadata_csv is None:
        metadata_csv = root / "metadata.csv"
    metadata_csv = Path(metadata_csv)

    # Eğer metadata.csv zaten varsa, onu kullan (üzerine yazma!)
    if metadata_csv.exists():
        logger.info("Mevcut metadata.csv kullanılıyor: %s", metadata_csv)
        df_meta = pd.read_csv(metadata_csv)
    else:
        df_meta = create_metadata_template(midi_root, metadata_csv)
    
    if df_meta.empty:
        logger.warning("Hiç MIDI bulunamadı, pipeline durdu.")
        return df_meta

    # 2) Kalite filtresi
    df_clean = apply_quality_filter(metadata_csv, clean_metadata_csv)

    # 3) Temiz dosyaları data/clean altına kopyala
    clean_root = CONFIG.paths.data_clean
    clean_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for _, row in df_clean.iterrows():
        rel = row["file_path"]
        src = root / rel
        if not src.exists():
            logger.warning("Temizlenecek dosya bulunamadı: %s", src)
            continue
        dest = clean_root / src.name
        shutil.copy2(src, dest)
        copied += 1

    logger.info(
        "Temiz dataset hazır: %d dosya data/clean/ altına kopyalandı.", copied
    )

    return df_clean


__all__ = ["build_clean_dataset"]

