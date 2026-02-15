#!/usr/bin/env python
"""
MAESTRO v3.0.0 dataset indirme, metadata oluşturma, kalite filtresi ve feature matrix.

~1200+ piyano kaydı; tek komutla 1000+ eser ve feature_matrix.csv üretir.

Kullanım:
    python scripts/download_maestro.py

İsteğe bağlı (zaten indirdiysen sadece metadata + pipeline):
    python scripts/download_maestro.py --skip-download
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger
from music_math.data.quality import apply_quality_filter
from music_math.features.extractor import build_feature_matrix

logger = get_logger(__name__)

MAESTRO_ZIP_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
MAESTRO_DIR_NAME = "maestro-v3.0.0"
CSV_NAME = "maestro-v3.0.0.csv"

# canonical_composer -> era (MAESTRO'daki isimlere göre)
COMPOSER_TO_ERA = {
    "Johann Sebastian Bach": "Baroque",
    "Antonio Vivaldi": "Baroque",
    "George Frideric Handel": "Baroque",
    "Wolfgang Amadeus Mozart": "Classical",
    "Joseph Haydn": "Classical",
    "Ludwig van Beethoven": "Classical",
    "Franz Schubert": "Romantic",
    "Frédéric Chopin": "Romantic",
    "Robert Schumann": "Romantic",
    "Johannes Brahms": "Romantic",
    "Pyotr Ilyich Tchaikovsky": "Romantic",
    "Franz Liszt": "Romantic",
    "Claude Debussy": "Late Romantic",
    "Sergei Rachmaninoff": "Late Romantic",
    "Maurice Ravel": "Late Romantic",
}


def composer_to_era(composer: str) -> str:
    return COMPOSER_TO_ERA.get(composer.strip(), "Other")


def download_maestro_zip(root: Path) -> Path:
    zip_path = root / "data" / "raw" / "maestro-v3.0.0.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        logger.info("MAESTRO zip zaten mevcut: %s", zip_path)
        return zip_path
    logger.info("MAESTRO indiriliyor (~1.2 GB)...")
    try:
        import urllib.request
        urllib.request.urlretrieve(MAESTRO_ZIP_URL, zip_path)
    except Exception as e:
        logger.error("İndirme hatası: %s", e)
        raise
    return zip_path


def unzip_maestro(root: Path, zip_path: Path) -> Path:
    extract_to = root / "data" / "raw"
    extract_to.mkdir(parents=True, exist_ok=True)
    maestro_dir = extract_to / MAESTRO_DIR_NAME
    if maestro_dir.exists() and (maestro_dir / CSV_NAME).exists():
        logger.info("MAESTRO zaten açılmış: %s", maestro_dir)
        return maestro_dir
    logger.info("MAESTRO zip açılıyor...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    return extract_to / MAESTRO_DIR_NAME


def build_metadata_from_maestro_csv(maestro_dir: Path, root: Path) -> pd.DataFrame:
    csv_path = maestro_dir / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"MAESTRO metadata bulunamadı: {csv_path}")

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            midi_filename = row.get("midi_filename", "").strip()
            if not midi_filename or not (midi_filename.endswith(".mid") or midi_filename.endswith(".midi")):
                continue
            # MAESTRO path: 2018/MIDI-Unprocessed_....midi (maestro_dir root'a göre relatif)
            rel_dir = maestro_dir.relative_to(root)
            rel_path = str(rel_dir / midi_filename).replace("\\", "/")
            full = root / rel_path
            if not full.exists():
                continue
            composer = row.get("canonical_composer", "Unknown").strip()
            era = composer_to_era(composer)
            rows.append({
                "file_path": rel_path,
                "composer": composer,
                "full_name": composer,
                "birth_year": 0,
                "death_year": 0,
                "era": era,
                "composition_year": int(row.get("year", 0) or 0),
                "form": "",
                "key": "",
                "instrumentation": "solo_piano",
                "tempo_marking": "",
                "duration_seconds": float(row.get("duration", 0) or 0),
                "total_notes": 0,
                "source": "maestro",
                "quality_flag": 1,
                "quality_note": "",
            })

    df = pd.DataFrame(rows)
    meta_path = root / "metadata.csv"
    df.to_csv(meta_path, index=False)
    logger.info("metadata.csv yazıldı: %d satır → %s", len(df), meta_path)
    return df


def main():
    parser = argparse.ArgumentParser(description="MAESTRO indir, metadata + pipeline çalıştır")
    parser.add_argument("--skip-download", action="store_true", help="Zip indirme / açma atla, sadece metadata + pipeline")
    parser.add_argument("--skip-features", action="store_true", help="Feature matrix oluşturma atla (sadece metadata + temizlik)")
    args = parser.parse_args()

    root = CONFIG.paths.root

    if not args.skip_download:
        zip_path = download_maestro_zip(root)
        maestro_dir = unzip_maestro(root, zip_path)
    else:
        # Hem data/raw/maestro-v3.0.0 hem data/raw/piano_midi/maestro-v3.0.0 aranır
        candidates = [
            root / "data" / "raw" / MAESTRO_DIR_NAME,
            root / "data" / "raw" / "piano_midi" / MAESTRO_DIR_NAME,
        ]
        maestro_dir = None
        for d in candidates:
            if d.exists() and (d / CSV_NAME).exists():
                maestro_dir = d
                break
        if maestro_dir is None:
            print("Hata: --skip-download kullandın ama data/raw/maestro-v3.0.0 veya data/raw/piano_midi/maestro-v3.0.0 yok.")
            sys.exit(1)

    print("\n[1/3] Metadata oluşturuluyor (MAESTRO CSV)...")
    build_metadata_from_maestro_csv(maestro_dir, root)

    meta_path = root / "metadata.csv"
    clean_path = root / "metadata_clean.csv"
    print("\n[2/3] Kalite filtresi uygulanıyor...")
    apply_quality_filter(meta_path, clean_path)

    df_clean = pd.read_csv(clean_path)
    print(f"   Temiz eser sayısı: {len(df_clean)}")

    if not args.skip_features:
        feat_path = root / "results" / "stats" / "feature_matrix.csv"
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        print("\n[3/3] Feature matrix oluşturuluyor (bu uzun sürebilir)...")
        build_feature_matrix(clean_path, feat_path)
        print(f"   Feature matrix: {feat_path}")
    else:
        print("\n[3/3] Feature matrix atlandı (--skip-features).")

    print("\nBitti. Sonraki: python analyze_features.py && python scripts/advanced_analysis.py")


if __name__ == "__main__":
    main()
