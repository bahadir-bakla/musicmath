#!/usr/bin/env python
"""
Küçük piano_midi dataset için uçtan uca pipeline ve analiz menüsü.

Kullanım:
    # Sadece veri pipeline'ı (eski davranış)
    python run_pipeline.py

    # Pipeline + tüm analizler
    python run_pipeline.py --run-analyses

    # Sadece analizler (feature_matrix zaten hazırsa)
    python run_pipeline.py --skip-pipeline --run-analyses
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from music_math.data.metadata import create_metadata_template
from music_math.data.pipeline import build_clean_dataset
from music_math.features.extractor import build_feature_matrix

from analyze_features import main as run_basic_analyses
from scripts.advanced_analysis import main as run_advanced_analyses
from scripts.classification_analysis import main as run_classification_analyses
from scripts.signature_metrics import main as run_signature_metrics
from scripts.brute_force_patterns import main as run_brute_force_patterns


def infer_composer(fp: str) -> str:
    """Dosya adından besteci çıkar."""
    f = fp.lower()
    if "bach" in f:
        return "Bach"
    if "mozart" in f or "mz_" in f:
        return "Mozart"
    if "chopin" in f or "chpn" in f:
        return "Chopin"
    if "deb" in f:
        return "Debussy"
    if "beethoven" in f or "mond_" in f or "appass" in f:
        return "Beethoven"
    if "liszt" in f or "liz_" in f:
        return "Liszt"
    if "schubert" in f or "schu" in f:
        return "Schubert"
    if "brahms" in f or "br_" in f:
        return "Brahms"
    if "haydn" in f or "hay_" in f:
        return "Haydn"
    if "schumann" in f or "scn" in f:
        return "Schumann"
    return "Unknown"


def infer_era(c: str) -> str:
    """Besteciden dönem çıkar."""
    if c == "Bach":
        return "Baroque"
    if c in ("Mozart", "Haydn", "Beethoven"):
        return "Classical"
    if c in ("Chopin", "Schubert", "Brahms", "Schumann", "Liszt"):
        return "Romantic"
    if c == "Debussy":
        return "Late Romantic"
    return "Unknown"


def run_core_pipeline(root: Path) -> None:
    print("=" * 60)
    print("FAZ 1+2: Piano MIDI Pipeline")
    print("=" * 60)

    # 1) data/raw altındaki tüm MIDI'ler için metadata.csv oluştur
    print("\n[1/4] Metadata template oluşturuluyor...")
    create_metadata_template(root / "data" / "raw", root / "metadata.csv")

    # 2) Basit composer / era / source doldurma
    print("\n[2/4] Composer ve era bilgileri ekleniyor...")
    meta_path = root / "metadata.csv"
    df = pd.read_csv(meta_path)

    if not df.empty:
        df["composer"] = df["file_path"].apply(infer_composer)
        df["era"] = df["composer"].apply(infer_era)
        df["source"] = "piano-midi"
        df.to_csv(meta_path, index=False)
        print(f"   ✓ {len(df)} dosya için metadata güncellendi")
        print(f"\n   Besteciler: {df['composer'].value_counts().to_dict()}")

    # 3) Kalite filtresi + data/clean doldurma + metadata_clean.csv
    print("\n[3/4] Kalite filtresi uygulanıyor ve temiz dataset oluşturuluyor...")
    clean_meta = root / "metadata_clean.csv"
    df_clean = build_clean_dataset(root / "data" / "raw", meta_path, clean_meta)
    print(f"   ✓ {len(df_clean)} dosya kalite filtresini geçti")

    # 4) Feature matrix (asıl matematiksel temsil)
    print("\n[4/4] Feature matrix oluşturuluyor (bu biraz zaman alabilir)...")
    feat_path = root / "results" / "stats" / "feature_matrix.csv"
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat = build_feature_matrix(clean_meta, feat_path)

    print("\n" + "=" * 60)
    print("Pipeline tamamlandı! 🎵")
    print("=" * 60)
    print(f"\nÇıktılar:")
    print(f"  • metadata.csv          → {meta_path}")
    print(f"  • metadata_clean.csv    → {clean_meta}")
    print(f"  • feature_matrix.csv    → {feat_path}")
    print(f"\nFeature matrix shape: {df_feat.shape}")
    print(f"  (satır={len(df_feat)} eser, sütun={len(df_feat.columns)} feature)\n")

    # Özet istatistikler
    if not df_feat.empty:
        print("Örnek feature'lar:")
        display_cols = [
            "filepath",
            "composer",
            "era",
            "pitch_entropy",
            "consonance_score",
            "interval_entropy",
            "rhythmic_entropy",
        ]
        available_cols = [c for c in display_cols if c in df_feat.columns]
        print(df_feat[available_cols].to_string(index=False))

        print(f"\n\nDönem bazında ortalama pitch_entropy:")
        if "era" in df_feat.columns and "pitch_entropy" in df_feat.columns:
            print(df_feat.groupby("era")["pitch_entropy"].mean().sort_values())


def run_all_analyses() -> None:
    """Feature matrix üzerinde tüm analiz script'lerini çalıştır."""
    print("\n" + "=" * 60)
    print("FAZ 3–4: Analiz Script'leri Çalıştırılıyor")
    print("=" * 60)

    print("\n[1/5] Temel PCA / mesafe / Markov analizleri (analyze_features.py)...")
    run_basic_analyses()

    print("\n[2/5] İleri seviye analizler (scripts/advanced_analysis.py)...")
    run_advanced_analyses()

    print("\n[3/5] Era & besteci sınıflandırma (scripts/classification_analysis.py)...")
    run_classification_analyses()

    print("\n[4/5] İmza metrikleri (scripts/signature_metrics.py)...")
    run_signature_metrics()

    print("\n[5/5] Brute-force interval motifleri (scripts/brute_force_patterns.py)...")
    run_brute_force_patterns()


def main() -> None:
    parser = argparse.ArgumentParser(description="Piano MIDI pipeline ve analiz aracı.")
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Veri pipeline'ini atla, sadece analizleri çalıştır.",
    )
    parser.add_argument(
        "--run-analyses",
        action="store_true",
        help="Feature matrix üzerinde tüm analiz script'lerini çalıştır.",
    )
    args = parser.parse_args()

    root = Path(".")

    if not args.skip_pipeline:
        run_core_pipeline(root)
    else:
        print("Veri pipeline'i atlandı (skip-pipeline=TRUE).")

    if args.run_analyses:
        run_all_analyses()


if __name__ == "__main__":
    main()
