#!/usr/bin/env python
"""
Matematiksel Pattern Keşif Aracı:
- Asal sayı harmoni analizi
- Golden Ratio & Fibonacci pattern'leri
- Bestecilere özgü matematiksel imzalar

Kullanım:
    python scripts/mathematical_patterns.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger
from music_math.data.loader import parse_midi_to_note_events
from music_math.analysis.prime_harmony import analyze_prime_structure
from music_math.analysis.golden_ratio import analyze_golden_ratio_structure

logger = get_logger(__name__)


def analyze_all_files() -> pd.DataFrame:
    """Tüm MIDI dosyaları için matematiksel pattern analizi."""
    meta_clean = CONFIG.paths.root / "metadata_clean.csv"
    
    if not meta_clean.exists():
        logger.error("metadata_clean.csv bulunamadı. Önce pipeline çalıştırın.")
        return pd.DataFrame()
    
    df_meta = pd.read_csv(meta_clean)
    
    results = []
    
    for _, row in df_meta.iterrows():
        filepath = CONFIG.paths.root / row['file_path']
        
        try:
            events = parse_midi_to_note_events(filepath)
            
            if len(events) < 20:
                continue
            
            # Asal sayı analizi
            prime_analysis = analyze_prime_structure(events)
            
            # Golden Ratio analizi
            golden_analysis = analyze_golden_ratio_structure(events)
            
            # Birleştir
            record = {
                "filepath": str(filepath.relative_to(CONFIG.paths.root)),
                "composer": row.get('composer', 'Unknown'),
                "era": row.get('era', 'Unknown'),
                
                # Prime metrics
                "interval_prime_density": prime_analysis.get('interval_prime_density', 0.0),
                "duration_prime_ratio": prime_analysis.get('duration_prime_ratio', 0.0),
                "num_prime_phrase_lengths": len(prime_analysis.get('phrase_length_primes', [])),
                
                # Golden Ratio metrics
                "climax_position_ratio": golden_analysis.get('climax_position_ratio', 0.0),
                "climax_golden_distance": golden_analysis.get('climax_golden_distance', 1.0),
                "climax_is_golden": golden_analysis.get('climax_is_golden', False),
                "fibonacci_section_ratio": golden_analysis.get('fibonacci_section_ratio', 0.0),
                "golden_ratio_in_durations": golden_analysis.get('golden_ratio_in_durations', 0.0),
                
                "total_notes": len(events),
            }
            
            results.append(record)
            
        except Exception as e:
            logger.warning("Analiz hatası: %s (%s)", filepath, e)
            continue
    
    df_results = pd.DataFrame(results)
    
    # Kaydet
    out_path = CONFIG.paths.stats / "mathematical_patterns.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)
    
    logger.info("Matematiksel pattern analizi tamamlandı: %s", out_path)
    return df_results


def plot_composer_prime_profile(df: pd.DataFrame):
    """Bestecilere göre asal sayı kullanımı."""
    if df.empty or 'composer' not in df.columns:
        return
    
    figures_dir = CONFIG.paths.figures
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Bestecilere göre ortalama
    composer_primes = df.groupby('composer')[
        ['interval_prime_density', 'duration_prime_ratio']
    ].mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Interval prime density
    composer_primes['interval_prime_density'].plot(
        kind='bar',
        ax=axes[0],
        color='steelblue',
        edgecolor='black'
    )
    axes[0].set_title("Interval'lerde Asal Sayı Yoğunluğu", fontweight='bold')
    axes[0].set_ylabel("Oran (0-1)", fontweight='bold')
    axes[0].set_xlabel("Besteci", fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Duration prime ratio
    composer_primes['duration_prime_ratio'].plot(
        kind='bar',
        ax=axes[1],
        color='coral',
        edgecolor='black'
    )
    axes[1].set_title("Nota Sürelerinde Asal Sayı Oranı", fontweight='bold')
    axes[1].set_ylabel("Oran (0-1)", fontweight='bold')
    axes[1].set_xlabel("Besteci", fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    out_path = figures_dir / "composer_prime_profile.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    logger.info("Asal sayı profili kaydedildi: %s", out_path)


def plot_golden_ratio_analysis(df: pd.DataFrame):
    """Bestecilere göre Golden Ratio kullanımı."""
    if df.empty or 'composer' not in df.columns:
        return
    
    figures_dir = CONFIG.paths.figures
    
    # Climax pozisyonu dağılımı
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1) Climax position distribution
    for composer in df['composer'].unique():
        composer_data = df[df['composer'] == composer]['climax_position_ratio']
        axes[0, 0].hist(
            composer_data,
            alpha=0.6,
            label=composer,
            bins=10,
            edgecolor='black'
        )
    
    # Golden Ratio çizgisi (0.618)
    golden_pos = 1.0 / 1.618
    axes[0, 0].axvline(golden_pos, color='red', linestyle='--', linewidth=2, label='Golden Ratio (0.618)')
    axes[0, 0].set_title("Climax Pozisyonu Dağılımı", fontweight='bold')
    axes[0, 0].set_xlabel("Pozisyon Oranı (0-1)", fontweight='bold')
    axes[0, 0].set_ylabel("Frekans", fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2) Fibonacci section ratio
    composer_golden = df.groupby('composer')[
        ['fibonacci_section_ratio', 'golden_ratio_in_durations']
    ].mean()
    
    composer_golden['fibonacci_section_ratio'].plot(
        kind='bar',
        ax=axes[0, 1],
        color='gold',
        edgecolor='black'
    )
    axes[0, 1].set_title("Fibonacci Bölüm Oranı", fontweight='bold')
    axes[0, 1].set_ylabel("Oran (0-1)", fontweight='bold')
    axes[0, 1].set_xlabel("Besteci", fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3) Golden ratio in durations
    composer_golden['golden_ratio_in_durations'].plot(
        kind='bar',
        ax=axes[1, 0],
        color='darkgoldenrod',
        edgecolor='black'
    )
    axes[1, 0].set_title("Nota Sürelerinde Golden Ratio", fontweight='bold')
    axes[1, 0].set_ylabel("Oran (0-1)", fontweight='bold')
    axes[1, 0].set_xlabel("Besteci", fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4) Climax golden distance
    df.boxplot(
        column='climax_golden_distance',
        by='composer',
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("Climax Golden Ratio'ya Uzaklık", fontweight='bold')
    axes[1, 1].set_ylabel("Mesafe", fontweight='bold')
    axes[1, 1].set_xlabel("Besteci", fontweight='bold')
    axes[1, 1].get_figure().suptitle('')  # Remove default title
    
    plt.tight_layout()
    out_path = figures_dir / "golden_ratio_analysis.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    logger.info("Golden Ratio analizi kaydedildi: %s", out_path)


def print_summary_statistics(df: pd.DataFrame):
    """Özet istatistikler."""
    print("\n" + "=" * 80)
    print("MATEMATİKSEL PATTERN ÖZETİ")
    print("=" * 80)
    
    if df.empty:
        print("Veri bulunamadı.")
        return
    
    print(f"\nToplam analiz edilen eser: {len(df)}")
    
    if 'composer' in df.columns:
        print("\n--- Bestecilere Göre Asal Sayı Kullanımı ---")
        prime_summary = df.groupby('composer')[
            ['interval_prime_density', 'duration_prime_ratio', 'num_prime_phrase_lengths']
        ].mean()
        print(prime_summary.round(4).to_string())
        
        print("\n--- Bestecilere Göre Golden Ratio & Fibonacci ---")
        golden_summary = df.groupby('composer')[
            ['climax_golden_distance', 'fibonacci_section_ratio', 'golden_ratio_in_durations']
        ].mean()
        print(golden_summary.round(4).to_string())
        
        print("\n--- Climax Golden Ratio'da Olan Eserler ---")
        golden_climax = df.groupby('composer')['climax_is_golden'].sum()
        print(golden_climax.to_string())
    
    print("\n" + "=" * 80)


def main():
    print("=" * 80)
    print("MATEMATİKSEL PATTERN KEŞİF ARACI")
    print("=" * 80)
    print("\nAnaliz ediliyor: Asal sayılar, Golden Ratio, Fibonacci...")
    
    df = analyze_all_files()
    
    if df.empty:
        print("\n❌ Analiz edilecek veri bulunamadı.")
        return
    
    print(f"\n✓ {len(df)} eser analiz edildi.")
    
    print("\n[1/3] Besteci asal sayı profilleri oluşturuluyor...")
    plot_composer_prime_profile(df)
    
    print("\n[2/3] Golden Ratio analizleri yapılıyor...")
    plot_golden_ratio_analysis(df)
    
    print("\n[3/3] Özet istatistikler...")
    print_summary_statistics(df)
    
    print("\n" + "=" * 80)
    print("✓ Matematiksel pattern analizi tamamlandı!")
    print("Çıktılar:")
    print("  • results/stats/mathematical_patterns.csv")
    print("  • results/figures/composer_prime_profile.png")
    print("  • results/figures/golden_ratio_analysis.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
