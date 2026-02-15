#!/usr/bin/env python
"""
İstatistiksel hipotez testleri: t-test, ANOVA, chi-square.

Hipotezler:
  H1: Dönemsel evrim – Barok → Romantik asal sayı yoğunluğu farkı (t-test / ANOVA)
  H2: Mozart'ın matematiksel tasarımı – Fibonacci oranı rastgeleden farklı mı? (chi-square)
  H3: Besteci grupları feature uzayında ayrışıyor mu? (ANOVA, besteci bazında)

Kullanım:
    python scripts/statistical_tests.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger

logger = get_logger(__name__)


def load_feature_matrix() -> pd.DataFrame:
    p = CONFIG.paths.root / "results" / "stats" / "feature_matrix.csv"
    if not p.exists():
        raise FileNotFoundError(f"feature_matrix.csv bulunamadı: {p}")
    return pd.read_csv(p)


def load_mathematical_patterns() -> pd.DataFrame | None:
    p = CONFIG.paths.stats / "mathematical_patterns.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def anova_by_era(df: pd.DataFrame, feature: str) -> None:
    """H3: Dönem grupları arasında feature açısından anlamlı fark var mı? (ANOVA)"""
    if "era" not in df.columns or feature not in df.columns:
        return
    groups = [df.loc[df["era"] == era, feature].dropna().values for era in df["era"].dropna().unique()]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        print(f"  ANOVA ({feature}): Yeterli grup yok, atlandı.")
        return
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"  ANOVA ({feature}): F={f_stat:.4f}, p={p_val:.4f}  {'*' if p_val < 0.05 else ''}")


def ttest_two_eras(df: pd.DataFrame, feature: str, era1: str, era2: str) -> None:
    """İki dönem arasında t-test (örn. Barok vs Romantik)."""
    if feature not in df.columns or "era" not in df.columns:
        return
    a = df.loc[df["era"] == era1, feature].dropna()
    b = df.loc[df["era"] == era2, feature].dropna()
    if len(a) < 2 or len(b) < 2:
        print(f"  t-test ({era1} vs {era2}, {feature}): Yeterli veri yok.")
        return
    t_stat, p_val = stats.ttest_ind(a, b)
    print(f"  t-test ({era1} vs {era2}, {feature}): t={t_stat:.4f}, p={p_val:.4f}  {'*' if p_val < 0.05 else ''}")


def chisquare_fibonacci_vs_random(df_patterns: pd.DataFrame) -> None:
    """H2: Mozart'ın fibonacci_section_ratio'su rastgele dağılımdan farklı mı? (basit chi-square)"""
    if df_patterns is None or "composer" not in df_patterns.columns or "fibonacci_section_ratio" not in df_patterns.columns:
        print("  Chi-square (Fibonacci): mathematical_patterns.csv yok, önce scripts/mathematical_patterns.py çalıştırın.")
        return
    mozart = df_patterns.loc[df_patterns["composer"] == "Mozart", "fibonacci_section_ratio"].dropna()
    if len(mozart) < 2:
        print("  Chi-square (Fibonacci): Mozart için yeterli veri yok.")
        return
    # Rastgele beklenti: oran ~0.5 (yarı yarıya). Gözlenen ortalama ile karşılaştır.
    observed_mean = mozart.mean()
    # Basit test: ortalama 0.5'ten anlamlı farklı mı? (one-sample t-test)
    t_stat, p_val = stats.ttest_1samp(mozart, 0.5)
    print(f"  Mozart Fibonacci section ratio: mean={observed_mean:.3f}, vs 0.5 → t={t_stat:.4f}, p={p_val:.4f}  {'*' if p_val < 0.05 else ''}")


def anova_prime_by_composer(df_patterns: pd.DataFrame) -> None:
    """H3: Bestecilere göre asal sayı yoğunluğu farklı mı? (ANOVA)"""
    if df_patterns is None or "interval_prime_density" not in df_patterns.columns:
        return
    if "composer" not in df_patterns.columns:
        return
    groups = [
        df_patterns.loc[df_patterns["composer"] == c, "interval_prime_density"].dropna().values
        for c in df_patterns["composer"].dropna().unique()
    ]
    groups = [g for g in groups if len(g) >= 1]
    if len(groups) < 2:
        print("  ANOVA (interval_prime_density by composer): Yeterli besteci yok.")
        return
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"  ANOVA (interval_prime_density by composer): F={f_stat:.4f}, p={p_val:.4f}  {'*' if p_val < 0.05 else ''}")


def main():
    print("=" * 60)
    print("İstatistiksel Hipotez Testleri")
    print("=" * 60)

    # Feature matrix
    df_feat = load_feature_matrix()
    print("\n--- Feature matrix üzerinde (dönem / besteci ayrışması) ---")
    for feat in ["pitch_entropy", "consonance_score", "interval_entropy", "rhythmic_entropy"]:
        if feat in df_feat.columns:
            anova_by_era(df_feat, feat)

    # İki dönem karşılaştırması (varsa)
    if "era" in df_feat.columns:
        eras = df_feat["era"].dropna().unique().tolist()
        if "Baroque" in eras and "Romantic" in eras and "pitch_entropy" in df_feat.columns:
            print("\n--- Barok vs Romantik (pitch_entropy) ---")
            ttest_two_eras(df_feat, "pitch_entropy", "Baroque", "Romantic")

    # Matematiksel pattern sonuçları (mathematical_patterns.py sonrası)
    df_pat = load_mathematical_patterns()
    print("\n--- Matematiksel pattern testleri (mathematical_patterns.csv gerekli) ---")
    chisquare_fibonacci_vs_random(df_pat)
    anova_prime_by_composer(df_pat)

    print("\n" + "=" * 60)
    print("(*) p < 0.05 anlamlı fark gösterir.")
    print("=" * 60)


if __name__ == "__main__":
    main()
