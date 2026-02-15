#!/usr/bin/env python
"""
Eser bazinda "aesthetic index" hesaplama araci.

Kaynak metrikler:
    - note_numeric_patterns.csv icinden:
        * prime_sum_ratio
        * phi_density_ratio
        * interval_self_similarity
        * rw_corr_z (interval nonrandomness)

Adimlar:
    1) Bu metrikleri z-score'la normalize et.
    2) Basit bir lineer kombinasyonla aesthetic_index tanimla.
    3) Sonuclari results/stats/aesthetic_index.csv olarak kaydet.

Kullanim:
    python scripts/aesthetic_index.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger


logger = get_logger(__name__)


METRIC_COLS = [
    "prime_sum_ratio",
    "phi_density_ratio",
    "interval_self_similarity",
    "rw_corr_z",
]


def _zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sigma


def compute_aesthetic_index(df: pd.DataFrame) -> pd.DataFrame:
    """Verilen DataFrame'e z-skorlar ve aesthetic_index kolonu ekle.

    Not:
        aesthetic_index once z-skorlu lineer kombinasyon olarak
        hesaplanir (aesthetic_z_raw), sonra tum eserler uzerinde
        min-max normalizasyon ile [0, 1] araligina olceklenir.
    """
    df = df.copy()

    for col in METRIC_COLS:
        if col not in df.columns:
            logger.warning("Kolon bulunamadi, aesthetic index disinda kalacak: %s", col)
            continue
        z_col = f"z_{col}"
        df[z_col] = _zscore(df[col])

    # Eksik kolonlar 0 kabul edilerek aesthetic index icin kullanilan
    # z-skorlarini cek
    z_prime = df.get("z_prime_sum_ratio", 0.0)
    z_phi = df.get("z_phi_density_ratio", 0.0)
    z_rw = df.get("z_rw_corr_z", 0.0)

    # Agirliklar: prime & phi hafif, esas olarak nonrandomness
    df["aesthetic_z_raw"] = 0.25 * z_prime + 0.25 * z_phi + 0.5 * z_rw

    # Min-max normalizasyon ile [0, 1] araligina cek
    z_vals = df["aesthetic_z_raw"]
    z_min = float(z_vals.min())
    z_max = float(z_vals.max())
    if z_max > z_min:
        df["aesthetic_index"] = (z_vals - z_min) / (z_max - z_min)
    else:
        df["aesthetic_index"] = 0.5  # tum degerler ayniysa ortada kal

    return df


def main() -> None:
    stats_dir = CONFIG.paths.root / "results" / "stats"
    src_csv = stats_dir / "note_numeric_patterns.csv"
    if not src_csv.exists():
        raise FileNotFoundError(
            f"note_numeric_patterns.csv bulunamadi, once scripts/note_numeric_patterns.py calistirin: {src_csv}"
        )

    df = pd.read_csv(src_csv)
    if df.empty:
        print("note_numeric_patterns.csv bos, aesthetic index hesaplanamadi.")
        return

    df_out = compute_aesthetic_index(df)

    out_csv = stats_dir / "aesthetic_index.csv"
    df_out.to_csv(out_csv, index=False)

    print("=" * 60)
    print("Aesthetic Index Hesaplandi")
    print("=" * 60)
    print(f"Eser sayisi: {len(df_out)}")
    print(f"Cikti: {out_csv}")

    if "era" in df_out.columns:
        print("\nEra bazinda ortalama aesthetic_index (0-1 olceginde):")
        print(
            df_out.groupby("era")["aesthetic_index"]
                .mean()
                .sort_values()
        )


if __name__ == "__main__":
    main()

