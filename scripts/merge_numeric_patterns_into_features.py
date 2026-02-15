#!/usr/bin/env python
"""
feature_matrix.csv ile note_numeric_patterns + aesthetic_index metriklerini
filepath uzerinden birlestirir. Cikti: feature_matrix_with_numeric_patterns.csv
Boylece siniflandirma / imza analizlerinde bu metrikler de kullanilabilir.

Kullanim:
    python scripts/merge_numeric_patterns_into_features.py
"""

from __future__ import annotations

import pandas as pd

from music_math.core.config import CONFIG


EXTRA_COLS = [
    "prime_sum_ratio",
    "phi_density_ratio",
    "interval_self_similarity",
    "rw_corr_z",
    "rw_corr_mean",
    "aesthetic_index",
    "aesthetic_z_raw",
]


def main() -> None:
    stats_dir = CONFIG.paths.root / "results" / "stats"
    feat_path = stats_dir / "feature_matrix.csv"
    aesthetic_path = stats_dir / "aesthetic_index.csv"

    if not feat_path.exists():
        raise FileNotFoundError(f"feature_matrix.csv bulunamadi: {feat_path}")
    if not aesthetic_path.exists():
        raise FileNotFoundError(
            f"aesthetic_index.csv bulunamadi; once note_numeric_patterns + aesthetic_index calistirin: {aesthetic_path}"
        )

    df_feat = pd.read_csv(feat_path)
    df_extra = pd.read_csv(aesthetic_path)

    # filepath uzerinden birlestir; sadece ek kolonlari al
    use_cols = ["filepath"] + [c for c in EXTRA_COLS if c in df_extra.columns]
    df_extra = df_extra[use_cols]
    df_merged = df_feat.merge(df_extra, on="filepath", how="left")

    out_path = stats_dir / "feature_matrix_with_numeric_patterns.csv"
    df_merged.to_csv(out_path, index=False)
    print(f"Birlestirildi: {len(df_merged)} satir -> {out_path}")
    print(f"Eklenen kolonlar: {[c for c in use_cols if c != 'filepath']}")


if __name__ == "__main__":
    main()
