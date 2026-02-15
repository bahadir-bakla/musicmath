#!/usr/bin/env python
"""
Nota sayisal oruntu ve aesthetic index metriklerini era bazinda boxplot ile gorsellestirir.

Kullanim:
    python scripts/plot_numeric_patterns_by_era.py
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from music_math.core.config import CONFIG


METRICS = [
    "prime_sum_ratio",
    "phi_density_ratio",
    "interval_self_similarity",
    "rw_corr_z",
    "aesthetic_index",
]
METRIC_LABELS = {
    "prime_sum_ratio": "Kümülatif toplamda asal oranı",
    "phi_density_ratio": "Altın oran bölgelerinde yoğunluk",
    "interval_self_similarity": "Aralık öz-benzerlik",
    "rw_corr_z": "Rastgele yürüyüşten sapma (z)",
    "aesthetic_index": "Estetik indeks (0–1)",
}


def main() -> None:
    stats_dir = CONFIG.paths.root / "results" / "stats"
    figures_dir = CONFIG.paths.figures
    figures_dir.mkdir(parents=True, exist_ok=True)

    # aesthetic_index.csv hem temel metrikleri hem aesthetic_index icerir
    src = stats_dir / "aesthetic_index.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"Önce scripts/note_numeric_patterns.py ve scripts/aesthetic_index.py çalıştırın: {src}"
        )
    df = pd.read_csv(src)
    if "era" not in df.columns or df.empty:
        print("'era' kolonu veya veri yok.")
        return

    era_order = ["Baroque", "Classical", "Romantic", "Late Romantic", "Other"]
    era_order = [e for e in era_order if e in df["era"].unique()]
    if not era_order:
        era_order = sorted(df["era"].dropna().unique().tolist())

    sns.set_style("whitegrid")
    for col in METRICS:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=df,
            x="era",
            y=col,
            order=era_order,
            palette="Set2",
            ax=ax,
        )
        ax.set_title(METRIC_LABELS.get(col, col), fontweight="bold")
        ax.set_xlabel("Dönem")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        out = figures_dir / f"numeric_pattern_{col}_by_era.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Kaydedildi: {out}")

    print("\nTüm boxplot'lar results/figures/ altında.")


if __name__ == "__main__":
    main()
