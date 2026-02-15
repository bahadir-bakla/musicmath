#!/usr/bin/env python
"""
Besteci / dönem bazli 'imza' metrikleri.

Basit ama yorumlanabilir skorlar uretir:
    - melodik_volatilite
    - motif_tekrar
    - konsonans_denge
    - yapisal_kompleksite

Kullanim:
    python scripts/signature_metrics.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger


logger = get_logger(__name__)


META_COLS = {"filepath", "composer", "era", "form"}


def load_feature_matrix() -> pd.DataFrame:
    feat_path = CONFIG.paths.root / "results" / "stats" / "feature_matrix.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_matrix.csv bulunamadi: {feat_path}")
    df = pd.read_csv(feat_path)
    logger.info("Feature matrix yuklendi: shape=%s", df.shape)
    return df


def _zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sigma


def compute_signature_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Parca bazinda z-score'a dayali imza skorlarini hesapla."""
    required = [
        "pitch_range",
        "pitch_std",
        "repetition_index",
        "consonance_score",
        "dissonance_index",
        "pitch_entropy",
        "interval_entropy",
        "rhythmic_entropy",
        "spectral_entropy",
        "fractal_dimension",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Su kolonlar eksik, imza metrikleri icin gerekli: {missing}")

    z_pitch_range = _zscore(df["pitch_range"])
    z_pitch_std = _zscore(df["pitch_std"])
    z_repetition = _zscore(df["repetition_index"])
    z_consonance = _zscore(df["consonance_score"])
    z_dissonance = _zscore(df["dissonance_index"])

    z_pitch_ent = _zscore(df["pitch_entropy"])
    z_interval_ent = _zscore(df["interval_entropy"])
    z_rhythm_ent = _zscore(df["rhythmic_entropy"])
    z_spectral_ent = _zscore(df["spectral_entropy"])
    z_fractal = _zscore(df["fractal_dimension"])

    # 1) Melodik volatilite: genis alan + yuksek pitch std
    df["sig_melodic_volatility"] = (z_pitch_range + z_pitch_std) / 2.0

    # 2) Motif tekrar: repetition index z-score
    df["sig_motif_repetition"] = z_repetition

    # 3) Konsonans dengesi: konsonans - dissonans
    df["sig_consonance_balance"] = z_consonance - z_dissonance

    # 4) Yapısal kompleksite: entropiler ve fraktal boyut ortalamasi
    df["sig_structural_complexity"] = (
        z_pitch_ent
        + z_interval_ent
        + z_rhythm_ent
        + z_spectral_ent
        + z_fractal
    ) / 5.0

    logger.info("Imza skor kolonlari eklendi.")
    return df


def _group_summary(
    df: pd.DataFrame, by_col: str, score_cols: List[str], min_pieces: int
) -> pd.DataFrame:
    """Besteci / dönem bazinda ortalama skorlar."""
    if by_col not in df.columns:
        raise ValueError(f"'{by_col}' kolonu yok")

    df_valid = df.dropna(subset=[by_col])
    counts = df_valid[by_col].value_counts()
    keep = counts[counts >= min_pieces].index
    df_use = df_valid[df_valid[by_col].isin(keep)].copy()

    summary = (
        df_use.groupby(by_col)[score_cols]
        .agg(["mean", "std", "count"])
        .sort_values((score_cols[0], "mean"))
    )
    summary.insert(0, "n_pieces", counts[summary.index].values)
    return summary


def save_summaries(df: pd.DataFrame, score_cols: List[str]) -> None:
    stats_dir = CONFIG.paths.root / "results" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    comp_summary = _group_summary(
        df, by_col="composer", score_cols=score_cols, min_pieces=5
    )
    era_summary = _group_summary(
        df, by_col="era", score_cols=score_cols, min_pieces=3
    )

    comp_path = stats_dir / "signature_composer_summary.csv"
    era_path = stats_dir / "signature_era_summary.csv"

    comp_summary.to_csv(comp_path)
    era_summary.to_csv(era_path)

    logger.info("Besteci ozetleri kaydedildi: %s", comp_path)
    logger.info("Era ozetleri kaydedildi: %s", era_path)


def plot_distributions(df: pd.DataFrame, score_cols: List[str]) -> None:
    """Era bazinda boxplot'lar ve volatile vs repetition scatter'lari ciz."""
    figures_dir = CONFIG.paths.figures
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set(style="whitegrid")

    if "era" in df.columns:
        for col in score_cols:
            plt.figure(figsize=(8, 6))
            sns.boxplot(
                data=df,
                x="era",
                y=col,
                order=sorted(df["era"].dropna().unique()),
            )
            plt.title(f"{col} – Dönem Bazinda Dagilim")
            plt.xlabel("Era")
            plt.ylabel(col)
            plt.xticks(rotation=20)
            plt.tight_layout()
            out_path = figures_dir / f"{col}_by_era_boxplot.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info("Boxplot kaydedildi: %s", out_path)

    # Melodik volatilite vs motif tekrar, era renklendirmeli scatter
    if {"sig_melodic_volatility", "sig_motif_repetition"}.issubset(df.columns):
        plt.figure(figsize=(8, 6))
        if "era" in df.columns:
            eras = sorted(df["era"].dropna().unique())
            palette = sns.color_palette("tab10", n_colors=len(eras))
            era_to_color = dict(zip(eras, palette))

            for era in eras:
                mask = df["era"] == era
                plt.scatter(
                    df.loc[mask, "sig_melodic_volatility"],
                    df.loc[mask, "sig_motif_repetition"],
                    label=era,
                    alpha=0.7,
                    edgecolors="black",
                    s=70,
                    c=[era_to_color[era]],
                )
            plt.legend(title="Era")
        else:
            plt.scatter(
                df["sig_melodic_volatility"],
                df["sig_motif_repetition"],
                alpha=0.7,
                edgecolors="black",
                s=70,
            )
        plt.xlabel("sig_melodic_volatility")
        plt.ylabel("sig_motif_repetition")
        plt.title("Melodik Volatilite vs Motif Tekrar")
        plt.tight_layout()
        out_path = figures_dir / "signature_volatility_vs_repetition.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info("Scatter plot kaydedildi: %s", out_path)


def main() -> None:
    print("=" * 60)
    print("Besteci / Dönem Imza Metrikleri")
    print("=" * 60)

    df = load_feature_matrix()
    df_sig = compute_signature_scores(df)

    score_cols = [
        "sig_melodic_volatility",
        "sig_motif_repetition",
        "sig_consonance_balance",
        "sig_structural_complexity",
    ]

    # Skorlarin oldugu tabloyu kaydet
    stats_dir = CONFIG.paths.root / "results" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    scores_path = stats_dir / "signature_scores.csv"
    df_sig.to_csv(scores_path, index=False)
    logger.info("Parca bazli imza skorlari kaydedildi: %s", scores_path)

    save_summaries(df_sig, score_cols)
    plot_distributions(df_sig, score_cols)

    print("\nImza metrikleri hesaplandi.")
    print("  - Parca bazli skorlar: results/stats/signature_scores.csv")
    print("  - Besteci ozetleri   : results/stats/signature_composer_summary.csv")
    print("  - Era ozetleri       : results/stats/signature_era_summary.csv")
    print("  - Figürler           : results/figures/*signature*.png")


if __name__ == "__main__":
    main()

