"""İnsan deneyi sonuç analizi ve güzellik fonksiyonu kalibrasyonu."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from music_math.model.beauty import BeautyConfig


def analyze_experiment_results(results_csv: str) -> pd.DataFrame:
    """Koşul bazında temel istatistikleri ve basit testleri hesapla."""
    df = pd.read_csv(results_csv)

    # 1. Koşul bazında ortalama beğeni
    summary = (
        df.groupby("condition")["likability"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # 2. ANOVA
    groups = [grp["likability"].values for _, grp in df.groupby("condition")]
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        summary.attrs["anova_F"] = float(f_stat)
        summary.attrs["anova_p"] = float(p_value)

    return df


def feature_preference_correlations(
    results_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_names: Iterable[str],
) -> Dict[str, Tuple[float, float]]:
    """
    Matematiksel feature'lar ile beğeni puanı arasındaki korelasyonları hesapla.
    """
    merged = results_df.merge(features_df, on="stimulus_id")

    correlations: Dict[str, Tuple[float, float]] = {}
    for feat in feature_names:
        if feat not in merged.columns:
            continue
        r, p = stats.pearsonr(merged[feat], merged["likability"])
        correlations[feat] = (float(r), float(p))

    return correlations


def update_beauty_config_from_correlations(
    base_cfg: BeautyConfig,
    correlations: Dict[str, Tuple[float, float]],
) -> BeautyConfig:
    """
    Korelasyonlara göre ağırlıkları yeniden ölçekleyen basit bir heuristik.
    """
    # Önem sırası: |r| ve anlamlılık (p < 0.05)
    scores = {}
    for feat, (r, p) in correlations.items():
        if p < 0.05:
            scores[feat] = abs(r)

    total = sum(scores.values()) or 1.0
    norm_scores = {k: v / total for k, v in scores.items()}

    cfg = BeautyConfig(**base_cfg.__dict__)
    if "pitch_entropy" in norm_scores:
        cfg.w_entropy = norm_scores["pitch_entropy"]
    if "consonance_score" in norm_scores:
        cfg.w_consonance = norm_scores["consonance_score"]
    if "repetition_index" in norm_scores:
        cfg.w_repetition = norm_scores["repetition_index"]
    if "fractal_dimension" in norm_scores:
        cfg.w_fractal = norm_scores["fractal_dimension"]

    # Yeniden normalize et
    s = cfg.w_entropy + cfg.w_consonance + cfg.w_repetition + cfg.w_fractal
    if s > 0:
        cfg.w_entropy /= s
        cfg.w_consonance /= s
        cfg.w_repetition /= s
        cfg.w_fractal /= s

    return cfg


__all__ = [
    "analyze_experiment_results",
    "feature_preference_correlations",
    "update_beauty_config_from_correlations",
]

