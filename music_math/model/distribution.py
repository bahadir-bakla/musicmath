"""Besteci bazlı istatistiksel dağılım modeli.

FAZ 4'te tarif edilen `ComposerDistributionModel`'in uygulaması.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import json

from music_math.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureStats:
    mean: float
    std: float
    min: float
    max: float
    median: float


ComposerProfile = Dict[str, FeatureStats]

# Mesafe hesaplamasında kullanılmayacak feature'lar (ölçek bağımlı / tek eserle std≈0 patlaması)
DISTANCE_EXCLUDE = frozenset({
    "filepath", "composer", "era", "form",
    "total_notes", "spectral_centroid", "dominant_frequency", "spectral_flatness",
    "pitch_range", "pitch_mean", "pitch_std", "duration_mean", "duration_std",
    "note_density", "duration_variance",
})


@dataclass
class ComposerDistributionModel:
    """Her bestecinin karakteristik istatistiksel profilini tutar."""

    profiles: Dict[str, ComposerProfile] = field(default_factory=dict)

    def fit_composer(self, composer_name: str, features_df: pd.DataFrame) -> ComposerProfile:
        """Tek bir besteci için feature dağılım profilini çıkar."""
        composer_data = features_df[features_df["composer"] == composer_name]
        feature_cols: List[str] = [
            c
            for c in features_df.columns
            if c not in {"filepath", "composer", "era", "form"}
            and np.issubdtype(features_df[c].dtype, np.number)
        ]

        profile: ComposerProfile = {}
        for feat in feature_cols:
            values = composer_data[feat].dropna().values
            if values.size == 0:
                continue
            profile[feat] = FeatureStats(
                mean=float(values.mean()),
                std=float(values.std()),
                min=float(values.min()),
                max=float(values.max()),
                median=float(np.median(values)),
            )

        self.profiles[composer_name] = profile
        return profile

    def fit_all(self, features_df: pd.DataFrame) -> None:
        """Tüm besteciler için profil çıkar."""
        self.profiles.clear()
        for composer in sorted(features_df["composer"].dropna().unique()):
            self.fit_composer(composer, features_df)
        logger.info("%d besteci profili oluşturuldu.", len(self.profiles))

    def musical_distance(
        self,
        composer1: str,
        composer2: str,
        features: Sequence[str] | None = None,
    ) -> float:
        """İki besteci arasındaki normalize ortalama uzaklık."""
        p1 = self.profiles.get(composer1, {})
        p2 = self.profiles.get(composer2, {})
        if not p1 or not p2:
            return float("nan")

        if features is None:
            common = set(p1.keys()) & set(p2.keys()) - DISTANCE_EXCLUDE
            features = sorted(common)

        distances: List[float] = []
        for feat in features:
            if feat not in p1 or feat not in p2:
                continue
            std_pool = (p1[feat].std + p2[feat].std) / 2.0 + 1e-8
            d = abs(p1[feat].mean - p2[feat].mean) / std_pool
            # Tek eser veya çok büyük farklarda patlamayı sınırla
            distances.append(min(d, 10.0))

        return float(np.mean(distances)) if distances else float("nan")

    def closest_composer(
        self,
        feature_vector: Mapping[str, float],
        feature_names: Iterable[str],
    ) -> tuple[str, Dict[str, float]]:
        """Verilen feature vektörüne en yakın besteciyi bul."""
        distances: Dict[str, float] = {}
        for composer, profile in self.profiles.items():
            dists = []
            for feat in feature_names:
                if feat not in profile or feat not in feature_vector:
                    continue
                std = profile[feat].std + 1e-8
                d = abs(feature_vector[feat] - profile[feat].mean) / std
                dists.append(d)
            if dists:
                distances[composer] = float(np.mean(dists))
        if not distances:
            raise ValueError("Hiçbir besteci için ortak feature bulunamadı.")
        best = min(distances, key=distances.get)
        return best, distances

    def save(self, filepath: str | Path) -> None:
        """Profilleri JSON olarak kaydet."""
        path = Path(filepath)
        serializable = {
            composer: {
                feat: stats.__dict__ for feat, stats in profile.items()
            }
            for composer, profile in self.profiles.items()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        logger.info("ComposerDistributionModel kaydedildi: %s", path)

    def load(self, filepath: str | Path) -> None:
        """JSON dosyadan profilleri yükle."""
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        profiles: Dict[str, ComposerProfile] = {}
        for composer, profile_data in data.items():
            profiles[composer] = {
                feat: FeatureStats(**stats_dict) for feat, stats_dict in profile_data.items()
            }
        self.profiles = profiles
        logger.info("ComposerDistributionModel yüklendi: %s", path)


def composer_distance_matrix(model: ComposerDistributionModel) -> pd.DataFrame:
    """Tüm besteciler arasındaki mesafe matrisini DataFrame olarak döndür."""
    composers = sorted(model.profiles.keys())
    n = len(composers)
    mat = np.zeros((n, n), dtype=float)
    for i, c1 in enumerate(composers):
        for j, c2 in enumerate(composers):
            if i == j:
                mat[i, j] = 0.0
            else:
                mat[i, j] = model.musical_distance(c1, c2)
    return pd.DataFrame(mat, index=composers, columns=composers)


__all__ = ["FeatureStats", "ComposerDistributionModel", "composer_distance_matrix", "DISTANCE_EXCLUDE"]

