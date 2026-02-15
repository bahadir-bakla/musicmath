"""Generatif üretim için matematiksel kısıtlar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np


STYLE_PRESETS = {
    "baroque": {
        "entropy_range": (1.8, 2.4),
        "consonance_min": 0.7,
        "step_ratio_min": 0.6,
        "repetition_target": 0.5,
        "fractal_dim_range": (1.2, 1.6),
    },
    "classical": {
        "entropy_range": (2.0, 2.6),
        "consonance_min": 0.6,
        "step_ratio_min": 0.5,
        "repetition_target": 0.4,
        "fractal_dim_range": (1.3, 1.6),
    },
    "romantic": {
        "entropy_range": (2.3, 3.0),
        "consonance_min": 0.4,
        "step_ratio_min": 0.4,
        "repetition_target": 0.3,
        "fractal_dim_range": (1.4, 1.8),
    },
    "late_romantic": {
        "entropy_range": (2.5, 3.2),
        "consonance_min": 0.3,
        "step_ratio_min": 0.35,
        "repetition_target": 0.25,
        "fractal_dim_range": (1.5, 2.0),
    },
}


@dataclass
class MusicalConstraints:
    """Müzikal üretim için stil bazlı kısıtlar."""

    style: str = "classical"

    def __post_init__(self) -> None:
        self.params = STYLE_PRESETS.get(self.style, STYLE_PRESETS["classical"])

    def is_valid(self, features: Mapping[str, float]) -> Tuple[bool, list[bool]]:
        """Üretilen bir pasajın kısıtları karşılayıp karşılamadığını kontrol et."""
        checks: list[bool] = []

        e = float(features.get("pitch_entropy", 0.0))
        low, high = self.params["entropy_range"]
        checks.append(low <= e <= high)

        c = float(features.get("consonance_score", 0.0))
        checks.append(c >= self.params["consonance_min"])

        sr = float(features.get("step_ratio", 0.0))
        checks.append(sr >= self.params["step_ratio_min"])

        # Diğer kısıtlar ileride eklenebilir

        return all(checks), checks

    def distance_from_target(self, features: Mapping[str, float]) -> float:
        """Hedef kısıtlardan ortalama uzaklık (daha düşük = daha iyi)."""
        distances = []

        e = float(features.get("pitch_entropy", 0.0))
        low, high = self.params["entropy_range"]
        target_e = (low + high) / 2.0
        distances.append(abs(e - target_e))

        rep = float(features.get("repetition_index", 0.0))
        target_rep = self.params["repetition_target"]
        distances.append(abs(rep - target_rep))

        fd = float(features.get("fractal_dimension", 1.5))
        fd_low, fd_high = self.params["fractal_dim_range"]
        target_fd = (fd_low + fd_high) / 2.0
        distances.append(abs(fd - target_fd))

        return float(np.mean(distances)) if distances else 0.0


__all__ = ["MusicalConstraints", "STYLE_PRESETS"]

