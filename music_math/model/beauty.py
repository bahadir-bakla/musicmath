"""Teorik 'müzikal etki / güzellik' fonksiyonu.

FAZ 4 ve 6'da insan deneyi verisi ile kalibre edilmek üzere tasarlanmıştır.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class BeautyConfig:
    """Güzellik fonksiyonu için ayarlanabilir parametreler."""

    entropy_optimal_low: float = 2.0
    entropy_optimal_high: float = 2.8
    target_consonance: float = 0.65
    target_repetition: float = 0.4
    target_fractal_dim: float = 1.5

    w_entropy: float = 0.35
    w_consonance: float = 0.25
    w_repetition: float = 0.20
    w_fractal: float = 0.20


def musical_impact_score(features: Mapping[str, float], cfg: BeautyConfig | None = None) -> float:
    """
    Teorik müzikal etki/güzellik skoru (0–1 arası).

    Bu fonksiyon, FAZ 6'da toplanacak insan deneyi verisine göre
    güncellenecektir; şimdilik mantıklı varsayımlara dayanır.
    """
    if cfg is None:
        cfg = BeautyConfig()

    entropy = float(features.get("pitch_entropy", 0.0))
    consonance = float(features.get("consonance_score", 0.0))
    repetition = float(features.get("repetition_index", 0.0))
    fractal_dim = float(features.get("fractal_dimension", cfg.target_fractal_dim))

    # Entropi skoru: optimal banda yakınlık
    if cfg.entropy_optimal_low <= entropy <= cfg.entropy_optimal_high:
        entropy_score = 1.0
    else:
        deviation = min(
            abs(entropy - cfg.entropy_optimal_low),
            abs(entropy - cfg.entropy_optimal_high),
        )
        entropy_score = max(0.0, 1.0 - deviation)

    # Konsonans skoru: orta konsonans optimum
    consonance_score = 1.0 - abs(consonance - cfg.target_consonance)

    # Tekrar skoru: orta tekrar en iyi
    repetition_score = 1.0 - abs(repetition - cfg.target_repetition)

    # Fraktal skor: ~1.5 civarı optimum
    fractal_score = 1.0 - abs(fractal_dim - cfg.target_fractal_dim)

    score = (
        cfg.w_entropy * entropy_score
        + cfg.w_consonance * consonance_score
        + cfg.w_repetition * repetition_score
        + cfg.w_fractal * fractal_score
    )

    # 0–1 aralığına kırp
    return float(max(0.0, min(1.0, score)))


__all__ = ["BeautyConfig", "musical_impact_score"]

