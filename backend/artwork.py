"""
Altyapi: Muzik -> gorsel sanat (renk paleti, Kalman filtre, cizim pipeline).
Hedef: Duygu uyandiran, modern sanat tarzi – sadece renk ve gecisler, somut bicim yok.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from enum import Enum


class RenderMode(str, Enum):
    """Muzikten goruntu uretme modlari (ileride implemente edilecek)."""
    KALMAN = "kalman"       # Zaman serisi -> Kalman filtre -> duzgunlesmis sinyal -> renk
    SPECTRAL = "spectral"   # Spektral ozellikler -> renk kanallari
    WAVEFORM = "waveform"   # Dalga formu yogunluk -> ton
    PHI_ARC = "phi_arc"     # Altin oran noktalarinda renk gecisleri
    FRACTAL = "fractal"     # Fractal boyut / self-similarity -> doku


# Duygu / estetik odakli renk paletleri (modern sanat, soyut)
PALETTES: List[Dict[str, Any]] = [
    {
        "id": "emotional_dark",
        "name": "Emotional Dark",
        "description": "Koyu tonlar, derin gecisler – huzun / yogunluk",
        "colors": ["#0a0a0f", "#1a1a2e", "#16213e", "#0f3460", "#e94560"],
    },
    {
        "id": "golden_hour",
        "name": "Golden Hour",
        "description": "Altin, amber, kirmizi – sicaklik / cosku",
        "colors": ["#1a0a00", "#4a2c00", "#c45c00", "#ff9a00", "#ffcc70"],
    },
    {
        "id": "baroque_velvet",
        "name": "Baroque Velvet",
        "description": "Kadife koyu mor, altin – donem hissi",
        "colors": ["#0d0221", "#2d1b4e", "#6b2d5c", "#b8860b", "#daa520"],
    },
    {
        "id": "romantic_cloud",
        "name": "Romantic Cloud",
        "description": "Pastel bulut – ask / hayal",
        "colors": ["#1e1e2f", "#3d3d5c", "#9b87b5", "#e8b4b8", "#f5e6e8"],
    },
    {
        "id": "spectral_abstract",
        "name": "Spectral Abstract",
        "description": "Spektrum – frekanslara gore renk (soyut)",
        "colors": ["#0b0b12", "#2e1065", "#4c1d95", "#06b6d4", "#22d3ee"],
    },
    {
        "id": "minimal_contrast",
        "name": "Minimal Contrast",
        "description": "Cok az renk, yuksek kontrast – modern minimal",
        "colors": ["#0c0c0c", "#171717", "#404040", "#a3a3a3", "#fafafa"],
    },
]


def get_palettes() -> List[Dict[str, Any]]:
    return PALETTES


def get_render_modes() -> List[Dict[str, Any]]:
    return [
        {"id": m.value, "name": m.name.replace("_", " ").title(), "description": _mode_description(m)}
        for m in RenderMode
    ]


def _mode_description(m: RenderMode) -> str:
    d = {
        RenderMode.KALMAN: "Smoothed time series via Kalman filter → color gradient over time.",
        RenderMode.SPECTRAL: "Spectral features mapped to RGB channels.",
        RenderMode.WAVEFORM: "Waveform intensity → luminance and hue.",
        RenderMode.PHI_ARC: "Golden-ratio time points drive color transitions.",
        RenderMode.FRACTAL: "Fractal dimension and self-similarity → texture and density.",
    }
    return d.get(m, "")


def artwork_config() -> Dict[str, Any]:
    """Frontend'in cizecegi canvas / render icin varsayilan konfig."""
    return {
        "width": 1200,
        "height": 800,
        "defaultPaletteId": "emotional_dark",
        "defaultRenderMode": RenderMode.KALMAN.value,
        "palettes": get_palettes(),
        "renderModes": get_render_modes(),
    }


# ---------------------------------------------------------------------------
# Pipeline: Muzik -> zaman serisi -> Kalman -> renk -> canvas (ileride)
# ---------------------------------------------------------------------------

def kalman_smooth_1d(
    series: List[float],
    process_var: float = 0.01,
    measure_var: float = 0.1,
) -> List[float]:
    """
    1D Kalman filter: zaman serisini duzgunlestirir (pitch, energy, vb.).
    Gercek implementasyon algorithms.kalman; burada uyumluluk icin tutuldu.
    """
    try:
        from algorithms import kalman
        return kalman(series, process_var=process_var, measure_var=measure_var)
    except ImportError:
        if not series:
            return []
        x, p = float(series[0]), 1.0
        out = [x]
        for i in range(1, len(series)):
            x_pred, p_pred = x, p + process_var
            k = p_pred / (p_pred + measure_var)
            x = x_pred + k * (float(series[i]) - x_pred)
            p = (1 - k) * p_pred
            out.append(x)
        return out


def series_to_color_weights(series: List[float], palette_len: int = 5) -> List[float]:
    """
    [0,1] normalize edilmis seriyi palet indeksine map eder (0 .. palette_len-1).
    Ileride series_to_colors ile hex renklerine cevrilecek.
    """
    if not series:
        return []
    mn, mx = min(series), max(series)
    span = (mx - mn) or 1.0
    return [((v - mn) / span) * (palette_len - 1) for v in series]
