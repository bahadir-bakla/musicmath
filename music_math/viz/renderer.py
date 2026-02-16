"""
Music-to-Art Renderer: muzik ozelliklerinden gorsel sanat uretimi.

5 render modu:
  1. KALMAN   - Pitch zaman serisi -> Kalman duzgunlestirme -> renk gradyani
  2. SPECTRAL - Frekans/harmonik ozellikler -> RGB kanallari
  3. WAVEFORM - Nota yogunluk/enerji -> dalga formu cizimi
  4. PHI_ARC  - Altin oran noktalari -> renk gecisleri, yayli cizim
  5. FRACTAL  - Fraktal benzerlik / self-similarity -> doku ve yogunluk

Her mod: nota listesi + palet + boyut -> PIL Image (PNG).
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import math
import io

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from music_math.core.types import NoteEvent
from music_math.features.harmony import CONSONANCE_MAP


PHI = (1 + math.sqrt(5)) / 2

DEFAULT_PALETTES = {
    "emotional_dark": ["#0a0a0f", "#1a1a2e", "#16213e", "#0f3460", "#e94560"],
    "golden_hour": ["#1a0a00", "#4a2c00", "#c45c00", "#ff9a00", "#ffcc70"],
    "baroque_velvet": ["#0d0221", "#2d1b4e", "#6b2d5c", "#b8860b", "#daa520"],
    "romantic_cloud": ["#1e1e2f", "#3d3d5c", "#9b87b5", "#e8b4b8", "#f5e6e8"],
    "spectral_abstract": ["#0b0b12", "#2e1065", "#4c1d95", "#06b6d4", "#22d3ee"],
    "minimal_contrast": ["#0c0c0c", "#171717", "#404040", "#a3a3a3", "#fafafa"],
}


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _value_to_palette_color(value: float, palette_rgb: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    n = len(palette_rgb) - 1
    if n <= 0:
        return palette_rgb[0] if palette_rgb else (128, 128, 128)
    value = max(0.0, min(1.0, value))
    idx = value * n
    lo = min(int(idx), n)
    hi = min(lo + 1, n)
    frac = idx - lo
    return _lerp_color(palette_rgb[lo], palette_rgb[hi], frac)


def _kalman_1d(series: List[float], pv: float = 0.02, mv: float = 0.08) -> List[float]:
    if not series:
        return []
    x, p = series[0], 1.0
    out = [x]
    for i in range(1, len(series)):
        x_pred, p_pred = x, p + pv
        k = p_pred / (p_pred + mv)
        x = x_pred + k * (series[i] - x_pred)
        p = (1 - k) * p_pred
        out.append(x)
    return out


def _normalize(series: List[float]) -> List[float]:
    if not series:
        return []
    mn, mx = min(series), max(series)
    span = mx - mn
    if span == 0:
        return [0.5] * len(series)
    return [(v - mn) / span for v in series]


@dataclass
class RenderConfig:
    width: int = 1200
    height: int = 800
    palette_id: str = "emotional_dark"
    blur_radius: float = 2.0
    line_width: int = 2
    show_golden_markers: bool = True


def render_kalman(
    notes: List[NoteEvent],
    config: RenderConfig,
    palette_hex: Optional[List[str]] = None,
) -> "Image.Image":
    if not HAS_PIL:
        raise ImportError("Pillow required: pip install Pillow")

    colors_hex = palette_hex or DEFAULT_PALETTES.get(config.palette_id, DEFAULT_PALETTES["emotional_dark"])
    palette_rgb = [_hex_to_rgb(c) for c in colors_hex]
    w, h = config.width, config.height

    pitches = [float(n.pitch) for n in notes]
    smoothed = _kalman_1d(pitches)
    norm_smooth = _normalize(smoothed)

    velocities = []
    for i, n in enumerate(notes):
        if i == 0:
            velocities.append(0.5)
        else:
            velocities.append(min(1.0, abs(n.pitch - notes[i - 1].pitch) / 12.0))

    img = Image.new("RGB", (w, h), palette_rgb[0])
    draw = ImageDraw.Draw(img)

    n_notes = len(notes)
    if n_notes < 2:
        return img

    for i in range(n_notes):
        x = int(i / (n_notes - 1) * (w - 1))

        color = _value_to_palette_color(norm_smooth[i], palette_rgb)
        brightness = 0.3 + 0.7 * velocities[i]
        color = tuple(int(c * brightness) for c in color)

        y_center = int((1.0 - norm_smooth[i]) * h * 0.8 + h * 0.1)
        bar_h = int(h * 0.05 + h * 0.15 * velocities[i])

        draw.rectangle(
            [x, y_center - bar_h // 2, x + max(1, w // n_notes), y_center + bar_h // 2],
            fill=color,
        )

    points = []
    for i in range(n_notes):
        x = int(i / (n_notes - 1) * (w - 1))
        y = int((1.0 - norm_smooth[i]) * h * 0.8 + h * 0.1)
        points.append((x, y))

    if len(points) >= 2:
        accent = palette_rgb[-1]
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=accent, width=config.line_width)

    if config.show_golden_markers:
        golden_positions = [1.0 / PHI, 1.0 / PHI**2, 1.0 - 1.0 / PHI]
        for gp in golden_positions:
            gx = int(gp * (w - 1))
            marker_color = _lerp_color(palette_rgb[-1], (255, 215, 0), 0.5)
            draw.line([(gx, 0), (gx, h)], fill=marker_color, width=1)

    if config.blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=config.blur_radius))

    return img


def render_spectral(
    notes: List[NoteEvent],
    config: RenderConfig,
    palette_hex: Optional[List[str]] = None,
) -> "Image.Image":
    if not HAS_PIL:
        raise ImportError("Pillow required")

    w, h = config.width, config.height
    colors_hex = palette_hex or DEFAULT_PALETTES.get(config.palette_id, DEFAULT_PALETTES["spectral_abstract"])
    palette_rgb = [_hex_to_rgb(c) for c in colors_hex]

    img = Image.new("RGB", (w, h), palette_rgb[0])
    draw = ImageDraw.Draw(img)

    n_notes = len(notes)
    if n_notes < 2:
        return img

    pitches_norm = _normalize([float(n.pitch) for n in notes])
    durations_norm = _normalize([n.duration for n in notes])
    intervals_norm = _normalize(
        [0.0] + [abs(notes[i].pitch - notes[i - 1].pitch) / 12.0 for i in range(1, n_notes)]
    )

    for i in range(n_notes):
        x = int(i / (n_notes - 1) * (w - 1))
        col_w = max(2, w // n_notes)

        r_val = pitches_norm[i]
        g_val = durations_norm[i]
        b_val = intervals_norm[i]

        base = _value_to_palette_color(r_val, palette_rgb)
        r = int(base[0] * 0.5 + r_val * 127)
        g = int(base[1] * 0.5 + g_val * 127)
        b = int(base[2] * 0.5 + b_val * 127)
        color = (min(255, r), min(255, g), min(255, b))

        for row in range(12):
            pc = notes[i].pitch % 12
            row_h = h // 12
            y = row * row_h
            intensity = 1.0 if row == pc else 0.15
            rc = tuple(int(c * intensity) for c in color)
            draw.rectangle([x, y, x + col_w, y + row_h], fill=rc)

    if config.blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=config.blur_radius * 0.5))

    return img


def render_waveform(
    notes: List[NoteEvent],
    config: RenderConfig,
    palette_hex: Optional[List[str]] = None,
) -> "Image.Image":
    if not HAS_PIL:
        raise ImportError("Pillow required")

    w, h = config.width, config.height
    colors_hex = palette_hex or DEFAULT_PALETTES.get(config.palette_id, DEFAULT_PALETTES["emotional_dark"])
    palette_rgb = [_hex_to_rgb(c) for c in colors_hex]

    img = Image.new("RGB", (w, h), palette_rgb[0])
    draw = ImageDraw.Draw(img)

    n_notes = len(notes)
    if n_notes < 2:
        return img

    pitches_norm = _normalize([float(n.pitch) for n in notes])
    mid_y = h // 2

    resolution = w
    wave = np.zeros(resolution)
    for i, n in enumerate(notes):
        if n_notes <= 1:
            continue
        start_x = int(i / (n_notes - 1) * (resolution - 1))
        dur_px = max(1, int(n.duration / 4.0 * resolution / n_notes))
        freq = 0.5 + (n.pitch % 12) / 12.0 * 4
        amplitude = 0.2 + pitches_norm[i] * 0.8
        for px in range(start_x, min(start_x + dur_px, resolution)):
            t = (px - start_x) / max(dur_px, 1)
            wave[px] += amplitude * math.sin(2 * math.pi * freq * t) * math.exp(-t * 2)

    wave_norm = _normalize(wave.tolist())

    for x in range(resolution - 1):
        y1 = int(mid_y - wave_norm[x] * h * 0.4)
        y2 = int(mid_y - wave_norm[x + 1] * h * 0.4)
        color = _value_to_palette_color(wave_norm[x], palette_rgb)
        draw.line([(x, y1), (x + 1, y2)], fill=color, width=config.line_width)
        y1_mirror = int(mid_y + wave_norm[x] * h * 0.4)
        y2_mirror = int(mid_y + wave_norm[x + 1] * h * 0.4)
        dim_color = tuple(int(c * 0.5) for c in color)
        draw.line([(x, y1_mirror), (x + 1, y2_mirror)], fill=dim_color, width=1)

    if config.blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=config.blur_radius))

    return img


def render_phi_arc(
    notes: List[NoteEvent],
    config: RenderConfig,
    palette_hex: Optional[List[str]] = None,
) -> "Image.Image":
    if not HAS_PIL:
        raise ImportError("Pillow required")

    w, h = config.width, config.height
    colors_hex = palette_hex or DEFAULT_PALETTES.get(config.palette_id, DEFAULT_PALETTES["golden_hour"])
    palette_rgb = [_hex_to_rgb(c) for c in colors_hex]

    img = Image.new("RGB", (w, h), palette_rgb[0])
    draw = ImageDraw.Draw(img)

    n_notes = len(notes)
    if n_notes < 2:
        return img

    pitches_norm = _normalize([float(n.pitch) for n in notes])
    total_dur = notes[-1].start + notes[-1].duration - notes[0].start
    if total_dur <= 0:
        total_dur = float(n_notes)

    golden_ratios = []
    depth = 8
    for d in range(1, depth + 1):
        golden_ratios.append(1.0 / PHI**d)
        golden_ratios.append(1.0 - 1.0 / PHI**d)
    golden_ratios = sorted(set(golden_ratios))

    cx, cy = w // 2, h // 2
    max_r = min(w, h) * 0.42

    for i in range(n_notes):
        pos = (notes[i].start - notes[0].start) / total_dur
        angle = pos * 2 * math.pi * PHI
        r = max_r * (0.3 + 0.7 * pitches_norm[i])

        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))

        near_golden = min(abs(pos - gp) for gp in golden_ratios) if golden_ratios else 1.0
        if near_golden < 0.02:
            color = _lerp_color(palette_rgb[-1], (255, 215, 0), 0.7)
            dot_r = 6
        elif near_golden < 0.05:
            color = _lerp_color(palette_rgb[-1], palette_rgb[-2], 0.5)
            dot_r = 4
        else:
            color = _value_to_palette_color(pitches_norm[i], palette_rgb)
            dot_r = 2

        draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=color)

    prev_pts = []
    for i in range(n_notes):
        pos = (notes[i].start - notes[0].start) / total_dur
        angle = pos * 2 * math.pi * PHI
        r = max_r * (0.3 + 0.7 * pitches_norm[i])
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        prev_pts.append((x, y))

    if len(prev_pts) >= 2:
        for i in range(len(prev_pts) - 1):
            alpha = pitches_norm[i]
            color = _value_to_palette_color(alpha, palette_rgb)
            dim = tuple(int(c * 0.4) for c in color)
            draw.line([prev_pts[i], prev_pts[i + 1]], fill=dim, width=1)

    for gp in golden_ratios[:6]:
        angle = gp * 2 * math.pi * PHI
        for r_frac in [0.3, 0.5, 0.7, 0.9]:
            r = max_r * r_frac
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 215, 0, 80))

    if config.blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=config.blur_radius))

    return img


def render_fractal(
    notes: List[NoteEvent],
    config: RenderConfig,
    palette_hex: Optional[List[str]] = None,
) -> "Image.Image":
    if not HAS_PIL:
        raise ImportError("Pillow required")

    w, h = config.width, config.height
    colors_hex = palette_hex or DEFAULT_PALETTES.get(config.palette_id, DEFAULT_PALETTES["emotional_dark"])
    palette_rgb = [_hex_to_rgb(c) for c in colors_hex]

    img = Image.new("RGB", (w, h), palette_rgb[0])
    draw = ImageDraw.Draw(img)

    n_notes = len(notes)
    if n_notes < 4:
        return img

    pitches = [float(n.pitch) for n in notes]

    def _draw_recursive(x0, y0, x1, y1, note_slice, depth):
        if depth <= 0 or len(note_slice) < 2:
            return

        mid_idx = len(note_slice) // 2

        left = note_slice[:mid_idx]
        right = note_slice[mid_idx:]

        left_mean = sum(n.pitch for n in left) / len(left) if left else 60
        right_mean = sum(n.pitch for n in right) / len(right) if right else 60

        left_y = y0 + (y1 - y0) * 0.5 * (1.0 - (left_mean - 36) / 60.0)
        right_y = y0 + (y1 - y0) * 0.5 * (1.0 - (right_mean - 36) / 60.0)

        mid_x = (x0 + x1) / 2

        avg_pitch_norm = (left_mean + right_mean - 72) / 60.0
        color = _value_to_palette_color(max(0.0, min(1.0, avg_pitch_norm + 0.5)), palette_rgb)

        line_w = max(1, depth)
        draw.line(
            [(int(x0), int(left_y)), (int(mid_x), int((left_y + right_y) / 2))],
            fill=color, width=line_w,
        )
        draw.line(
            [(int(mid_x), int((left_y + right_y) / 2)), (int(x1), int(right_y))],
            fill=color, width=line_w,
        )

        _draw_recursive(x0, y0, mid_x, y1, left, depth - 1)
        _draw_recursive(mid_x, y0, x1, y1, right, depth - 1)

    max_depth = min(8, int(math.log2(max(n_notes, 2))))
    _draw_recursive(0, 0, w, h, notes, max_depth)

    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    for scale in [1, 2, 4, 8]:
        if scale >= n_notes:
            break
        for i in range(0, n_notes - scale, scale):
            x = int(i / (n_notes - 1) * (w - 1))
            chunk = intervals[i:i + scale] if i + scale <= len(intervals) else intervals[i:]
            if not chunk:
                continue
            mean_iv = sum(chunk) / len(chunk)
            y = int(h * 0.8 - mean_iv / 12.0 * h * 0.6)
            dot_r = scale
            alpha = 1.0 / (scale + 1)
            c = _value_to_palette_color(mean_iv / 12.0, palette_rgb)
            c_dim = tuple(int(cv * alpha) for cv in c)
            draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=c_dim)

    if config.blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=config.blur_radius * 1.5))

    return img


# Import new renderers
try:
    from music_math.viz.renderer_new import (
        render_emotional_heatmap,
        render_spectrogram_art,
        render_pattern_art,
    )
    HAS_NEW_RENDERERS = True
except ImportError:
    HAS_NEW_RENDERERS = False
    render_emotional_heatmap = render_spectrogram_art = render_pattern_art = None

RENDERERS = {
    "kalman": render_kalman,
    "spectral": render_spectral,
    "waveform": render_waveform,
    "phi_arc": render_phi_arc,
    "fractal": render_fractal,
}

if HAS_NEW_RENDERERS:
    RENDERERS.update({
        "emotional_heatmap": render_emotional_heatmap,
        "spectrogram_art": render_spectrogram_art,
        "pattern_art": render_pattern_art,
    })


def render_music_art(
    notes: List[NoteEvent],
    mode: str = "kalman",
    config: Optional[RenderConfig] = None,
    palette_hex: Optional[List[str]] = None,
) -> "Image.Image":
    if config is None:
        config = RenderConfig()
    renderer = RENDERERS.get(mode)
    if renderer is None:
        raise ValueError(f"Unknown render mode: {mode}. Available: {list(RENDERERS.keys())}")
    return renderer(notes, config, palette_hex)


def render_to_bytes(
    notes: List[NoteEvent],
    mode: str = "kalman",
    config: Optional[RenderConfig] = None,
    palette_hex: Optional[List[str]] = None,
    fmt: str = "PNG",
) -> bytes:
    img = render_music_art(notes, mode, config, palette_hex)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=95)
    return buf.getvalue()


__all__ = [
    "RenderConfig",
    "render_music_art",
    "render_to_bytes",
    "RENDERERS",
]
