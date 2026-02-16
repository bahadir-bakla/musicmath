"""
Additional Renderers - FAZ 4 Yeni Görsel Modları

1. Emotional Heatmap Renderer
2. Spectrogram Art Renderer  
3. Pattern Art Renderer

Bu modül renderer.py'ye import edilmek üzere tasarlanmıştır.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from collections import Counter
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from music_math.core.types import NoteEvent


# Helper functions (renderer.py'den kopyalanacak)
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


def _normalize(series: List[float]) -> List[float]:
    if not series:
        return []
    mn, mx = min(series), max(series)
    span = mx - mn
    if span == 0:
        return [0.5] * len(series)
    return [(v - mn) / span for v in series]


# Paletler
DEFAULT_PALETTES = {
    "emotional_dark": ["#0a0a0f", "#1a1a2e", "#16213e", "#0f3460", "#e94560"],
    "spectral_abstract": ["#0b0b12", "#2e1065", "#4c1d95", "#06b6d4", "#22d3ee"],
    "minimal_contrast": ["#0c0c0c", "#171717", "#404040", "#a3a3a3", "#fafafa"],
}


# =============================================================================
# 1. EMOTIONAL HEATMAP RENDERER
# =============================================================================

def render_emotional_heatmap(
    notes: List[NoteEvent],
    config,
    palette_hex: Optional[List[str]] = None,
):
    """Emotional Heatmap: Pitch → duygu haritası."""
    if not HAS_PIL:
        raise ImportError("Pillow required")
    
    from dataclasses import dataclass
    
    w, h = config.width, config.height
    colors_hex = palette_hex or DEFAULT_PALETTES.get("emotional_dark", DEFAULT_PALETTES["emotional_dark"])
    palette_rgb = [_hex_to_rgb(c) for c in colors_hex]
    
    img = Image.new("RGB", (w, h), palette_rgb[0])
    draw = ImageDraw.Draw(img)
    
    n_notes = len(notes)
    if n_notes < 2:
        return img
    
    pitches = [n.pitch for n in notes]
    
    # Tonalite tahmini
    pitch_classes = [p % 12 for p in pitches]
    pc_counts = Counter(pitch_classes)
    
    major_third = (pc_counts.get(4, 0) + pc_counts.get(7, 0)) / max(1, sum(pc_counts.values()))
    minor_third = (pc_counts.get(3, 0) + pc_counts.get(7, 0)) / max(1, sum(pc_counts.values()))
    is_major = major_third > minor_third
    
    emotion_scale = 0.7 if is_major else 0.3
    
    # Heatmap grid
    grid_cols = min(50, n_notes)
    grid_rows = 12
    cell_w = w // grid_cols
    cell_h = h // grid_rows
    
    heatmap = [[0.0 for _ in range(grid_cols)] for _ in range(grid_rows)]
    
    for i, note in enumerate(notes):
        col = int((i / n_notes) * grid_cols)
        row = 11 - (note.pitch % 12)
        
        if 0 <= row < grid_rows and 0 <= col < grid_cols:
            intensity = note.duration * 0.5 + getattr(note, 'velocity', 64) / 127 * 0.5
            heatmap[row][col] += intensity
    
    max_val = max(max(row) for row in heatmap) if heatmap else 1
    if max_val > 0:
        heatmap = [[v / max_val for v in row] for row in heatmap]
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = col * cell_w
            y = row * cell_h
            
            value = heatmap[row][col]
            if value > 0.05:
                if is_major:
                    base_color = _value_to_palette_color(value * emotion_scale, palette_rgb)
                else:
                    base_color = _value_to_palette_color(value * (1 - emotion_scale), palette_rgb[::-1])
                
                brightness = 0.3 + 0.7 * value
                color = tuple(int(c * brightness) for c in base_color)
                
                draw.rectangle([x, y, x + cell_w, y + cell_h], fill=color)
    
    if config.blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=config.blur_radius))
    
    return img


# =============================================================================
# 2. SPECTROGRAM ART RENDERER
# =============================================================================

def render_spectrogram_art(
    notes: List[NoteEvent],
    config,
    palette_hex: Optional[List[str]] = None,
):
    """Spectrogram Art: Zaman-frekans spektrogramı."""
    if not HAS_PIL:
        raise ImportError("Pillow required")
    
    w, h = config.height, config.width
    colors_hex = palette_hex or DEFAULT_PALETTES.get("spectral_abstract", DEFAULT_PALETTES["spectral_abstract"])
    palette_rgb = [_hex_to_rgb(c) for c in colors_hex]
    
    img = Image.new("RGB", (w, h), palette_rgb[0])
    draw = ImageDraw.Draw(img)
    
    n_notes = len(notes)
    if n_notes < 2:
        return img.transpose(Image.Transpose.ROTATE_90) if w > h else img
    
    total_time = notes[-1].start + notes[-1].duration
    
    num_bins = 12
    num_time_frames = min(100, n_notes)
    
    spectrogram = [[0.0 for _ in range(num_time_frames)] for _ in range(num_bins)]
    
    for note in notes:
        time_norm = note.start / total_time if total_time > 0 else 0
        time_frame = int(time_norm * (num_time_frames - 1))
        pc = note.pitch % 12
        energy = (getattr(note, 'velocity', 64) / 127) * (note.duration / total_time * n_notes)
        
        if 0 <= time_frame < num_time_frames:
            spectrogram[pc][time_frame] += energy
    
    max_val = max(max(row) for row in spectrogram) if spectrogram else 1
    if max_val > 0:
        spectrogram = [[v / max_val for v in row] for row in spectrogram]
    
    bin_h = h // num_bins
    frame_w = w // num_time_frames
    
    for pc in range(num_bins):
        for t in range(num_time_frames):
            y = h - (pc + 1) * bin_h
            x = t * frame_w
            
            value = spectrogram[pc][t]
            if value > 0.02:
                color = _value_to_palette_color(value, palette_rgb)
                brightness = 0.4 + 0.6 * value
                color = tuple(min(255, int(c * brightness)) for c in color)
                draw.rectangle([x, y, x + frame_w, y + bin_h], fill=color)
    
    img = img.transpose(Image.Transpose.ROTATE_270)
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img = img.resize((config.width, config.height), Image.Resampling.LANCZOS)
    
    if config.blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=config.blur_radius * 0.5))
    
    return img


# =============================================================================
# 3. PATTERN ART RENDERER
# =============================================================================

def render_pattern_art(
    notes: List[NoteEvent],
    config,
    palette_hex: Optional[List[str]] = None,
):
    """Pattern Art: Bulunan pattern'leri görselleştirme."""
    if not HAS_PIL:
        raise ImportError("Pillow required")
    
    w, h = config.width, config.height
    colors_hex = palette_hex or DEFAULT_PALETTES.get("minimal_contrast", DEFAULT_PALETTES["minimal_contrast"])
    palette_rgb = [_hex_to_rgb(c) for c in colors_hex]
    
    img = Image.new("RGB", (w, h), palette_rgb[0])
    draw = ImageDraw.Draw(img)
    
    n_notes = len(notes)
    if n_notes < 4:
        return img
    
    pitches = [n.pitch for n in notes]
    intervals = list(np.diff(pitches))
    
    # Pattern bul
    pattern_lengths = [3, 4, 5, 6]
    found_patterns = []
    
    for length in pattern_lengths:
        for start in range(len(intervals) - length + 1):
            pattern = tuple(intervals[start:start + length])
            occurrences = []
            for i in range(len(intervals) - length + 1):
                if tuple(intervals[i:i + length]) == pattern:
                    occurrences.append(i)
            
            if len(occurrences) >= 2:
                found_patterns.append({
                    "pattern": pattern,
                    "length": length,
                    "occurrences": occurrences
                })
    
    found_patterns.sort(key=lambda p: len(p["occurrences"]), reverse=True)
    top_patterns = found_patterns[:5]
    
    # Arka plan
    norm_pitches = _normalize(pitches)
    points = []
    
    for i in range(n_notes):
        x = int(i / (n_notes - 1) * (w - 1))
        y = int((1.0 - norm_pitches[i]) * h * 0.8 + h * 0.1)
        points.append((x, y))
    
    for i in range(len(points) - 1):
        dim_color = tuple(int(c * 0.3) for c in palette_rgb[2])
        draw.line([points[i], points[i + 1]], fill=dim_color, width=1)
    
    # Pattern'leri vurgula
    for pattern_idx, pattern_data in enumerate(top_patterns):
        color_idx = (pattern_idx + 3) % len(palette_rgb)
        pattern_color = palette_rgb[color_idx]
        
        for occ_start in pattern_data["occurrences"]:
            note_start = occ_start
            note_end = occ_start + pattern_data["length"]
            
            if note_start < len(points) and note_end < len(points):
                x1 = points[note_start][0]
                x2 = points[note_end][0]
                y_center = (points[note_start][1] + points[note_end][1]) // 2
                
                box_height = h // 8
                draw.rectangle(
                    [x1, y_center - box_height // 2, x2, y_center + box_height // 2],
                    outline=pattern_color,
                    width=2
                )
                
                for i in range(note_start, min(note_end + 1, len(points) - 1)):
                    draw.line([points[i], points[i + 1]], fill=pattern_color, width=3)
    
    if config.blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=config.blur_radius * 0.3))
    
    return img


__all__ = [
    "render_emotional_heatmap",
    "render_spectrogram_art",
    "render_pattern_art",
]
