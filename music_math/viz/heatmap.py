"""Pitch-time heatmap görselleştirmesi."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from music_math.core.types import NoteEvent


def pitch_time_heatmap(
    notes: Iterable[NoteEvent],
    title: str = "",
    resolution: int = 50,
):
    events = list(notes)
    if not events:
        return None

    max_time = max(e.start + e.duration for e in events)
    time_bins = resolution
    pitch_bins = 88  # Piyano tuşu sayısı (A0–C8)

    grid = np.zeros((pitch_bins, time_bins), dtype=float)

    for e in events:
        t_start = int((e.start / max_time) * (time_bins - 1))
        t_end = int(((e.start + e.duration) / max_time) * (time_bins - 1))
        p_idx = min(max(e.pitch - 21, 0), pitch_bins - 1)
        grid[p_idx, t_start : t_end + 1] += 1.0

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap="hot",
        interpolation="gaussian",
    )
    ax.set_xlabel("Zaman")
    ax.set_ylabel("Pitch (düşük → yüksek)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Yoğunluk")
    plt.tight_layout()
    return fig


__all__ = ["pitch_time_heatmap"]

