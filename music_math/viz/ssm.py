"""Self-similarity matrix görselleştirmesi."""

from __future__ import annotations

import matplotlib.pyplot as plt

from music_math.features.structural import self_similarity_matrix


def visualize_ssm(pitches, title: str = "", window: int = 16):
    ssm = self_similarity_matrix(pitches, window=window)
    if ssm.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(ssm, cmap="viridis", origin="upper", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("Zaman (window index)")
    ax.set_ylabel("Zaman (window index)")
    ax.set_title(title or "Self-Similarity Matrix")
    plt.colorbar(im, ax=ax, label="Benzerlik")
    plt.tight_layout()
    return fig


__all__ = ["visualize_ssm"]

