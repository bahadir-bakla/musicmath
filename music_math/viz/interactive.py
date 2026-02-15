"""Plotly tabanlı basit interaktif embedding görselleştirmesi."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def interactive_composer_space(df: pd.DataFrame, umap_coords):
    """
    UMAP uzayında besteci / dönem dağılımını interaktif olarak göster.
    """
    df_plot = df.copy()
    df_plot["umap_x"] = umap_coords[:, 0]
    df_plot["umap_y"] = umap_coords[:, 1]

    fig = px.scatter(
        df_plot,
        x="umap_x",
        y="umap_y",
        color="era",
        symbol="composer",
        hover_data=["composer", "era", "form"],
        title="Klasik Müziğin Matematiksel Haritası",
        labels={"umap_x": "Boyut 1", "umap_y": "Boyut 2"},
    )
    return fig


__all__ = ["interactive_composer_space"]

