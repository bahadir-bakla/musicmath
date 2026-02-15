#!/usr/bin/env python
"""
Küçük piano_midi feature matrix'i üzerinde derinlemesine analiz:

- Besteci mesafe matrisi (heatmap)
- PCA (PC1/PC2) ve PC1 feature importance
- Bach tarzında basit bir Markov generatif denemesi

Kullanım:
    python analyze_features.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger
from music_math.data.loader import parse_midi_to_note_events
from music_math.generation.generator import ClassicalMusicGenerator, save_score_as_midi
from music_math.model.distribution import ComposerDistributionModel, composer_distance_matrix
from music_math.model.markov import MusicMarkovModel


logger = get_logger(__name__)


def load_feature_matrix() -> pd.DataFrame:
    feat_path = CONFIG.paths.root / "results" / "stats" / "feature_matrix.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_matrix.csv bulunamadı: {feat_path}")
    df = pd.read_csv(feat_path)
    logger.info("feature_matrix yüklendi: shape=%s", df.shape)
    return df


def run_pca_and_plot(df: pd.DataFrame) -> None:
    """PC1/PC2 scatter ve PC1 yüklerini hesapla."""
    meta_cols = {"filepath", "composer", "era", "form"}
    feat_cols = [c for c in df.columns if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)]

    X = df[feat_cols].fillna(0.0).values
    
    # KRITIK: Feature'ları standardize et (mean=0, std=1)
    # Böylece büyük ölçekli feature'lar (örn. total_notes) PCA'yı domine etmez
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X_scaled)

    print("\n=== PCA Açıklanan Varyans (Standardized Features) ===")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var:.3f}")

    figures_dir = CONFIG.paths.figures
    figures_dir.mkdir(parents=True, exist_ok=True)

    # PC1 vs PC2, dönemlere göre renklendirilmiş
    plt.figure(figsize=(8, 6))
    eras = sorted(df["era"].dropna().unique())
    era_colors = {
        "Baroque": "#1f77b4",
        "Classical": "#ff7f0e", 
        "Romantic": "#2ca02c",
        "Late Romantic": "#d62728"
    }
    era_markers = {
        "Baroque": "o",
        "Classical": "s",
        "Romantic": "^", 
        "Late Romantic": "D"
    }
    
    for era in eras:
        mask = df["era"] == era
        if not mask.any():
            continue
        plt.scatter(
            X2[mask, 0],
            X2[mask, 1],
            label=era,
            marker=era_markers.get(era, "o"),
            color=era_colors.get(era, "gray"),
            s=150,
            alpha=0.8,
            edgecolors='black',
            linewidth=1.5
        )
        
        # Eser isimlerini ekle
        for idx in df[mask].index:
            filepath = df.loc[idx, 'filepath']
            filename = Path(filepath).stem if isinstance(filepath, str) else str(idx)
            plt.annotate(
                filename,
                (X2[idx, 0], X2[idx, 1]),
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
    
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontweight='bold')
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)
    plt.title("PCA – Dönem bazında dağılım (Standardized Features)", fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    out_pca = figures_dir / "pca_era.png"
    plt.savefig(out_pca, dpi=200)
    plt.close()
    print(f"PCA grafiği kaydedildi: {out_pca}")

    # PC1 feature importance (yükler)
    pc1 = pca.components_[0]
    loadings = sorted(
        zip(feat_cols, pc1),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    print("\n=== PC1'i en çok domine eden 10 feature ===")
    for name, w in loadings[:10]:
        print(f"{name:25s}  loading={w:+.3f}")


def run_composer_distance_matrix(df: pd.DataFrame) -> None:
    """Besteci bazlı dağılım modeli ve mesafe matrisi."""
    # Feature'ları standardize et
    meta_cols = {"filepath", "composer", "era", "form"}
    feat_cols = [c for c in df.columns if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)]
    
    X = df[feat_cols].fillna(0.0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Bestecilere göre ortalama feature vektörlerini hesapla
    composers = sorted(df['composer'].dropna().unique())
    composer_means = {}
    
    for composer in composers:
        mask = df['composer'] == composer
        composer_means[composer] = X_scaled[mask].mean(axis=0)
    
    # Basit Euclidean distance matrisi (standardized verilerde)
    n = len(composers)
    dist_matrix = np.zeros((n, n))
    
    for i, c1 in enumerate(composers):
        for j, c2 in enumerate(composers):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                # Euclidean distance
                dist_matrix[i, j] = np.linalg.norm(composer_means[c1] - composer_means[c2])
    
    dist_df = pd.DataFrame(dist_matrix, index=composers, columns=composers)

    print("\n=== Besteciler Arası Matematiksel Mesafe Matrisi (Euclidean, Standardized) ===")
    print(dist_df.round(3).to_string())

    figures_dir = CONFIG.paths.figures
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        dist_df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Mesafe (d)"},
    )
    plt.title("Besteciler Arası Matematiksel Mesafe")
    plt.tight_layout()
    out_mat = figures_dir / "composer_distance_matrix.png"
    plt.savefig(out_mat, dpi=200)
    plt.close()
    print(f"Mesafe matrisi heatmap'i kaydedildi: {out_mat}")


def train_bach_markov_and_generate() -> Path | None:
    """
    Sadece Bach eserlerinden Markov modeli eğit ve kısa bir Bach tarzı melodi üret.
    """
    meta_clean = CONFIG.paths.root / "metadata_clean.csv"
    if not meta_clean.exists():
        print("metadata_clean.csv bulunamadı, Bach Markov denemesi atlanıyor.")
        return None

    df_meta = pd.read_csv(meta_clean)
    bach_rows = df_meta[df_meta["composer"] == "Bach"]
    if bach_rows.empty:
        print("metadata_clean.csv içinde Bach eseri yok, Markov denemesi atlanıyor.")
        return None

    note_sequences = []
    for _, row in bach_rows.iterrows():
        path = CONFIG.paths.root / row["file_path"]
        events = parse_midi_to_note_events(path)
        if events:
            note_sequences.append(events)

    if not note_sequences:
        print("Bach için parse edilebilen nota dizisi bulunamadı.")
        return None

    markov = MusicMarkovModel(order=1)
    markov.fit(note_sequences)

    generator = ClassicalMusicGenerator(markov_model=markov)
    score, notes = generator.generate(style="baroque", length_bars=16, tonic=60)

    out_midi = CONFIG.paths.generated_midi / "bach_style_sample.mid"
    save_score_as_midi(score, out_midi)

    print(f"\nBach tarzında örnek üretim MIDI dosyası: {out_midi}")
    return out_midi


def main():
    print("=" * 60)
    print("FAZ 3–4 Analizleri: PCA, Mesafe Matrisi, Markov")
    print("=" * 60)

    df = load_feature_matrix()

    run_pca_and_plot(df)
    run_composer_distance_matrix(df)
    train_bach_markov_and_generate()

    print("\nAnaliz tamamlandı. Figürler 'results/figures', MIDI 'results/generated_midi' altında.")


if __name__ == "__main__":
    main()

