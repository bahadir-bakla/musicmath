#!/usr/bin/env python
"""
İleri seviye analizler: Hierarchical Clustering, t-SNE, UMAP, Feature Importance

Kullanım:
    python scripts/advanced_analysis.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger

logger = get_logger(__name__)


def load_and_prepare_data():
    """Feature matrix'i yükle ve standardize et."""
    feat_path = CONFIG.paths.root / "results" / "stats" / "feature_matrix.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_matrix.csv bulunamadı: {feat_path}")
    
    df = pd.read_csv(feat_path)
    logger.info("Feature matrix yüklendi: shape=%s", df.shape)
    
    # Feature kolonlarını ayır
    meta_cols = {"filepath", "composer", "era", "form"}
    feat_cols = [c for c in df.columns if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)]
    
    # Standardize
    X = df[feat_cols].fillna(0.0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled, feat_cols


def plot_hierarchical_clustering(df: pd.DataFrame, X_scaled: np.ndarray):
    """Hierarchical clustering dendrogram (eser ve besteci bazlı)."""
    figures_dir = CONFIG.paths.figures
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) Eser bazlı dendrogram
    plt.figure(figsize=(12, 6))
    
    # Ward linkage (minimum variance)
    Z = linkage(X_scaled, method='ward')
    
    # Eser isimleri için label oluştur
    labels = []
    for _, row in df.iterrows():
        filepath = row['filepath']
        filename = Path(filepath).stem if isinstance(filepath, str) else str(_)
        composer = row.get('composer', 'Unknown')
        labels.append(f"{composer[:4]}_{filename[:10]}")
    
    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0.7 * max(Z[:, 2])  # Renk threshold'u
    )
    
    plt.title("Hierarchical Clustering: Eserlerin Benzerlik Ağacı", fontweight='bold', fontsize=14)
    plt.xlabel("Eser", fontweight='bold')
    plt.ylabel("Mesafe (Ward)", fontweight='bold')
    plt.tight_layout()
    
    out_path = figures_dir / "hierarchical_clustering_works.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Eser bazlı dendrogram kaydedildi: %s", out_path)
    
    # 2) Besteci bazlı dendrogram
    if 'composer' in df.columns:
        composers = sorted(df['composer'].dropna().unique())
        composer_means = {}
        
        for composer in composers:
            mask = df['composer'] == composer
            composer_means[composer] = X_scaled[mask].mean(axis=0)
        
        composer_matrix = np.array([composer_means[c] for c in composers])
        
        plt.figure(figsize=(8, 6))
        Z_comp = linkage(composer_matrix, method='ward')
        
        dendrogram(
            Z_comp,
            labels=composers,
            leaf_rotation=45,
            leaf_font_size=12,
            color_threshold=0.7 * max(Z_comp[:, 2])
        )
        
        plt.title("Hierarchical Clustering: Bestecilerin Benzerlik Ağacı", fontweight='bold', fontsize=14)
        plt.xlabel("Besteci", fontweight='bold')
        plt.ylabel("Mesafe (Ward)", fontweight='bold')
        plt.tight_layout()
        
        out_path_comp = figures_dir / "hierarchical_clustering_composers.png"
        plt.savefig(out_path_comp, dpi=300)
        plt.close()
        logger.info("Besteci bazlı dendrogram kaydedildi: %s", out_path_comp)


def plot_tsne(df: pd.DataFrame, X_scaled: np.ndarray):
    """t-SNE 2D projeksiyon."""
    n = len(X_scaled)
    if n < 4:
        logger.warning("t-SNE için en az 4 örnek gerekir (şu an %d), atlanıyor.", n)
        return
    figures_dir = CONFIG.paths.figures

    perplexity = max(2, min(30, (n - 1) // 2))
    logger.info("t-SNE hesaplanıyor (perplexity=%d)...", perplexity)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    
    if 'era' in df.columns:
        eras = sorted(df['era'].dropna().unique())
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
            mask = df['era'] == era
            if not mask.any():
                continue
            
            plt.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                label=era,
                marker=era_markers.get(era, "o"),
                color=era_colors.get(era, "gray"),
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidth=1.5
            )
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=150, alpha=0.7, edgecolors='black')
    
    plt.xlabel("t-SNE Dim 1", fontweight='bold')
    plt.ylabel("t-SNE Dim 2", fontweight='bold')
    plt.title("t-SNE Projeksiyon: Dönem Bazlı Dağılım", fontweight='bold', fontsize=14)
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    out_path = figures_dir / "tsne_projection.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("t-SNE grafiği kaydedildi: %s", out_path)


def plot_umap(df: pd.DataFrame, X_scaled: np.ndarray):
    """UMAP 2D projeksiyon (varsa)."""
    if not HAS_UMAP:
        logger.warning("UMAP yüklü değil, atlanıyor. Yüklemek için: pip install umap-learn")
        return
    
    figures_dir = CONFIG.paths.figures
    
    logger.info("UMAP hesaplanıyor...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    
    if 'era' in df.columns:
        eras = sorted(df['era'].dropna().unique())
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
            mask = df['era'] == era
            if not mask.any():
                continue
            
            plt.scatter(
                X_umap[mask, 0],
                X_umap[mask, 1],
                label=era,
                marker=era_markers.get(era, "o"),
                color=era_colors.get(era, "gray"),
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidth=1.5
            )
    else:
        plt.scatter(X_umap[:, 0], X_umap[:, 1], s=150, alpha=0.7, edgecolors='black')
    
    plt.xlabel("UMAP Dim 1", fontweight='bold')
    plt.ylabel("UMAP Dim 2", fontweight='bold')
    plt.title("UMAP Projeksiyon: Dönem Bazlı Dağılım", fontweight='bold', fontsize=14)
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    out_path = figures_dir / "umap_projection.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("UMAP grafiği kaydedildi: %s", out_path)


def plot_feature_importance(df: pd.DataFrame, X_scaled: np.ndarray, feat_cols: list):
    """Random Forest ile feature importance analizi."""
    if 'composer' not in df.columns:
        logger.warning("'composer' kolonu yok, feature importance atlanıyor")
        return
    
    # Composer'ları label encoding yap
    composers = df['composer'].dropna()
    if len(composers.unique()) < 2:
        logger.warning("Yeterli besteci yok, feature importance atlanıyor")
        return
    
    y = composers.astype('category').cat.codes
    mask = ~composers.isna()
    X_train = X_scaled[mask]
    y_train = y.values
    
    logger.info("Feature importance hesaplanıyor (Random Forest)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    
    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=feat_cols)
    top_features = importances.nlargest(20)
    
    figures_dir = CONFIG.paths.figures
    
    plt.figure(figsize=(10, 8))
    top_features.sort_values().plot(kind='barh', color='steelblue', edgecolor='black')
    plt.xlabel("Importance (Gini)", fontweight='bold')
    plt.ylabel("Feature", fontweight='bold')
    plt.title("Top 20 Feature Importance: Besteci Tahmini", fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    out_path = figures_dir / "feature_importance_rf.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Feature importance grafiği kaydedildi: %s", out_path)
    
    # Console'a yazdır
    print("\n=== Top 20 En Önemli Feature'lar (Random Forest) ===")
    for feat, imp in top_features.sort_values(ascending=False).items():
        print(f"{feat:30s}  importance={imp:.4f}")


def main():
    print("=" * 60)
    print("İleri Seviye Analizler")
    print("=" * 60)
    
    df, X_scaled, feat_cols = load_and_prepare_data()
    
    print("\n[1/5] Hierarchical Clustering...")
    plot_hierarchical_clustering(df, X_scaled)
    
    print("\n[2/5] t-SNE Projeksiyon...")
    plot_tsne(df, X_scaled)
    
    print("\n[3/5] UMAP Projeksiyon...")
    plot_umap(df, X_scaled)
    
    print("\n[4/5] Feature Importance (Random Forest)...")
    plot_feature_importance(df, X_scaled, feat_cols)
    
    print("\n" + "=" * 60)
    print("Tüm analizler tamamlandı!")
    print("Çıktılar: results/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
