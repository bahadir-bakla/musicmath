#!/usr/bin/env python
"""
Feature matrix üzerinde PCA analizi ve görselleştirme.

Kullanım:
    python analyze_pca.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    feat_path = Path("results/stats/feature_matrix.csv")
    
    if not feat_path.exists():
        print(f"❌ Feature matrix bulunamadı: {feat_path}")
        print("Önce 'python run_pipeline.py' komutunu çalıştırın.")
        return
    
    print("=" * 60)
    print("PCA Analizi: Dönemlere göre feature uzayı görselleştirmesi")
    print("=" * 60)
    
    # Feature matrix'i yükle
    df = pd.read_csv(feat_path)
    print(f"\n✓ Feature matrix yüklendi: {df.shape}")
    
    # Metadata kolonları
    meta_cols = ["filepath", "composer", "era", "form"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    
    # Feature kolonları (sayısal değerler)
    feat_cols = [c for c in df.columns if c not in meta_cols]
    print(f"  • Metadata kolonları: {len(meta_cols)}")
    print(f"  • Feature kolonları: {len(feat_cols)}")
    
    # Feature matrisini hazırla (NaN'ları 0 ile doldur)
    X = df[feat_cols].fillna(0.0).values
    print(f"\n✓ Feature matrisi: shape={X.shape}")
    
    # PCA: 2 bileşene indir
    print("\n⚙️  PCA (n_components=2) uygulanıyor...")
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    
    explained_var = pca.explained_variance_ratio_
    print(f"  • PC1 açıklanan varyans: {explained_var[0]:.2%}")
    print(f"  • PC2 açıklanan varyans: {explained_var[1]:.2%}")
    print(f"  • Toplam açıklanan varyans: {explained_var.sum():.2%}")
    
    # Görselleştirme
    print("\n📊 Grafik oluşturuluyor...")
    
    plt.figure(figsize=(10, 7))
    
    # Dönemlere göre farklı marker ve renk
    era_styles = {
        "Baroque": {"marker": "o", "color": "#1f77b4", "label": "Baroque (Bach)"},
        "Classical": {"marker": "s", "color": "#ff7f0e", "label": "Classical (Mozart)"},
        "Romantic": {"marker": "^", "color": "#2ca02c", "label": "Romantic (Chopin)"},
        "Late Romantic": {"marker": "D", "color": "#d62728", "label": "Late Romantic (Debussy)"},
    }
    
    if 'era' in df.columns:
        for era, style in era_styles.items():
            mask = df["era"] == era
            if mask.any():
                plt.scatter(
                    X2[mask, 0], 
                    X2[mask, 1], 
                    marker=style["marker"],
                    color=style["color"],
                    s=150,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=1.5,
                    label=style["label"]
                )
                
                # Eser isimlerini göster
                for idx in df[mask].index:
                    filepath = df.loc[idx, 'filepath']
                    filename = Path(filepath).stem
                    plt.annotate(
                        filename,
                        (X2[idx, 0], X2[idx, 1]),
                        xytext=(8, 8),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7
                    )
    else:
        # Era yoksa hepsini aynı şekilde çiz
        plt.scatter(X2[:, 0], X2[:, 1], s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    plt.xlabel(f"PC1 ({explained_var[0]:.1%} variance)", fontsize=12, fontweight='bold')
    plt.ylabel(f"PC2 ({explained_var[1]:.1%} variance)", fontsize=12, fontweight='bold')
    plt.title(
        "Klasik Müzik Feature Uzayı: Dönemlere Göre PCA\n" +
        f"(Piano MIDI dataset, {len(df)} eser, {len(feat_cols)} feature)",
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Grafiği kaydet
    output_path = Path("results/figures")
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "pca_era_separation.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Grafik kaydedildi: {fig_path}")
    
    # Grafiği göster
    plt.show()
    
    print("\n" + "=" * 60)
    print("Analiz tamamlandı! 🎵")
    print("=" * 60)
    
    # Dönemler arası mesafe analizi
    if 'era' in df.columns:
        print("\n📏 Dönemler arası Euclidean mesafeler (PC uzayında):")
        print("-" * 60)
        eras = df['era'].unique()
        for i, era1 in enumerate(eras):
            for era2 in eras[i+1:]:
                mask1 = df['era'] == era1
                mask2 = df['era'] == era2
                
                # Her dönemin centroid'ini hesapla
                centroid1 = X2[mask1].mean(axis=0)
                centroid2 = X2[mask2].mean(axis=0)
                
                # Euclidean mesafe
                dist = np.linalg.norm(centroid1 - centroid2)
                print(f"  {era1:20s} ↔ {era2:20s} : {dist:.3f}")

if __name__ == '__main__':
    main()
