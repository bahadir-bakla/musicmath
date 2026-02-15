# 🎨 FAZ 7 — GÖRSELLEŞTİRME & SANATSAL KATMAN

> **Süre:** 4 hafta  
> **Önceki Faz:** FAZ 6 — İnsan Deneyi  
> **Sonraki Faz:** FAZ 8 — Paper Yazımı

---

## 🎯 FAZ AMACI

Projeye **estetik boyut** kazandırmak. Müziğin matematiksel yapısını görsel olarak ortaya koymak. Hem paper figürleri hem de standalone sanat eseri olabilecek görseller üretmek.

---

## ✅ FAZ ÇIKTILARI

- [ ] "Müziğin Geometrisi" görsel seti (her dönem için)
- [ ] Besteci matematiksel parmak izi görselleri
- [ ] Interactive web demo (basit)
- [ ] Paper için publication-quality figürler
- [ ] (Opsiyonel) Sergi materyali

---

## 🎨 7.1 TEMEL GÖRSELLEŞTİRME TİPLERİ

### Tip 1: Pitch-Time Heatmap (Müziğin Renk Haritası)

```python
# src/viz/heatmap.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def pitch_time_heatmap(notes_data, title="", resolution=50):
    """
    Zamanı X eksenine, pitch'i Y eksenine koy.
    Yoğunluk = nota süresi / velocity (varsa)
    
    Çıktı: Müziğin 'resmi'
    """
    if not notes_data:
        return
    
    max_time = max(n['start'] + n['duration'] for n in notes_data)
    
    # Grid oluştur
    time_bins = resolution
    pitch_bins = 88  # Piyano tuşu sayısı (A0-C8)
    
    grid = np.zeros((pitch_bins, time_bins))
    
    for n in notes_data:
        # Zaman indeksi
        t_start = int((n['start'] / max_time) * (time_bins - 1))
        t_end = int(((n['start'] + n['duration']) / max_time) * (time_bins - 1))
        
        # Pitch indeksi (21=A0, 108=C8)
        p_idx = min(max(n['pitch'] - 21, 0), pitch_bins - 1)
        
        grid[p_idx, t_start:t_end+1] += 1
    
    # Renk haritası: Harmonik yoğunluğa göre
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Custom colormap: Siyah-kırmızı-sarı-beyaz (volkanik)
    cmap = plt.cm.hot
    
    im = ax.imshow(grid, aspect='auto', origin='lower',
                   cmap=cmap, interpolation='gaussian')
    
    # Eksen etiketleri
    ax.set_xlabel("Zaman →", fontsize=12)
    ax.set_ylabel("Pitch (düşük → yüksek)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Oktav çizgileri
    for octave in range(2, 8):
        midi = octave * 12 + 24 - 21  # C notası
        if 0 <= midi < pitch_bins:
            ax.axhline(y=midi, color='gray', alpha=0.3, linewidth=0.5)
    
    plt.colorbar(im, ax=ax, label='Yoğunluk')
    plt.tight_layout()
    
    return fig

# Örnek kullanım
for composer in ['Bach', 'Mozart', 'Chopin', 'Debussy']:
    sample_notes = get_representative_notes(composer, df)
    fig = pitch_time_heatmap(sample_notes, f"{composer}: Müziğin Renk Haritası")
    fig.savefig(f'results/figures/heatmap_{composer}.png', dpi=200)
    plt.close()
```

### Tip 2: Polar Harmony Map (Renk Çemberi)

```python
def polar_harmony_map(notes_data, title="", key_root=0):
    """
    12 pitch class'ı dairesel olarak göster.
    Renk = kullanım sıklığı
    
    Pitch-class circle = müzikal renk tekerleği
    C = 12:00, G = 2:00, D = 4:00, ... (beşliler çemberi)
    """
    
    CIRCLE_OF_FIFTHS_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    NOTE_NAMES = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
    
    # Renk → Pitch class mapping (spektrum)
    colors = plt.cm.hsv(np.linspace(0, 1, 12))
    
    pitches = [n['pitch'] % 12 for n in notes_data]
    counts = [pitches.count(pc) for pc in CIRCLE_OF_FIFTHS_ORDER]
    total = sum(counts)
    freqs = [c/total for c in counts]
    
    # Polar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    
    bars = ax.bar(angles, freqs, width=2*np.pi/12 * 0.9,
                  color=colors, alpha=0.7, edgecolor='white')
    
    # Note isimlerini ekle
    ax.set_xticks(angles)
    ax.set_xticklabels(NOTE_NAMES, fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.set_title(title, size=14, fontweight='bold', pad=20)
    
    # Tonal merkez vurgula
    tonic_idx = CIRCLE_OF_FIFTHS_ORDER.index(key_root % 12)
    bars[tonic_idx].set_edgecolor('gold')
    bars[tonic_idx].set_linewidth(3)
    
    plt.tight_layout()
    return fig

# 4 besteci yan yana
fig, axes = plt.subplots(2, 2, figsize=(14, 14),
                          subplot_kw=dict(polar=True))
composers = ['Bach', 'Mozart', 'Chopin', 'Debussy']

for idx, (composer, ax) in enumerate(zip(composers, axes.flatten())):
    sample = get_representative_notes(composer, df)
    # Her biri için polar harita
    ...

plt.suptitle("Bestecilerin Harmonik Renk Çemberleri", fontsize=16)
plt.savefig('results/figures/polar_harmony_4composers.png', dpi=200)
```

### Tip 3: Self-Similarity Matrix (Müziğin Aynası)

```python
def visualize_ssm(notes, title="", window=16):
    """
    Self-similarity matrix görselleştirmesi.
    Köşegen çizgiler = tekrar eden pasajlar
    Kare bloklar = bölüm yapısı
    Bu Bach füg yapılarında muhteşem görünür.
    """
    from ..features.structural_features import self_similarity_matrix
    
    ssm = self_similarity_matrix(notes, window=window)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(ssm, cmap='viridis', aspect='auto',
                   vmin=0, vmax=1, origin='upper')
    
    ax.set_xlabel("Zaman (window index)")
    ax.set_ylabel("Zaman (window index)")
    ax.set_title(f"{title}\nSelf-Similarity Matrix", fontsize=13)
    
    plt.colorbar(im, label="Benzerlik (0-1)")
    plt.tight_layout()
    
    return fig

# Bach füg
bach_notes = get_notes('data/clean/bach_bwv851.mid')
fig = visualize_ssm(bach_notes, "Bach - Füg BWV 851")
fig.savefig('results/figures/ssm_bach_fug.png', dpi=200)
# → Köşegen çizgiler ve simetrik bloklar görmeli → BOMBA görsel

# Karşılaştır
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (composer, midi) in zip(axes, [
    ('Bach Füg', 'bach_bwv851.mid'),
    ('Mozart Sonat', 'mozart_k331.mid'),
    ('Chopin Noktürn', 'chopin_op9_2.mid')
]):
    notes = get_notes(f'data/clean/{midi}')
    ssm = self_similarity_matrix(notes)
    ax.imshow(ssm, cmap='viridis', aspect='auto')
    ax.set_title(composer)
    
plt.suptitle("Müzikal Öz-Benzerlik Yapıları", fontsize=14)
plt.savefig('results/figures/ssm_comparison.png', dpi=200)
```

### Tip 4: Network Graph (Nota Geçiş Ağı)

```python
def pitch_transition_network(notes, title="", top_n=20):
    """
    Notalar = düğüm (node)
    Geçişler = kenar (edge)
    Kenar kalınlığı = geçiş sıklığı
    Düğüm rengi = pitch class
    
    Bu her bestecinin harmonik 'DNA ağını' gösterir.
    """
    import networkx as nx
    
    pitch_classes = [n % 12 for n in notes]
    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    
    # Geçiş sayılarını hesapla
    transitions = {}
    for i in range(len(pitch_classes) - 1):
        edge = (pitch_classes[i], pitch_classes[i+1])
        transitions[edge] = transitions.get(edge, 0) + 1
    
    # En sık geçişleri al
    top_transitions = sorted(transitions.items(), 
                             key=lambda x: x[1], reverse=True)[:top_n]
    
    # Graf oluştur
    G = nx.DiGraph()
    
    # Düğümler
    for pc in range(12):
        count = pitch_classes.count(pc)
        G.add_node(pc, name=note_names[pc], frequency=count)
    
    # Kenarlar
    max_weight = max(v for _, v in top_transitions)
    for (src, dst), weight in top_transitions:
        G.add_edge(src, dst, weight=weight/max_weight)
    
    # Layout: Beşliler çemberi düzeni
    angles = {pc: (CIRCLE_OF_FIFTHS_ORDER.index(pc) / 12) * 2 * np.pi 
              for pc in range(12)}
    pos = {pc: (np.cos(angles[pc]), np.sin(angles[pc])) for pc in range(12)}
    
    # Çiz
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Renk haritası
    node_colors = [plt.cm.hsv(pc/12) for pc in range(12)]
    
    # Düğüm boyutu = kullanım sıklığı
    node_sizes = [G.nodes[pc]['frequency'] * 5 for pc in range(12)]
    
    # Kenar kalınlığı
    edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, 
                            labels={pc: note_names[pc] for pc in range(12)},
                            font_size=12, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths,
                           arrows=True, edge_color='gray',
                           connectionstyle='arc3,rad=0.1', ax=ax)
    
    ax.set_title(f"{title}\nNota Geçiş Ağı", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    return fig, G

# 4 bestecinin ağını karşılaştır
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
for ax, composer in zip(axes.flatten(), ['Bach', 'Mozart', 'Chopin', 'Debussy']):
    notes = get_representative_notes_flat(composer, df)
    plt.sca(ax)
    pitch_transition_network(notes, title=composer)

plt.suptitle("Klasik Bestecilerin Harmonik DNA Ağları", fontsize=16)
plt.savefig('results/figures/network_comparison.png', dpi=200)
```

---

## 📈 7.2 TEMPORAL EVRİM GÖRSELLERİ

```python
def musical_evolution_timeline(df):
    """
    1600-1920 arası müzikal matematiksel evrim animasyonu (veya statik seri).
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    features = [
        ('pitch_entropy', 'Pitch Entropy (Çeşitlilik)'),
        ('dissonance_index', 'Dissonans İndeksi'),
        ('fractal_dimension', 'Fraktal Boyut'),
        ('rhythmic_entropy', 'Ritmik Entropi'),
        ('repetition_index', 'Tekrar İndeksi'),
        ('interval_entropy', 'Interval Entropisi'),
    ]
    
    era_colors = {
        'Baroque': '#1f77b4',
        'Classical': '#2ca02c',
        'Romantic': '#d62728',
        'Late Romantic': '#9467bd'
    }
    
    for ax, (feat, label) in zip(axes.flatten(), features):
        for era, color in era_colors.items():
            mask = df['era'] == era
            subset = df[mask].dropna(subset=[feat, 'composition_year'])
            
            ax.scatter(subset['composition_year'], subset[feat],
                      c=color, alpha=0.4, s=20, label=era)
            
            if len(subset) > 5:
                # Trend
                z = np.polyfit(subset['composition_year'], subset[feat], 1)
                p = np.poly1d(z)
                years = np.linspace(subset['composition_year'].min(),
                                   subset['composition_year'].max(), 50)
                ax.plot(years, p(years), c=color, linewidth=2, alpha=0.8)
        
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel("Yıl")
        ax.grid(alpha=0.2)
    
    axes[0][0].legend(fontsize=9)
    
    plt.suptitle("Klasik Müziğin Matematiksel Evrimi (1600-1920)",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/figures/full_evolution.png', dpi=200, bbox_inches='tight')
```

---

## 🌐 7.3 İNTERAKTİF WEB DEMO

```python
# Plotly ile interaktif görsel
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def interactive_composer_space(df, X_umap):
    """
    UMAP uzayında interaktif besteci haritası.
    Hover: eser adı, dönem, matematiksel özellikler
    """
    
    df_plot = df.copy()
    df_plot['umap_x'] = X_umap[:, 0]
    df_plot['umap_y'] = X_umap[:, 1]
    df_plot['hover_text'] = (
        df_plot['composer'] + '<br>' +
        df_plot.get('title', df_plot['filepath'].apply(lambda x: x.split('/')[-1])) +
        '<br>Entropi: ' + df_plot['pitch_entropy'].round(3).astype(str) +
        '<br>Konsonans: ' + df_plot['consonance_score'].round(3).astype(str)
    )
    
    fig = px.scatter(
        df_plot,
        x='umap_x', y='umap_y',
        color='era',
        symbol='composer',
        hover_data=['hover_text'],
        title='Klasik Müziğin Matematiksel Haritası',
        labels={'umap_x': 'Matematiksel Boyut 1',
                'umap_y': 'Matematiksel Boyut 2'},
        color_discrete_map={
            'Baroque': '#1f77b4',
            'Classical': '#2ca02c',
            'Romantic': '#d62728',
            'Late Romantic': '#9467bd'
        }
    )
    
    fig.update_layout(
        width=900, height=700,
        template='plotly_dark',
        font_size=12,
    )
    
    fig.write_html('results/interactive_map.html')
    print("İnteraktif harita kaydedildi: results/interactive_map.html")
    return fig
```

---

## 🖼️ 7.4 PUBLICATION QUALITY FIGÜRLER

Paper için hazırlanacak figürler:

```python
# Tüm paper figürlerini toplu üret
def generate_paper_figures(df, X_pca, X_umap, feature_cols):
    
    figures = {
        'fig1_dataset_overview': plot_dataset_overview(df),
        'fig2_pca_eras': plot_pca_by_era(X_pca, df),
        'fig3_umap_composers': plot_umap_composers(X_umap, df),
        'fig4_feature_importance': plot_feature_importance(rf_model, feature_cols),
        'fig5_entropy_evolution': plot_entropy_evolution(df),
        'fig6_composer_dendrogram': plot_dendrogram(composer_centroids),
        'fig7_ssm_comparison': plot_ssm_grid(['Bach', 'Mozart', 'Chopin']),
        'fig8_human_experiment_results': plot_experiment_results(results_df),
        'fig9_entropy_beauty_correlation': plot_entropy_beauty(merged_df),
        'fig10_generated_vs_original': plot_comparison(generated_df, original_df),
    }
    
    # Tüm figürleri kaydet (300 DPI)
    for name, fig in figures.items():
        fig.savefig(f'results/figures/paper_{name}.pdf', 
                   format='pdf', bbox_inches='tight')
        fig.savefig(f'results/figures/paper_{name}.png', 
                   dpi=300, bbox_inches='tight')
    
    print(f"{len(figures)} paper figürü üretildi")
```

---

## 🎪 7.5 SANATSAL ENSTALASYON (OPSİYONEL)

Eğer projeyi bir sergi veya sunuma dönüştürmek istersen:

### Konsept: "Müziğin Geometrisi"

```
Sergi odası 4 bölüme ayrılmış:
  1. Barok Köşesi → Bach füg SSM: Simetrik, mavi tonlar
  2. Klasik Köşesi → Mozart sonat heatmap: Dengeli, yeşil
  3. Romantik Köşesi → Chopin nokturnu: Asimetrik, kırmızı
  4. Geçiş Köşesi → Debussy: Bulanık, mor

Her köşede:
  - Büyük boy görsel (printer veya projeksiyon)
  - QR kod → web demo'ya link
  - Kısa matematiksel açıklama
  - Müzik parçası çalıyor (o besteciden snippet)
```

### Sunum için "Master Visual"

```python
def create_master_visual(composers=['Bach', 'Mozart', 'Chopin', 'Debussy']):
    """
    4 bestecinin yan yana 4 farklı görseli.
    4x4 = 16 panel poster.
    """
    fig = plt.figure(figsize=(24, 24))
    
    visualizations = ['heatmap', 'polar', 'ssm', 'network']
    
    for col, composer in enumerate(composers):
        for row, viz_type in enumerate(visualizations):
            ax = fig.add_subplot(4, 4, row*4 + col + 1)
            notes = get_representative_notes_flat(composer, df)
            
            if viz_type == 'heatmap':
                render_heatmap_on_ax(ax, notes, composer if row==0 else "")
            elif viz_type == 'polar':
                render_polar_on_ax(ax, notes, visualizations[row] if col==0 else "")
            elif viz_type == 'ssm':
                render_ssm_on_ax(ax, notes)
            elif viz_type == 'network':
                render_network_on_ax(ax, notes)
    
    plt.suptitle("MÜZİĞİN GEOMETRİSİ: Klasik Bestecilerin Matematiksel DNA'sı",
                 fontsize=20, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.savefig('results/master_poster.png', dpi=200, bbox_inches='tight')
    print("Master poster oluşturuldu!")
```

---

## ⚠️ FAZ 7 RİSKLERİ

| Risk | İhtimal | Çözüm |
|------|---------|-------|
| Görseller yeterince "etkileyici" olmaz | Orta | Renk paletini ve layout'u iterate et |
| Interactive demo çok zaman alır | Orta | Sadece plotly HTML export yeterli |
| Baskı kalitesi düşük | Düşük | 300 DPI + PDF formatı |

---

## 🏁 FAZ 7 TAMAMLANDI SAYILIR WHEN

- [ ] 4 besteci için heatmap görselleri
- [ ] 4 besteci için polar harmony map
- [ ] SSM karşılaştırma görseli
- [ ] Nota geçiş ağ görseli
- [ ] Temporal evrim grafiği
- [ ] Interactive HTML demo çalışıyor
- [ ] Paper için 8-10 figür hazır (300 DPI)

---

## 🚀 FAZ 8'E GEÇİŞ KOŞULU

> Paper için gerekli tüm figürler hazır ve publication-quality → FAZ 8'e geç.

---

*Sonraki: [FAZ 8 — Paper Yazımı](FAZ_8_Paper_Yazimi.md)*
