# 🎵 Quick Start: Piano MIDI Pipeline

Bu döküman, küçük piano_midi dataset'i ile pipeline'ı lokal makinenizde çalıştırmanız için hazırlanmıştır.

## 📋 Gereksinimler

- Python 3.10+
- Tüm bağımlılıklar (`pyproject.toml`'deki paketler)

## 🚀 Kurulum

### 1. Virtual Environment Oluştur

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

### 2. Bağımlılıkları Yükle

```bash
pip install -U pip
pip install -e .[dev]
```

Bu komut şu paketleri yükler:
- `music21`, `librosa`, `pretty_midi` → MIDI işleme
- `numpy`, `pandas`, `scikit-learn` → Veri analizi ve ML
- `matplotlib`, `seaborn`, `plotly` → Görselleştirme
- `torch`, `umap-learn`, `gudhi`, `networkx` → İleri düzey modelleme
- `pytest`, `black`, `ruff`, `mypy` → Geliştirme araçları

## 📂 Mevcut Dataset

`data/raw/piano_midi/` altında şu 6 MIDI dosyası bulunuyor:

```
bach_846.mid          → J.S. Bach (Baroque)
bach_847.mid          → J.S. Bach (Baroque)
mz_330_1.mid          → W.A. Mozart (Classical)
chpn_op10_e12.mid     → F. Chopin (Romantic)
chpn-p4.mid           → F. Chopin (Romantic)
deb_clai.mid          → C. Debussy (Late Romantic)
```

## 🎯 Pipeline Çalıştırma

### Adım 1: Tam Pipeline (Metadata → Clean → Features)

```bash
python run_pipeline.py
```

Bu script şunları yapar:
1. `data/raw/` altındaki tüm MIDI dosyalarını tarar
2. `metadata.csv` oluşturur (composer, era, source bilgileriyle)
3. Kalite filtresi uygular → `metadata_clean.csv`
4. Temiz dosyaları `data/clean/` altına kopyalar
5. **Feature matrix** oluşturur → `results/stats/feature_matrix.csv`

**Beklenen Çıktı:**
```
Feature matrix shape: (6, ~80-100)
```

Her satır bir eser, her sütun bir feature:
- **Kimlik**: `filepath`, `composer`, `era`, `form`
- **Pitch**: `pitch_entropy`, `tonal_center_strength`, `pitch_range`, `pc_0` … `pc_11`
- **Interval**: `interval_entropy`, `directional_bias`, `step_ratio`, `leap_ratio`
- **Harmony**: `consonance_score`, `dissonance_index`, `duration_variance`
- **Rhythm**: `rhythmic_entropy`, `note_density`, `tempo_variance`
- **Structural**: `repetition_index`, `fractal_dimension`, `unique_pitch_classes`
- **Spectral**: `spectral_centroid`, `spectral_entropy`, `dominant_frequency`

### Adım 2: PCA Görselleştirmesi

```bash
python analyze_pca.py
```

Bu script:
1. `feature_matrix.csv`'yi yükler
2. 2-boyutlu PCA uygular
3. Dönemlere göre renklendirilmiş scatter plot oluşturur
4. `results/figures/pca_era_separation.png` olarak kaydeder

**Beklenen Sonuç:**

Genelde şu şekilde bir ayrışma görülür:
- **Bach (Baroque)** → Düşük pitch/interval entropy, yüksek konsonans
- **Mozart (Classical)** → Orta düzey karmaşıklık, dengeli yapı
- **Chopin (Romantic)** → Yüksek interval entropy, daha geniş pitch range
- **Debussy (Late Romantic)** → Yüksek dissonans, spektral zenginlik

Bu, 6 eserlik küçük dataset'te bile **dönemsel matematiksel imzaların** var olduğunu gösterir.

## 📊 Sonuç Dosyaları

Başarılı çalıştırma sonrası şu dosyalar oluşacak:

```
music_analysisi/
├── metadata.csv                           # Ham metadata
├── metadata_clean.csv                     # Kalite filtresinden geçen metadata
├── data/
│   └── clean/                             # Temiz MIDI dosyaları (6 adet)
│       ├── bach_846.mid
│       ├── bach_847.mid
│       └── ...
└── results/
    ├── stats/
    │   └── feature_matrix.csv             # 6 x ~80-100 feature matrix
    └── figures/
        └── pca_era_separation.png         # PCA görselleştirmesi
```

## 🔍 İleri Analizler

Feature matrix hazır olduktan sonra yapabilecekleriniz:

### 1. Besteci Mesafe Matrisi

```python
import pandas as pd
from scipy.spatial.distance import pdist, squareform

df = pd.read_csv("results/stats/feature_matrix.csv")
meta_cols = ["filepath", "composer", "era", "form"]
feat_cols = [c for c in df.columns if c not in meta_cols]

# Bestecilere göre ortalama feature vektörü
composer_features = df.groupby('composer')[feat_cols].mean()

# Euclidean mesafe matrisi
distances = squareform(pdist(composer_features.values, metric='euclidean'))
dist_df = pd.DataFrame(distances, 
                       index=composer_features.index, 
                       columns=composer_features.index)
print(dist_df)
```

### 2. Mini Markov Model ile Generatif Deneme

```python
from music_math.model.markov import train_pitch_markov
from music_math.generation.generator import generate_from_markov

# Bach üzerinden öğren
bach_files = df[df['composer'] == 'Bach']['filepath'].tolist()
markov_model = train_pitch_markov(bach_files, order=2)

# Bach tarzında yeni sekans üret
new_pitches = generate_from_markov(markov_model, length=100)
print(new_pitches)
```

### 3. UMAP ile 2D Projeksiyon

```python
from umap import UMAP
import matplotlib.pyplot as plt

X = df[feat_cols].fillna(0).values
reducer = UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=df['era'].astype('category').cat.codes)
plt.colorbar(label='Era')
plt.title('UMAP: Music Feature Space')
plt.show()
```

## 🐛 Sorun Giderme

### Hata: `ModuleNotFoundError: No module named 'music21'`

```bash
pip install -e .[dev]
```

### Hata: `pandas import segfault (exit 139)`

Bu, bazı ortamlarda (özellikle sanal makineler) pandas/numpy'ın C bağımlılıkları ile ilgili bir sorundur. Çözüm:

```bash
# Conda kullanıyorsanız:
conda install pandas numpy scikit-learn

# pip kullanıyorsanız, sistem paketlerini kullanmayı deneyin:
pip install --no-binary :all: pandas
```

### Kalite filtresi tüm dosyaları eleerse

`music_math/data/quality.py` içindeki `QualityConfig` parametrelerini düzenleyin:

```python
@dataclass
class QualityConfig:
    min_notes: int = 20              # 50 yerine 20
    max_duration_quarter_length: float = 3000.0  # daha yüksek limit
```

## 📚 Daha Fazla Bilgi

- **Genel Proje Planı**: `GENEL_PROJE_PLANI.md`
- **Faz 1 (Veri Toplama)**: `FAZ_1_Veri_Toplama.md`
- **Faz 2 (Feature Engineering)**: `FAZ_2_Feature_Engineering.md`
- **Faz 4 (Matematiksel Model)**: `FAZ_4_Matematiksel_Model.md`

---

**Hazırlayan**: Kombai AI Assistant  
**Tarih**: 2026-02-12  
**Versiyon**: 0.1.0
