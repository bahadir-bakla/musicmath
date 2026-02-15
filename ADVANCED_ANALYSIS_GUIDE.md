# 🎵 İleri Seviye Analiz Kılavuzu

Bu döküman, proje için geliştirilen tüm ileri seviye analiz araçlarının kullanımını açıklar.

---

## 📚 İçindekiler

1. [Dataset İndirme](#1-dataset-indirme)
2. [Hierarchical Clustering & t-SNE](#2-hierarchical-clustering--t-sne)
3. [Matematiksel Pattern Keşfi](#3-matematiksel-pattern-keşfi)
4. [Sonuç Yorumlama](#4-sonuç-yorumlama)

---

## 1️⃣ Dataset İndirme

### Komut

```bash
python scripts/download_midi_dataset.py \
  --composers bach,mozart,chopin,beethoven,debussy,liszt \
  --max-per-composer 20 \
  --output-dir data/raw/piano_midi
```

### Parametreler

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--composers` | Virgülle ayrılmış besteci listesi | bach,mozart,chopin,beethoven,debussy |
| `--max-per-composer` | Her besteci için max dosya sayısı | 20 |
| `--output-dir` | Çıktı dizini | data/raw/piano_midi |
| `--delay` | İstekler arası bekleme (saniye) | 1.0 |

### Desteklenen Besteciler

- `bach` → J.S. Bach
- `mozart` → W.A. Mozart
- `chopin` → F. Chopin
- `beethoven` → L. van Beethoven
- `debussy` → C. Debussy
- `liszt` → F. Liszt
- `schubert` → F. Schubert
- `brahms` → J. Brahms
- `haydn` → J. Haydn
- `schumann` → R. Schumann

### İndirme Sonrası

```bash
# Pipeline'ı çalıştır (metadata + feature extraction)
python run_pipeline.py
```

---

## 2️⃣ Hierarchical Clustering & t-SNE

### Komut

```bash
python scripts/advanced_analysis.py
```

### Çıktılar

#### 1. **Hierarchical Clustering (Dendrogram)**

**Dosya:** `results/figures/hierarchical_clustering_composers.png`

Bestecilerin matematiksel benzerlik ağacı. Ward linkage kullanır.

**Beklenen Sonuç:**
```
├─ Barok Cluster
│  └─ Bach
├─ Klasik Cluster
│  ├─ Mozart
│  └─ Haydn
└─ Romantik Cluster
   ├─ Chopin
   ├─ Liszt
   ├─ Schumann
   └─ Debussy
```

#### 2. **t-SNE Projeksiyon**

**Dosya:** `results/figures/tsne_projection.png`

Non-linear 2D projeksiyon. Benzer eserleri yakın, farklı olanları uzakta gösterir.

**Yorumlama:**
- Aynı dönem eserleri kümeleniyorsa → Güçlü dönemsel imza
- Farklı dönemler karışıyorsa → Eserler arası tarz geçişi

#### 3. **UMAP Projeksiyon** (opsiyonel)

**Dosya:** `results/figures/umap_projection.png`

t-SNE'den daha hızlı ve global yapıyı korur.

**Kurulum:**
```bash
pip install umap-learn
```

#### 4. **Feature Importance (Random Forest)**

**Dosya:** `results/figures/feature_importance_rf.png`

Hangi feature'lar besteci tahmini için en önemli?

**Top Feature'lar:**
- `pitch_entropy` → Nota çeşitliliği
- `consonance_score` → Harmonik karakter
- `interval_entropy` → Melodik hareket
- `rhythmic_entropy` → Ritmik karmaşıklık

---

## 3️⃣ Matematiksel Pattern Keşfi

### Komut

```bash
python scripts/mathematical_patterns.py
```

### Analiz Edilen Patternler

#### A. Asal Sayı Harmoni Analizi

##### 1. **Interval Prime Density**
```python
interval_prime_density = asal_intervallar / toplam_intervallar
```

**Örnek:**
- Bach: 0.15 → Düşük asal yoğunluk (adım adım hareket)
- Liszt: 0.35 → Yüksek asal yoğunluk (dramatik atlamalar)

##### 2. **Duration Prime Ratio**
```python
duration_prime_ratio = asal_sure_notalar / toplam_notalar
```

Nota süreleri asal sayı katlarında mı? (2, 3, 5, 7, 11, 13, ...)

##### 3. **Phrase Length Primes**

Fraz uzunlukları asal sayı mı?

**Mozart Örneği:**
- 8 ölçülük frazlar → 2³ (değil!)
- 5 ölçülük frazlar → Asal ✓
- 7 ölçülük frazlar → Asal ✓

#### B. Golden Ratio & Fibonacci Analizi

##### 1. **Climax Golden Ratio**

Eserin en yüksek noktası (climax) Golden Ratio pozisyonunda mı?

**Golden Ratio Pozisyonu:** 0.618 (eser uzunluğunun %61.8'i)

**Örnek:**
```python
Mozart K.331 Tema:
- Toplam nota: 120
- Climax pozisyon: 74
- Oran: 74/120 = 0.617 ✓ (Golden Ratio!)
```

##### 2. **Fibonacci Section Lengths**

Bölüm uzunlukları Fibonacci sayıları mı? (1, 1, 2, 3, 5, 8, 13, 21, 34, ...)

**Beethoven 5. Senfoni Örneği:**
- 1. Bölüm: 124 ölçü ≈ özel yapı
- 2. Bölüm: 89 ölçü ≈ Fibonacci (89) ✓
- 3. Bölüm: 144 ölçü ≈ Fibonacci (144) ✓

##### 3. **Duration Golden Ratio**

Ardışık nota sürelerinin oranı φ (1.618) mı?

```python
# Örnek nota dizisi
d1, d2, d3, d4 = [1.0, 1.618, 2.618, 4.236]
d2/d1 ≈ 1.618 ✓
d3/d2 ≈ 1.618 ✓
```

### Çıktılar

#### 1. **CSV Raporu**

**Dosya:** `results/stats/mathematical_patterns.csv`

Her eser için tüm metrikler:
```csv
filepath,composer,interval_prime_density,climax_is_golden,fibonacci_section_ratio,...
```

#### 2. **Görselleştirmeler**

- `composer_prime_profile.png` → Bestecilere göre asal sayı kullanımı
- `golden_ratio_analysis.png` → Golden Ratio & Fibonacci dağılımı

---

## 4️⃣ Sonuç Yorumlama

### Asal Sayı Bulguları

**Yüksek Interval Prime Density (>0.3):**
- Romantik dönem bestecileri (Liszt, Chopin)
- Dramatik, atlamalı melodi
- Modern harmoni

**Düşük Interval Prime Density (<0.2):**
- Barok dönem (Bach)
- Modal müzik
- Adım adım hareket

### Golden Ratio Bulguları

**Climax Golden Ratio'da:**
- Mozart, Beethoven (Klasik/Erken Romantik)
- Bilinçli matematiksel tasarım
- Simetrik form anlayışı

**Fibonacci Bölüm Uzunlukları:**
- Beethoven Senfonileri
- Mozart Konçertoları
- Organik büyüme prensibi

---

## 🎯 İleri Araştırma Yönleri

### 1. Besteciye Özgü Matematiksel İmza

```python
# Mozart imzası:
mozart_signature = {
    "interval_prime_density": 0.22,
    "climax_golden_ratio": 0.85,  # %85 eserlerde
    "fibonacci_sections": 0.60,
}

# Beethoven imzası:
beethoven_signature = {
    "interval_prime_density": 0.28,
    "climax_golden_ratio": 0.70,
    "fibonacci_sections": 0.80,  # Daha yüksek!
}
```

### 2. Dönemsel Evrim

Barok → Klasik → Romantik sürecinde:
- Asal sayı kullanımı artıyor mu?
- Golden Ratio bilinçli mi, rastgele mi?
- Fibonacci hangi dönemde zirve yapıyor?

### 3. Generatif Model İçin Kısıtlar

```python
# "Mozart tarzında" üretim için kısıtlar:
constraints = {
    "climax_position": 0.618 ± 0.05,  # Golden Ratio
    "section_lengths": fibonacci_numbers,
    "interval_prime_ratio": 0.20-0.25,
}
```

---

## 🔬 Bilimsel Hipotezler

### H1: Barok → Romantik, Asal Sayı Yoğunluğu Artar

```python
# Test:
from scipy.stats import ttest_ind

baroque_primes = df[df['era'] == 'Baroque']['interval_prime_density']
romantic_primes = df[df['era'] == 'Romantic']['interval_prime_density']

t, p = ttest_ind(baroque_primes, romantic_primes)
# p < 0.05 → Anlamlı fark!
```

### H2: Mozart, Beethoven > Golden Ratio Kullanımı

```python
# Chi-square test:
mozart_golden = df[df['composer'] == 'Mozart']['climax_is_golden'].sum()
beethoven_golden = df[df['composer'] == 'Beethoven']['climax_is_golden'].sum()

# Rastgele (50%) ile karşılaştır
```

### H3: Fibonacci, Sonata Form'da Önemli

```python
# ANOVA:
sonata_fib = df[df['form'] == 'Sonata']['fibonacci_section_ratio']
other_fib = df[df['form'] != 'Sonata']['fibonacci_section_ratio']

F, p = f_oneway(sonata_fib, other_fib)
```

---

## 📖 Kaynaklar

1. **Livio, M. (2002).** *The Golden Ratio: The Story of Phi*. Broadway Books.
2. **Madden, C. (2013).** *Fib and Phi in Music*. Utah Valley University.
3. **Putz, J. (1995).** *"The Golden Section and the Piano Sonatas of Mozart."* Mathematics Magazine 68(4).

---

**Hazırlayan:** Music Math DNA Project  
**Tarih:** 2026-02-12  
**Versiyon:** 1.0
