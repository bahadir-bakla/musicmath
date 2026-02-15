# Seçenek D – Sırayla Çalıştırma Rehberi

Tüm analizleri sırayla çalıştırmak için aşağıdaki adımları izle. Proje kökünde `python` ve `pip` erişimin olduğunu varsayıyoruz.

---

## Adım 0: Ortam (bir kez)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

---

## A) Mevcut 6 eserle matematiksel pattern testi

```bash
python scripts/mathematical_patterns.py
```

**Çıktılar:**
- `results/stats/mathematical_patterns.csv` – Asal sayı, Golden Ratio, Fibonacci metrikleri
- `results/figures/composer_prime_profile.png` – Bestecilere göre asal sayı kullanımı
- `results/figures/golden_ratio_analysis.png` – Climax pozisyonu, Fibonacci bölüm oranı

**Konsol:** Bestecilere göre asal sayı / Fibonacci özeti.

---

## B) Dataset’i genişlet (50–100+ eser)

### B1) MIDI indir

```bash
python scripts/download_midi_dataset.py \
  --composers bach,mozart,chopin,beethoven,debussy,liszt \
  --max-per-composer 20
```

İndirilen dosyalar `data/raw/piano_midi/<besteci>/` altına gider.

### B2) Pipeline (metadata + temizlik + feature matrix)

```bash
python run_pipeline.py
```

Bunu çalıştırdıktan sonra `metadata_clean.csv` ve `results/stats/feature_matrix.csv` güncellenir.  
**Not:** `run_pipeline.py` içinde besteci/era ataması dosya yoluna göre yapılıyor; yeni besteciler (liszt, beethoven) için `run_pipeline.py` veya metadata’da `infer_composer` / `infer_era` fonksiyonlarını genişletmen gerekebilir.

### B3) Tüm analizler

```bash
python analyze_features.py              # PCA, mesafe matrisi, Bach Markov örnek
python scripts/advanced_analysis.py    # Clustering, t-SNE, UMAP, feature importance
python scripts/mathematical_patterns.py # Asal / Fibonacci (geniş dataset ile)
```

---

## C) İstatistiksel testler

Matematiksel pattern’leri en az bir kez ürettikten sonra:

```bash
python scripts/statistical_tests.py
```

**Ne yapar:**
- **ANOVA:** Dönemlere / bestecilere göre feature farkı (pitch_entropy, consonance, interval_entropy, rhythmic_entropy, interval_prime_density).
- **t-test:** Barok vs Romantik pitch_entropy.
- **Mozart Fibonacci:** Fibonacci section ratio’nun 0.5’ten anlamlı farkı (one-sample t-test).

`*` işareti p &lt; 0.05 anlamlı farkı gösterir.

---

## Kısa özet (6 eserle hızlı tur)

```bash
python run_pipeline.py
python analyze_features.py
python scripts/advanced_analysis.py
python scripts/mathematical_patterns.py
python scripts/statistical_tests.py
```

---

## Çıktı dizinleri

| Dizin / dosya | İçerik |
|---------------|--------|
| `results/figures/` | PCA, mesafe heatmap, dendrogram, t-SNE, UMAP, feature importance, asal sayı / Golden Ratio grafikleri |
| `results/stats/` | `feature_matrix.csv`, `mathematical_patterns.csv` |
| `results/generated_midi/` | Bach tarzı örnek MIDI |

---

## Notlar

- **Mesafe matrisi:** `analyze_features.py` standardize edilmiş feature’larla Euclidean mesafe kullanıyor; büyük sayı patlaması olmaz.
- **t-SNE:** Örnek sayısı 4’ten azsa `advanced_analysis.py` t-SNE’i atlar.
- **mathematical_patterns.py:** `music_math.analysis.prime_harmony` ve `golden_ratio` modüllerine bağımlı; paket kurulu olmalı (`pip install -e .`).
