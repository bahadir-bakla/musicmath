# 🎵 KLASİK MÜZİĞİN MATEMATİKSEL DNA'SI
## Genel Proje Planı

---

## 🎯 PROJE VİZYONU

> Klasik müziğin bilinçsiz matematiksel yapılarını tersine mühendislik yöntemiyle keşfetmek, bu yapıları formal modele dökmek, modelden yeni müzik üretmek ve üretilen müziğin insan algısı üzerindeki etkisini ölçmek.

---

## 🔬 ANA ARAŞTIRMA SORULARI

| # | Araştırma Sorusu | Faz |
|---|-----------------|-----|
| RQ1 | Klasik müzikte matematiksel olarak tespit edilebilir yapısal pattern'ler var mı? | Faz 2-3 |
| RQ2 | Bu pattern'ler dönem/besteci bazında kümeleniyor mu? | Faz 3 |
| RQ3 | Bestecilerin bilinçsiz matematiksel "imzaları" var mı? | Faz 3-4 |
| RQ4 | Müzikal "güzellik" ile korelasyonlu matematiksel metrikler var mı? | Faz 3-4 |
| RQ5 | Matematiksel modelden üretilen müzik insan algısında orijinale yakın mı? | Faz 6 |

---

## 🗺️ FAZ HARİTASI

```
FAZ 0: Altyapı & Literatür        [3-4 hafta]
         ↓
FAZ 1: Veri Toplama & Temizleme   [4-6 hafta]
         ↓
FAZ 2: Feature Engineering        [6-8 hafta]
         ↓
FAZ 3: Matematiksel Pattern Keşfi [8-10 hafta]
         ↓
FAZ 4: Matematiksel Model         [4-6 hafta]
         ↓
FAZ 5: Generatif Müzik Üretimi    [4-6 hafta]
         ↓
FAZ 6: İnsan Değerlendirme        [6-8 hafta]
         ↓
FAZ 7: Görselleştirme             [4 hafta]
         ↓
FAZ 8: Paper Yazımı               [6-8 hafta]

TOPLAM: ~45-60 hafta (~1.0-1.5 yıl)
```

---

## 📦 MATEMATİKSEL TEMSİL KATMANLARI

Müziği 6 katmanlı matematiksel uzayda temsil ediyoruz:

```
Katman 1: Frekans / Pitch Domain
Katman 2: Interval Domain
Katman 3: Harmoni Domain
Katman 4: Spektral Domain
Katman 5: Yapısal / Fraktal Domain
Katman 6: Temporal / Ritmik Domain
```

Her eser → **~80-100 boyutlu feature vektörü**

---

## 🎼 DATASET STRATEJİSİ

### Hedef: 1500-2000 Eser

| Dönem | Besteciler | Hedef Eser |
|-------|-----------|------------|
| Barok (1600-1750) | Bach, Vivaldi, Handel, Telemann | 230 |
| Klasik (1750-1820) | Mozart, Haydn, Beethoven | 280 |
| Romantik (1820-1910) | Chopin, Brahms, Schumann, Tchaikovsky | 230 |
| Geç Romantik (1880-1920) | Debussy, Rachmaninoff, Ravel | 100 |

**Öncelik:** Solo piyano + Solo keman  
**Format:** MIDI  
**Kaynak:** MAESTRO Dataset, IMSLP, MuseScore

---

## ⚗️ KEŞFEDİLMESİ BEKLENEN BULGULAR

### %80 Olasılıkla:
- Dönem bazında belirgin matematiksel clustering
- Besteci bazında ayırt edici feature profile'ları
- Entropi vs kompleksite korelasyonu

### %40 Olasılıkla:
- Fibonacci / Altın oran sinyali
- Universal "güzellik metrikleri"
- Matematiksel evrim zaman serisi (Barok → Romantik)

### %20 Olasılıkla (ama olursa BOMBA 💥):
- Besteci matematiksel "imza algoritması"
- Müzikal filogenetik ağaç
- Generatif modelin insan testi geçmesi

---

## 🛠️ TEKNİK STACK

```python
# Müzik Analizi
music21          # MIDI parsing, müzik teorisi
librosa          # Audio analiz, spektral features
pretty_midi      # MIDI manipulation

# Veri & ML
numpy            # Sayısal hesaplama
pandas           # Veri yönetimi
scikit-learn     # ML, clustering, PCA
torch            # Deep learning (Faz 5+)
umap-learn       # Dimensionality reduction

# Topoloji & Grafik
gudhi            # Persistent homology
networkx         # Graf teorisi analizi

# Görselleştirme
matplotlib       # Temel görselleştirme
seaborn          # İstatistiksel görselleştirme
plotly           # İnteraktif görselleştirme
```

---

## 📝 HEDEF PAPER

**Başlık:** *"Mathematical Patterns in Classical Music: A Computational Analysis of Composer Signatures and Generative Modelling"*

**Hedef Konferanslar:**
- ISMIR (International Society for Music Information Retrieval)
- ICMC (International Computer Music Conference)

**Hedef Dergiler:**
- Computer Music Journal
- Journal of New Music Research

---

## 🚩 KRİTİK CHECKPOINT'LER

| Checkpoint | Zaman | Soru |
|-----------|-------|------|
| CP1 | Hafta 4 | Pitch feature'ları bestecileri ayırıyor mu? |
| CP2 | Hafta 10 | Feature matrix anlamlı clustering veriyor mu? |
| CP3 | Hafta 18 | Matematiksel model çalışıyor mu? |
| CP4 | Hafta 24 | Generated müzik kulağa geçerli mi? |
| CP5 | Hafta 30 | İnsan testi istatistiksel anlamlı mı? |

---

## 👥 MÜZİKOLOG DANIŞMAN KULLANIMI

Müzik teorisi doğrulama için kritik noktalarda danışılacak:

1. **Hafta 4 sonu** — Feature seçimlerinin müzikal geçerliliği
2. **Hafta 10 sonu** — Clustering sonuçlarının yorumlanması
3. **Faz 5 sonu** — Generated MIDI'ların kalite değerlendirmesi
4. **Paper yazımı** — Domain-specific terminoloji ve related work

---

*Bu doküman genel yol haritasıdır. Her faz için detaylı planlar ayrı dosyalarda mevcuttur.*
