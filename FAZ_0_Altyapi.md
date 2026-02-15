# 🔵 FAZ 0 — ALTYAPI & LİTERATÜR

> **Süre:** 3–4 hafta (esnek)  
> **Önceki Faz:** Yok (başlangıç)  
> **Sonraki Faz:** FAZ 1 — Veri Toplama

---

## 🎯 FAZ AMACI

Problemi matematiksel olarak formüle etmek, literatürü anlamak ve teknik altyapıyı kurmak.

Bu faz olmadan ilerlemek = sağlam temelsiz bina inşa etmek.

---

## ✅ FAZ ÇIKTILARI

- [ ] Çalışan Python environment
- [ ] İlk 10 MIDI başarıyla parse edilmiş
- [ ] Related work özet dokümanı
- [ ] "Biz ne yenilik getiriyoruz?" sorusunun cevabı
- [ ] Matematiksel temsil kararı verilmiş

---

## 📚 0.1 LİTERATÜR TARAMASI

### Öncelikli Kitaplar

| Kitap | Yazar | Neden Önemli |
|-------|-------|--------------|
| The Geometry of Musical Rhythm | Godfried Toussaint | Ritim ve matematik birleşimi, temel kaynak |
| Experiments in Musical Intelligence | David Cope | Algoritmik kompozisyonun babası |
| The Topos of Music | Guerino Mazzola | Matematiksel müzik teorisinin devasa kaynağı |
| Music and Probability | David Temperley | Markov modelleri ve müzik |
| Sweet Anticipation | David Huron | Entropi, beklenti ve müzikal güzellik |

### Akademik Makaleler (Öncelikli)

- **Voss & Clarke (1975)** — "1/f noise in music and speech" → fraktal yapı kanıtı
- **Narmour (1992)** — Implication-realization modeli → melodic expectation
- **Lerdahl & Jackendoff** — Generative Theory of Tonal Music (GTTM) → yapısal analiz
- **Liu et al. (2013)** — Complexity and music appreciation korelasyonu

### Taranacak Kaynaklar

```
Google Scholar aramaları:
- "entropy classical music computational"
- "fractal dimension music analysis"
- "mathematical fingerprint composer identification"
- "music information retrieval classical"
- "harmonic tension mathematics"

ISMIR proceedings 2018–2024
Computer Music Journal son 5 yıl
```

### Related Work Özet Tablosu (dolduracaksın)

| Çalışma | Yöntem | Dataset | Bulgu | Bizden Farkı |
|---------|--------|---------|-------|--------------|
| ... | ... | ... | ... | ... |

### Bizi Farklılaştıran Şeyler
1. Çok katmanlı feature (pitch + interval + harmoni + yapı + fraktal)
2. Geniş multi-composer, multi-era dataset (1500+ eser)
3. Matematiksel model → generatif üretim → insan deneyi zinciri
4. "Güzellik metrikleri" formülizasyonu denemesi

---

## 🧮 0.2 MATEMATİKSEL TEMSİL KARARI

### Müziği Hangi Katmanlarda Temsil Edeceğiz?

#### Katman 1: Pitch Space
```
MIDI note number → integer (0–127)
Pitch class → mod 12 (C=0, C#=1, D=2, ... B=11)
Kullanım: Hangi notalar ne sıklıkla?
```

#### Katman 2: Interval Space
```
Interval = note[i+1] - note[i]  (semitones)
Range: -12 ile +12 arası
Kullanım: Notadan notaya nasıl atlıyor?
```

#### Katman 3: Harmonic Space
```
Chord = eş zamanlı nota seti
Tonal tension = toniğe olan uzaklık
Kullanım: Harmonik gerilim ve çözülme
```

#### Katman 4: Temporal Space
```
Duration = nota süresi (quarter note = 1.0)
Beat position = ölçüdeki konum
Tempo = BPM
Kullanım: Ritim ve zamanlama kalıpları
```

#### Katman 5: Structural Space
```
Self-similarity matrix = pasajların birbirine benzeme oranı
Section boundaries = yapısal segmentasyon
Fractal dimension = ölçek-bağımsız öz-benzerlik
Kullanım: Makro form analizi
```

### Başlangıç Kararı
Faz 1–2'de **Katman 1 + 2**'den başla.  
Faz 2'nin ortasında **Katman 3 + 4** ekle.  
Faz 3'te **Katman 5** (fraktal, topoloji) ekle.

---

## ⚙️ 0.3 TEKNİK ALTYAPI KURULUMU

### Environment Setup

```bash
# Conda ile izole environment
conda create -n music_math python=3.10
conda activate music_math

# Temel kütüphaneler
pip install music21
pip install librosa
pip install pretty_midi
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install umap-learn
pip install networkx
pip install jupyter jupyterlab
pip install plotly

# İleri seviye (Faz 3-4 için)
pip install torch
pip install gudhi

# Test
python -c "import music21, librosa, sklearn; print('Setup OK!')"
```

### Proje Dizin Yapısı

```bash
mkdir -p music_math_project/{data/raw,data/clean,notebooks,src,results/{figures,generated_midi,stats},paper}

# Dizin açıklamaları:
# data/raw/        → Ham indirilen MIDI dosyaları
# data/clean/      → Normalize edilmiş MIDIlar
# notebooks/       → Jupyter analiz notebook'ları
# src/             → Python modülleri
# results/         → Analiz çıktıları
# paper/           → Paper taslakları
```

### İlk Test: Parse ve Görselleştir

```python
# test_faz0.py
from music21 import converter, note, chord
import numpy as np
import matplotlib.pyplot as plt

def parse_midi(filepath):
    score = converter.parse(filepath)
    notes = []
    for element in score.flatten().notes:
        if isinstance(element, note.Note):
            notes.append(element.pitch.midi)
        elif isinstance(element, chord.Chord):
            for n in element.notes:
                notes.append(n.pitch.midi)
    return notes

def plot_pitch_histogram(notes, title=""):
    pitch_classes = [n % 12 for n in notes]
    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    counts = [pitch_classes.count(i) for i in range(12)]
    freqs = np.array(counts) / sum(counts)
    
    entropy = -np.sum(freqs[freqs > 0] * np.log2(freqs[freqs > 0]))
    
    plt.figure(figsize=(10, 4))
    plt.bar(note_names, freqs, color='steelblue', alpha=0.8)
    plt.title(f"{title} | Entropy: {entropy:.3f} bits")
    plt.ylabel("Oran")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return entropy

# Kullanım
notes = parse_midi("bach_prelude.mid")
entropy = plot_pitch_histogram(notes, "Bach BWV 846")
print(f"Entropy: {entropy:.3f}")
```

---

## 🎓 0.4 MATEMATİKSEL KAVRAMLAR HATIRLATICI

### Shannon Entropy

Sistemin rastgelelik / çeşitlilik ölçüsü:

```
H(X) = -Σ p(x) · log₂(p(x))
```

**Müzikte:**
- `H = 0.0` → Tek nota tekrar eder (maksimum sıkıcı)
- `H = 3.58` → 12 nota eşit kullanılır (maksimum kaotik)
- `H ≈ 2.0–2.8` → Klasik müziğin sweet spot'u (hipotez)

### Markov Zincirleri

```
P(nota_t+1 | nota_t) = Geçiş matrisi
Boyut: 12×12 (pitch class) veya 128×128 (tam MIDI)
```

**Müzikte:**  
Her bestecinin benzersiz geçiş matrisi = matematiksel parmak izi.

### Interval Vector

Atonalite teorisinden:
```
Bir akor/dizinin 6-boyutlu interval içeriği vektörü
<m2, M2, m3, M3, P4, TT>
```

### Fractal Dimension (Box-counting)

```
D = log(N) / log(1/r)
D ≈ 1.0 → Düz çizgi (çok basit)
D ≈ 1.5 → Karmaşık ama yapılı (müzikal sweet spot?)
D ≈ 2.0 → Tamamen rastgele
```

---

## ⚠️ FAZ 0 RİSKLERİ

| Risk | İhtimal | Çözüm |
|------|---------|-------|
| Literatür çok zaman alır | Orta | Abstract + conclusion oku, detaya sonra dön |
| music21 kurulum sorunları | Düşük | `pip install music21` genellikle sorunsuz |
| Matematiksel temsil yanlış seçilir | Orta | Çok katman koy, sonradan elenebilir |
| MIDI dosyası bozuk çıkabilir | Orta | İlk 10'u tanınmış kaynaklardan seç |

---

## 🏁 FAZ 0 TAMAMLANDI SAYILIR WHEN

- [ ] `conda activate music_math` ve tüm importlar çalışıyor
- [ ] 10 MIDI parse edilmiş
- [ ] 10 eser için pitch histogram görselleştirilmiş
- [ ] 10 eser için entropy değerleri hesaplanmış
- [ ] Entropiler birbirinden farklı mı? (Evet → FAZ 1'e geç)
- [ ] Related work özeti yazılmış (5-10 kaynak)
- [ ] Proje dizin yapısı oluşturulmuş

---

## 🚀 FAZ 1'E GEÇİŞ KOŞULU

> 10 farklı eser için hesaplanan entropy değerleri arasında **anlamlı fark** (std > 0.2) görülüyorsa → FAZ 1'e geç.

---

*Sonraki: [FAZ 1 — Veri Toplama](FAZ_1_Veri_Toplama.md)*
