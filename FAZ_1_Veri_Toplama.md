# 🟢 FAZ 1 — VERİ TOPLAMA & İLK EKSPLORASYONa

> **Süre:** 4–6 hafta (esnek)  
> **Önceki Faz:** FAZ 0 — Altyapı  
> **Sonraki Faz:** FAZ 2 — Feature Engineering

---

## 🎯 FAZ AMACI

Temiz, zengin, dengeli bir MIDI dataset oluşturmak. Bu dataset projenin temelidir. Kötü veri = kötü sonuç.

---

## ✅ FAZ ÇIKTILARI

- [ ] 1500–2000 clean MIDI dosyası
- [ ] `metadata.csv` (composer, year, era, form, key, instrumentation)
- [ ] EDA notebook (görsel analizler)
- [ ] Data quality raporu
- [ ] İlk "bu yaklaşım işe yarıyor mu?" sorusunun cevabı

---

## 📦 1.1 DATASET STRATEJİSİ

### Temel İlkeler

1. **Dönem dengesi** — Her dönemden yeterli eser
2. **Besteci dengesi** — Bir besteci diğerlerine baskın olmasın
3. **Form çeşitliliği** — Fügden sonat'a, prelüdden konçerto'ya
4. **Enstrüman temizliği** — Başlangıçta solo piyano öncelikli
5. **Kalite kontrolü** — Bozuk / yanlış etiketli MIDI'ları ele

### Hedef Dataset Dağılımı

#### Barok Dönem (1600–1750) — Hedef: 230 eser
```
Johann Sebastian Bach      → 100 eser
  - Well-Tempered Clavier (48 prelüd + füg)
  - İnvensiyonlar ve Sinfoniler
  - Partitalar
  - Çello süitleri (MIDI olarak)
  
Antonio Vivaldi            → 50 eser
  - The Four Seasons
  - Çeşitli konçertolar
  
George Frideric Handel     → 50 eser
  - Keyboard suites
  - Piyano eserleri
  
Georg Philipp Telemann     → 30 eser
  - Fantaziler
```

#### Klasik Dönem (1750–1820) — Hedef: 280 eser
```
Wolfgang Amadeus Mozart    → 100 eser
  - Piyano sonatları (18 sonat)
  - Piyano konçertoları
  - Varyasyonlar
  - Rondo'lar
  
Joseph Haydn               → 80 eser
  - Piyano sonatları
  - String quartet'ler (MIDI)
  
Ludwig van Beethoven       → 100 eser
  - Piyano sonatları (32 sonat = hazır!)
  - Bagateller
  - Varyasyonlar
```

#### Romantik Dönem (1820–1910) — Hedef: 230 eser
```
Frédéric Chopin            → 80 eser
  - Noktürnler (21 adet)
  - Etütler (27 adet)
  - Prelüdler (24 adet)
  - Balladlar

Johannes Brahms            → 60 eser
  - İntermezzo'lar
  - Kapriçler

Robert Schumann            → 50 eser
  - Kinderszenen
  - Papillons
  - Kreisleriana

Pyotr Ilyich Tchaikovsky   → 40 eser
  - The Seasons
  - Piyano parçaları
```

#### Geç Romantik / Empresyonizm (1890–1920) — Hedef: 140 eser
```
Claude Debussy             → 50 eser
  - Prelüdler (24 adet)
  - Images
  - Children's Corner

Sergei Rachmaninoff        → 50 eser
  - Prelüdler
  - Moments Musicaux

Maurice Ravel              → 40 eser
  - Gaspard de la Nuit
  - Sonatine
```

**TOPLAM: ~880 solo piyano eseri (Faz 1 hedefi)**

> Not: Sonraki adımda keman, çello ekleyerek 1500+'a çıkarılır.

---

## 🌐 1.2 VERİ KAYNAKLARI

### Kaynak 1: MAESTRO Dataset (En İyi Başlangıç!)
```
URL: https://magenta.tensorflow.org/datasets/maestro
Format: MIDI + metadata JSON
Boyut: 200+ saat piyano müziği
İçerik: Ağırlıklı Romantik dönem piyano
Avantaj: Temiz, labeled, profesyonel kayıtlardan
```

### Kaynak 2: Piano-MIDI.de
```
URL: http://www.piano-midi.de/
Format: MIDI
İçerik: Temiz, besteci bazında organize
Avantaj: Yüksek kalite, iyi labeled
```

### Kaynak 3: IMSLP (Petrucci Music Library)
```
URL: https://imslp.org
Format: MusicXML, MIDI (bazıları)
İçerik: Neredeyse her klasik eser
Not: MusicXML → MIDI dönüşümü gerekebilir
```

### Kaynak 4: MuseScore
```
URL: https://musescore.com
Format: MuseScore, MIDI export
İçerik: Geniş, community-maintained
Not: Kalite değişken, filter gerekli
```

### Kaynak 5: Classical Archives
```
URL: https://www.classicalarchives.com
Format: MIDI
İçerik: Geniş koleksiyon
```

### Hızlı Başlangıç Önerisi
```
1. MAESTRO'yu indir (hazır, temiz)
2. piano-midi.de'den Bach koleksiyonunu al
3. Bu ikisiyle başla → 300-400 eser
4. Sonra genişlet
```

---

## 🗄️ 1.3 METADATA YAPISI

Her MIDI için bu bilgileri tut:

```python
# metadata.csv sütunları:
metadata_columns = {
    'file_path': str,        # data/raw/bach/bwv846.mid
    'composer': str,         # "Bach"
    'full_name': str,        # "Johann Sebastian Bach"
    'birth_year': int,       # 1685
    'death_year': int,       # 1750
    'era': str,              # "Baroque" | "Classical" | "Romantic" | "Late Romantic"
    'composition_year': int, # 1722 (yaklaşık)
    'form': str,             # "prelude" | "fugue" | "sonata" | "nocturne" ...
    'key': str,              # "C_major" | "A_minor"
    'instrumentation': str,  # "solo_piano" | "violin" | "chamber"
    'tempo_marking': str,    # "Allegro" | "Andante"
    'duration_seconds': float,
    'total_notes': int,
    'source': str,           # "maestro" | "piano-midi" | "imslp"
    'quality_flag': int,     # 1=ok, 0=kontrol et, -1=sil
}
```

### Metadata Oluşturma Script

```python
# src/create_metadata.py
import os
import pandas as pd
from music21 import converter

def estimate_duration(filepath):
    try:
        score = converter.parse(filepath)
        return score.duration.quarterLength
    except:
        return None

def create_metadata_template(midi_dir, output_csv):
    """Tüm MIDI dosyalarını tarayıp metadata şablonu oluştur"""
    records = []
    
    for root, dirs, files in os.walk(midi_dir):
        for f in files:
            if f.endswith('.mid') or f.endswith('.midi'):
                filepath = os.path.join(root, f)
                records.append({
                    'file_path': filepath,
                    'composer': '',       # Manuel doldur
                    'era': '',            # Manuel doldur
                    'form': '',           # Manuel doldur
                    'key': '',            # Otomatik doldurul.
                    'source': '',         # Manuel doldur
                    'quality_flag': 1
                })
    
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"{len(records)} dosya bulundu. CSV: {output_csv}")
    return df
```

---

## 🧹 1.4 VERİ TEMİZLEME

### Bilinen Sorunlar

| Sorun | Etki | Çözüm |
|-------|------|-------|
| Farklı BPM'ler | Feature karşılaştırması bozulur | Tempo normalize et |
| Velocity farklılıkları | Dinamik analiz yanıltıcı olur | Velocity normalize et |
| Bozuk MIDI dosyaları | Parse hatası | Try-except + quality flag |
| Yanlış etiket | Clustering'i kirletir | Manuel kontrol |
| Çok kısa eserler (<30 sn) | Yetersiz veri | Min. note count filtresi |
| Çok uzun eserler (>20 dk) | Hesaplama yükü | Segment al veya sil |

### Temizleme Pipeline

```python
# src/data_cleaning.py
from music21 import converter, tempo
import numpy as np

def check_midi_quality(filepath, min_notes=50, max_duration=1200):
    """
    MIDI kalitesini kontrol et.
    Returns: (is_valid, reason)
    """
    try:
        score = converter.parse(filepath)
        
        # Nota sayısı kontrolü
        notes = list(score.flatten().notes)
        if len(notes) < min_notes:
            return False, f"Too few notes: {len(notes)}"
        
        # Süre kontrolü
        duration = score.duration.quarterLength
        if duration > max_duration:
            return False, f"Too long: {duration:.0f}s"
        
        return True, "OK"
    
    except Exception as e:
        return False, f"Parse error: {str(e)}"

def normalize_tempo(score, target_bpm=120):
    """
    Tempo'yu normalize et.
    Not: Bu pitch ilişkilerini değiştirmez, sadece
    zaman-bazlı feature'lar için önemli.
    """
    # music21 ile tempo normalizasyonu
    original_tempos = score.flat.getElementsByClass('MetronomeMark')
    # ... implementasyon detayı
    pass

def filter_dataset(metadata_csv, output_csv):
    """Kalitesiz dosyaları filtrele"""
    df = pd.read_csv(metadata_csv)
    
    results = []
    for _, row in df.iterrows():
        valid, reason = check_midi_quality(row['file_path'])
        row['quality_flag'] = 1 if valid else 0
        row['quality_note'] = reason
        results.append(row)
    
    df_clean = pd.DataFrame(results)
    df_clean[df_clean['quality_flag'] == 1].to_csv(output_csv, index=False)
    
    n_total = len(df)
    n_clean = sum(df_clean['quality_flag'] == 1)
    print(f"Toplam: {n_total} | Temiz: {n_clean} | Elenen: {n_total - n_clean}")
```

---

## 📊 1.5 EKSPLORATİF VERİ ANALİZİ (EDA)

### EDA Notebook Yapısı: `notebooks/01_EDA.ipynb`

#### Bölüm 1: Dataset Özeti
```python
# Kaç eser var? Hangi dönemlerden?
df.groupby('era')['file_path'].count().plot(kind='bar')

# Kaç besteci var?
print(f"Besteci sayısı: {df['composer'].nunique()}")
print(f"Eser sayısı: {len(df)}")
```

#### Bölüm 2: Pitch Dağılımları
```python
# Her besteci için ortalama pitch kullanımı
for composer in df['composer'].unique():
    subset = df[df['composer'] == composer]
    # ... pitch histogram çiz
```

#### Bölüm 3: İlk Karşılaştırma
```python
# Bach vs Mozart vs Chopin pitch entropy
entropies = {}
for composer in ['Bach', 'Mozart', 'Chopin']:
    subset_files = df[df['composer'] == composer]['file_path']
    composer_entropies = [calculate_entropy(f) for f in subset_files]
    entropies[composer] = composer_entropies

# Box plot
import matplotlib.pyplot as plt
plt.boxplot([entropies['Bach'], entropies['Mozart'], entropies['Chopin']],
            labels=['Bach', 'Mozart', 'Chopin'])
plt.title("Pitch Entropy Comparison")
plt.ylabel("Entropy (bits)")
plt.show()
```

#### Bölüm 4: Zaman İçinde Değişim
```python
# Dönem bazında entropy değişimi
df['entropy'] = df['file_path'].apply(calculate_entropy)
df.sort_values('composition_year').plot(
    x='composition_year', y='entropy', 
    kind='scatter', alpha=0.5
)
plt.title("Entropy Over Time (1600-1920)")
```

---

## ⚠️ FAZ 1 RİSKLERİ

| Risk | İhtimal | Çözüm |
|------|---------|-------|
| MIDI kaynakları değişmiş/kapanmış | Orta | Birden fazla kaynak kullan |
| Metadata doldurmak çok zaman alır | Yüksek | Otomasyonu maksimize et, önce composer/era yeterli |
| Dataset dengesiz çıkar | Orta | Veri toplarken sayıları takip et |
| Bozuk MIDI oranı yüksek | Düşük | Quality filter otomatik halleder |

---

## 🏁 FAZ 1 TAMAMLANDI SAYILIR WHEN

- [ ] 500+ clean MIDI (minimum) / 1500+ (hedef)
- [ ] `metadata.csv` dolu ve temiz
- [ ] Her besteci için en az 20 eser
- [ ] EDA notebook tamamlanmış
- [ ] İlk pitch entropy karşılaştırması yapılmış
- [ ] Besteciler arasında görsel fark gözlemlenmiş

---

## 🚀 FAZ 2'YE GEÇİŞ KOŞULU

> EDA'da besteciler arasında pitch dağılımında ve entropi değerlerinde **gözlemlenebilir fark** varsa → FAZ 2'ye geç.

---

*Sonraki: [FAZ 2 — Feature Engineering](FAZ_2_Feature_Engineering.md)*
