# 🟡 FAZ 2 — FEATURE ENGINEERING

> **Süre:** 6–8 hafta (esnek)  
> **Önceki Faz:** FAZ 1 — Veri Toplama  
> **Sonraki Faz:** FAZ 3 — Pattern Keşfi

---

## 🎯 FAZ AMACI

Her müzik parçasını **matematiksel bir vektöre** dönüştürmek. Bu vektör parçanın "matematiksel DNA'sı" olacak.

Çıktı: `feature_matrix.csv` → Shape: `(N_eser × ~100_feature)`

---

## ✅ FAZ ÇIKTILARI

- [ ] 6 katmanlı feature extraction pipeline (Python modülü)
- [ ] Her eser için 80–100 boyutlu feature vektörü
- [ ] `feature_matrix.csv`
- [ ] Feature korelasyon analizi
- [ ] "Hangi feature'lar en bilgilendirici?" sorusunun ilk cevabı
- [ ] Feature engineering notebook

---

## 🏗️ FEATURE ARCHITECTURE

Her özellik 6 katmana ayrılır:

```
KATMAN 1: Pitch           → Hangi notalar kullanılıyor?
KATMAN 2: Interval        → Notalar arasındaki atlamalar?
KATMAN 3: Harmony         → Akorlar ve gerilim?
KATMAN 4: Rhythm/Tempo    → Zamanlama ve ritim?
KATMAN 5: Structure       → Makro form ve tekrar?
KATMAN 6: Spectral        → Frekans dağılımı?
```

---

## 🎵 KATMAN 1: PITCH FEATURES (12 feature)

Hangi notaların, hangi sıklıkla kullanıldığını ölçer.

```python
# src/features/pitch_features.py
import numpy as np
from scipy.stats import entropy as scipy_entropy

def pitch_class_histogram(notes):
    """12-bin normalize histogram (mod 12)"""
    pcs = [n % 12 for n in notes]
    hist = np.bincount(pcs, minlength=12).astype(float)
    return hist / hist.sum()

def pitch_entropy(notes):
    """
    Shannon entropy of pitch class distribution.
    Yüksek = çok çeşitli nota kullanımı
    Düşük = az nota üzerinde yoğunlaşma
    """
    hist = pitch_class_histogram(notes)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def tonal_center_strength(notes):
    """
    En baskın pitch class'ın oranı.
    Yüksek = güçlü tonalite / tek nota baskın
    Düşük = zayıf tonalite / çok çeşitli
    """
    hist = pitch_class_histogram(notes)
    return float(hist.max())

def pitch_range(notes):
    """Kullanılan pitch aralığı (semitone)"""
    return int(max(notes) - min(notes))

def pitch_mean(notes):
    """Ortalama pitch (merkez)"""
    return float(np.mean(notes))

def pitch_std(notes):
    """Pitch standart sapması"""
    return float(np.std(notes))

def chromatic_saturation(notes):
    """
    Kaç farklı pitch class kullanılmış? (max 12)
    12'ye yakın = kromatik zenginlik
    3-4 = modal / tonal sadelik
    """
    return len(set(n % 12 for n in notes))

def extract_pitch_features(notes):
    """Tüm pitch feature'larını döndür"""
    hist = pitch_class_histogram(notes)
    return {
        'pitch_entropy': pitch_entropy(notes),
        'tonal_center_strength': tonal_center_strength(notes),
        'pitch_range': pitch_range(notes),
        'pitch_mean': pitch_mean(notes),
        'pitch_std': pitch_std(notes),
        'chromatic_saturation': chromatic_saturation(notes),
        # Histogram'ın her bin'i ayrı feature olarak
        **{f'pc_{i}': float(hist[i]) for i in range(12)}
    }
```

---

## 🔁 KATMAN 2: INTERVAL FEATURES (10 feature)

Notadan notaya **atlamaların** matematiksel profili.

```python
# src/features/interval_features.py

def extract_intervals(notes):
    """Ardışık notalar arası fark (semitone)"""
    return np.diff(notes)

def interval_entropy(notes):
    """
    Interval dağılımının entropisi.
    Bach: Küçük adımlar → düşük entropi
    Liszt: Büyük atlamalar → yüksek entropi
    """
    intervals = extract_intervals(notes)
    # -12 to +12 range, clipped
    intervals = np.clip(intervals, -12, 12)
    hist, _ = np.histogram(intervals, bins=25, range=(-12.5, 12.5))
    hist = hist.astype(float)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def directional_bias(notes):
    """
    Yükselen / alçalan nota tercihi.
    +1.0 = tamamen yükselen
    -1.0 = tamamen alçalan
    0.0 = dengeli
    """
    intervals = extract_intervals(notes)
    ascending = np.sum(intervals > 0)
    descending = np.sum(intervals < 0)
    total = ascending + descending
    if total == 0:
        return 0.0
    return float((ascending - descending) / total)

def step_ratio(notes):
    """Küçük adım (1-2 semitone) oranı"""
    intervals = np.abs(extract_intervals(notes))
    return float(np.sum(intervals <= 2) / len(intervals))

def leap_ratio(notes):
    """Büyük atlama (>4 semitone) oranı"""
    intervals = np.abs(extract_intervals(notes))
    return float(np.sum(intervals > 4) / len(intervals))

def mean_interval_size(notes):
    """Ortalama interval büyüklüğü (absolut değer)"""
    intervals = np.abs(extract_intervals(notes))
    return float(np.mean(intervals))

def interval_transition_matrix(notes, normalize=True):
    """
    12x12 pitch class geçiş matrisi.
    M[i][j] = i pitch class'tan j'ye geçiş olasılığı
    Bu bestecinin 'harmonik imzası'dır.
    """
    pcs = [n % 12 for n in notes]
    matrix = np.zeros((12, 12))
    for i in range(len(pcs) - 1):
        matrix[pcs[i]][pcs[i+1]] += 1
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = matrix / row_sums
    return matrix

def extract_interval_features(notes):
    return {
        'interval_entropy': interval_entropy(notes),
        'directional_bias': directional_bias(notes),
        'step_ratio': step_ratio(notes),
        'leap_ratio': leap_ratio(notes),
        'mean_interval': mean_interval_size(notes),
        'interval_std': float(np.std(np.abs(extract_intervals(notes)))),
    }
```

---

## 🎼 KATMAN 3: HARMONİ FEATURES (8 feature)

Harmonik gerilim ve çözülme kalıpları.

```python
# src/features/harmony_features.py
from music21 import harmony, roman, key as m21key

# Dissonans tablosu (müzik teorisinden)
CONSONANCE_MAP = {
    0: 1.0,   # Unison - tam konsonans
    1: 0.0,   # m2 - sert dissonans
    2: 0.2,   # M2 - yumuşak dissonans
    3: 0.8,   # m3 - konsonans
    4: 0.8,   # M3 - konsonans
    5: 0.9,   # P4 - konsonans
    6: 0.1,   # TT - sert dissonans
    7: 1.0,   # P5 - tam konsonans
    8: 0.7,   # m6 - konsonans
    9: 0.7,   # M6 - konsonans
    10: 0.3,  # m7 - dissonans
    11: 0.2,  # M7 - sert dissonans
}

def consonance_score(notes):
    """
    Ortalama konsonans skoru (0-1).
    1.0 = tamamen konsonant (Barok)
    0.0 = tamamen dissonant (20. yy atonalite)
    """
    intervals = np.abs(np.diff(notes)) % 12
    scores = [CONSONANCE_MAP.get(i, 0.5) for i in intervals]
    return float(np.mean(scores))

def dissonance_index(notes):
    return 1.0 - consonance_score(notes)

def harmonic_rhythm_variance(notes, durations):
    """
    Akor değişim hızının varyansı.
    Yüksek = düzensiz, dramatik değişimler (Beethoven?)
    Düşük = düzenli, sakin değişimler (Bach?)
    """
    # Basitleştirilmiş: duration değişkenliği
    return float(np.std(durations))

def extract_harmony_features(notes, durations=None):
    features = {
        'consonance_score': consonance_score(notes),
        'dissonance_index': dissonance_index(notes),
    }
    if durations:
        features['duration_variance'] = harmonic_rhythm_variance(notes, durations)
    return features
```

---

## ⏱️ KATMAN 4: RİTİM & TEMPO FEATURES (8 feature)

Zamanlama ve ritim kalıpları.

```python
# src/features/rhythm_features.py

def rhythmic_entropy(durations):
    """
    Nota sürelerinin entropisi.
    Yüksek = çok çeşitli ritim değerleri (Chopin rubato?)
    Düşük = tekdüze ritim (Bach koraller?)
    """
    # Kuantize et (1/16'lık birimler)
    quantized = np.round(np.array(durations) * 4) / 4
    unique, counts = np.unique(quantized, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))

def note_density(notes, total_duration):
    """Birim zamandaki nota sayısı"""
    return float(len(notes) / total_duration) if total_duration > 0 else 0

def syncopation_estimate(durations, beats=4):
    """
    Senkopasyon tahmini.
    Zayıf vuruşlardaki uzun notalar → yüksek senkopasyon
    """
    # Basit yaklaşım: Süre varyansı
    return float(np.std(durations) / (np.mean(durations) + 1e-8))

def tempo_variance(notes_data):
    """
    IOI (Inter-Onset Interval) varyansı.
    Yüksek = tempo değişkenliği, rubato (Chopin)
    Düşük = sabit tempo (Barok dans formları)
    """
    starts = [n['start'] for n in notes_data if 'start' in n]
    if len(starts) < 2:
        return 0.0
    iois = np.diff(sorted(starts))
    return float(np.std(iois) / (np.mean(iois) + 1e-8))

def extract_rhythm_features(notes_data):
    durations = [n['duration'] for n in notes_data]
    starts = [n['start'] for n in notes_data]
    total_dur = max(starts) + durations[-1] if starts else 1
    
    return {
        'rhythmic_entropy': rhythmic_entropy(durations),
        'note_density': note_density(notes_data, total_dur),
        'syncopation_estimate': syncopation_estimate(durations),
        'tempo_variance': tempo_variance(notes_data),
        'duration_mean': float(np.mean(durations)),
        'duration_std': float(np.std(durations)),
    }
```

---

## 🌀 KATMAN 5: YAPISAL FEATURES (10 feature)

Makro form, tekrar ve öz-benzerlik.

```python
# src/features/structural_features.py
from scipy.spatial.distance import cosine

def self_similarity_matrix(notes, window=20):
    """
    Müziğin kendine benzerlik matrisi.
    Tekrarlayan temalar köşegen çizgiler oluşturur.
    """
    pcs = [n % 12 for n in notes]
    n_windows = len(pcs) - window
    
    windows = np.array([pcs[i:i+window] for i in range(n_windows)])
    
    # Normalize histogram vektörlerine çevir
    def to_hist(w):
        h = np.bincount(w, minlength=12).astype(float)
        return h / (h.sum() + 1e-8)
    
    hists = np.array([to_hist(w) for w in windows])
    
    # Cosine similarity matrix
    n = len(hists)
    ssm = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = 1 - cosine(hists[i], hists[j])
            ssm[i][j] = ssm[j][i] = sim
    
    return ssm

def repetition_index(notes, window=20):
    """
    Ortalama öz-benzerlik skoru.
    Yüksek = çok tekrar (Bach füg)
    Düşük = az tekrar (serbest form)
    """
    ssm = self_similarity_matrix(notes, window)
    # Diyagonelden uzak elemanlara bak
    n = len(ssm)
    if n < 10:
        return 0.0
    off_diag = []
    for i in range(n):
        for j in range(i+5, n):  # En az 5 adım uzakta
            off_diag.append(ssm[i][j])
    return float(np.mean(off_diag))

def fractal_dimension_estimate(notes, n_segments=8):
    """
    Basit fraktal boyut tahmini.
    Müziğin ölçek-bağımsız yapı karmaşıklığı.
    ~1.0 = çok basit
    ~1.5 = orta karmaşıklık (müzikal sweet spot?)
    ~2.0 = rastgele
    """
    pitches = np.array(notes, dtype=float)
    pitches = (pitches - pitches.min()) / (pitches.max() - pitches.min() + 1e-8)
    
    # Box counting (basitleştirilmiş)
    counts = []
    sizes = []
    for s in range(2, n_segments + 1):
        box_size = len(pitches) / s
        occupied = set()
        for i, p in enumerate(pitches):
            box_x = int(i / box_size)
            box_y = int(p * s)
            occupied.add((box_x, box_y))
        counts.append(len(occupied))
        sizes.append(1.0 / s)
    
    # Log-log eğimi = fraktal boyut
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    slope = np.polyfit(log_sizes, log_counts, 1)[0]
    return float(-slope)

def extract_structural_features(notes):
    return {
        'repetition_index': repetition_index(notes),
        'fractal_dimension': fractal_dimension_estimate(notes),
        'unique_pitch_classes': len(set(n % 12 for n in notes)),
        'total_notes': len(notes),
    }
```

---

## 🌊 KATMAN 6: SPEKTRAL FEATURES (6 feature)

Pitch değerlerinin Fourier analizi ile periyodik yapı tespiti.

```python
# src/features/spectral_features.py
from scipy.fft import fft

def spectral_features_from_pitch(notes):
    """
    Pitch serisinin Fourier dönüşümü.
    Müzikal periyodikliği yakalar.
    """
    pitches = np.array(notes, dtype=float)
    pitches = pitches - pitches.mean()  # detrend
    
    fft_vals = np.abs(fft(pitches))
    fft_vals = fft_vals[:len(fft_vals)//2]  # pozitif frekanslar
    
    total_power = np.sum(fft_vals**2)
    if total_power == 0:
        return {'spectral_centroid': 0, 'spectral_entropy': 0,
                'spectral_rolloff': 0, 'dominant_frequency': 0}
    
    freqs = np.arange(len(fft_vals))
    power = fft_vals**2
    
    # Spectral centroid: Ağırlıklı ortalama frekans
    centroid = float(np.sum(freqs * power) / total_power)
    
    # Spectral entropy: Frekans dağılımının çeşitliliği
    prob = power / total_power
    prob = prob[prob > 0]
    sp_entropy = float(-np.sum(prob * np.log2(prob)))
    
    # Dominant frequency: En güçlü periyodik bileşen
    dominant = int(np.argmax(power))
    
    return {
        'spectral_centroid': centroid,
        'spectral_entropy': sp_entropy,
        'dominant_frequency': dominant,
        'spectral_flatness': float(np.exp(np.mean(np.log(power + 1e-8))) / (np.mean(power) + 1e-8))
    }
```

---

## 🔗 ANA FEATURE EXTRACTION PIPELINE

```python
# src/features/extractor.py
import pandas as pd
from music21 import converter, note, chord
from .pitch_features import extract_pitch_features
from .interval_features import extract_interval_features
from .harmony_features import extract_harmony_features
from .rhythm_features import extract_rhythm_features
from .structural_features import extract_structural_features
from .spectral_features import spectral_features_from_pitch

def extract_all_features(filepath):
    """
    Ana extraction fonksiyonu.
    Tek bir MIDI → feature dict
    """
    try:
        score = converter.parse(filepath)
        notes_data = []
        
        for element in score.flatten().notes:
            if isinstance(element, note.Note):
                notes_data.append({
                    'pitch': element.pitch.midi,
                    'duration': float(element.duration.quarterLength),
                    'start': float(element.offset),
                })
            elif isinstance(element, chord.Chord):
                for n in element.notes:
                    notes_data.append({
                        'pitch': n.pitch.midi,
                        'duration': float(element.duration.quarterLength),
                        'start': float(element.offset),
                    })
        
        if len(notes_data) < 20:
            return None
        
        pitches = [n['pitch'] for n in notes_data]
        durations = [n['duration'] for n in notes_data]
        
        features = {}
        features.update(extract_pitch_features(pitches))
        features.update(extract_interval_features(pitches))
        features.update(extract_harmony_features(pitches, durations))
        features.update(extract_rhythm_features(notes_data))
        features.update(extract_structural_features(pitches))
        features.update(spectral_features_from_pitch(pitches))
        
        features['filepath'] = filepath
        return features
    
    except Exception as e:
        print(f"Error: {filepath} → {e}")
        return None

def build_feature_matrix(metadata_csv, output_csv):
    """
    Tüm dataset için feature matrix oluştur.
    """
    df_meta = pd.read_csv(metadata_csv)
    df_meta = df_meta[df_meta['quality_flag'] == 1]
    
    all_features = []
    for _, row in df_meta.iterrows():
        features = extract_all_features(row['file_path'])
        if features:
            features['composer'] = row['composer']
            features['era'] = row['era']
            features['form'] = row.get('form', '')
            all_features.append(features)
        
        if len(all_features) % 50 == 0:
            print(f"Progress: {len(all_features)} eser işlendi")
    
    df_features = pd.DataFrame(all_features)
    df_features.to_csv(output_csv, index=False)
    print(f"Feature matrix: {df_features.shape}")
    return df_features
```

---

## 📊 FEATURE KORELASYON ANALİZİ

```python
# notebooks/02_features.ipynb

# Korelasyon matrisi
feature_cols = [c for c in df.columns 
                if c not in ['filepath','composer','era','form']]

corr_matrix = df[feature_cols].corr()

# Heatmap
import seaborn as sns
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
            vmin=-1, vmax=1, center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig('results/figures/feature_correlation.png', dpi=150)

# Yüksek korelasyonlu feature çiftlerini bul (redundancy)
high_corr = []
for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix)):
        if abs(corr_matrix.iloc[i,j]) > 0.9:
            high_corr.append((corr_matrix.index[i], 
                              corr_matrix.columns[j],
                              corr_matrix.iloc[i,j]))
print("Yüksek korelasyonlu feature çiftleri:")
for pair in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
    print(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")
```

---

## ⚠️ FAZ 2 RİSKLERİ

| Risk | İhtimal | Çözüm |
|------|---------|-------|
| Bazı feature'lar yüksek korelasyonlu | Yüksek | PCA zaten halleder, ama manual da silinebilir |
| Harmony features music21 API'siyle zor | Orta | Önce basit yaklaşım, sonra derinleştir |
| Fraktal hesaplama yavaş | Orta | Subset üzerinde test et, optimize et |
| Feature normalizasyonu unutulursa | Orta | StandardScaler sonra her zaman |

---

## 🏁 FAZ 2 TAMAMLANDI SAYILIR WHEN

- [ ] Tüm 6 katman implement edilmiş
- [ ] `feature_matrix.csv` oluşturulmuş (N_eser × 80+ feature)
- [ ] Korelasyon analizi yapılmış
- [ ] En önemli 20 feature belirlenmiş (görsel olarak)
- [ ] Besteci bazında feature dağılımları görselleştirilmiş

---

## 🚀 FAZ 3'E GEÇİŞ KOŞULU

> Feature matrix hazır ve en az 2 feature, besteci bazında istatistiksel olarak anlamlı farklılık gösteriyorsa → FAZ 3'e geç.

---

*Sonraki: [FAZ 3 — Pattern Keşfi](FAZ_3_Pattern_Kesfi.md)*
