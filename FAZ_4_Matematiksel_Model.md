# 🟣 FAZ 4 — MATEMATİKSEL MODEL FORMÜLASYONU

> **Süre:** 4–6 hafta  
> **Önceki Faz:** FAZ 3 — Pattern Keşfi  
> **Sonraki Faz:** FAZ 5 — Generatif Müzik

---

## 🎯 FAZ AMACI

FAZ 3'te keşfettiğimiz pattern'leri **formal matematiksel modele** dönüştürmek. Bu model hem açıklanabilir hem de üretim için kullanılabilir olmalı.

---

## ✅ FAZ ÇIKTILARI

- [ ] Markov geçiş modeli (pitch + rhythm)
- [ ] "Müzikal imza" vektörleri (her besteci için)
- [ ] Entropi-güzellik fonksiyonu taslağı
- [ ] Constraint-based üretim kuralları
- [ ] Model validation metrikleri
- [ ] Paper'ın Methods bölümüne katkı

---

## 🧮 4.1 MODEL MİMARİSİ

Projenin matematiksel modeli üç katmandan oluşacak:

```
KATMAN A: İstatistiksel Dağılım Modeli
  → Pitch, interval, ritim dağılımları
  → Her besteci/dönem için karakteristik parametreler

KATMAN B: Markov Geçiş Modeli
  → Notadan notaya geçiş olasılıkları
  → Harmonik ilerleme kuralları
  → Temporal yapı kalıpları

KATMAN C: Kısıt (Constraint) Modeli
  → Entropi bandı hedefi
  → Harmonik gerilim kısıtları
  → Yapısal form kısıtları
```

---

## 📊 4.2 İSTATİSTİKSEL DAĞILIM MODELİ

### Her Besteci için Dağılım Profili

```python
# src/model/distribution_model.py
import numpy as np
from scipy.stats import entropy as kl_divergence
import json

class ComposerDistributionModel:
    """
    Her bestecinin karakteristik istatistiksel profilini öğrenir.
    """
    
    def __init__(self):
        self.profiles = {}
    
    def fit_composer(self, composer_name, features_df):
        """
        Bir bestecinin özelliklerinden dağılım profili çıkar.
        """
        composer_data = features_df[features_df['composer'] == composer_name]
        
        # Her feature için Gaussian dağılım parametresi
        profile = {}
        feature_cols = [c for c in features_df.columns 
                       if c not in ['filepath','composer','era','form']]
        
        for feat in feature_cols:
            values = composer_data[feat].dropna().values
            if len(values) > 0:
                profile[feat] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                }
        
        self.profiles[composer_name] = profile
        return profile
    
    def fit_all(self, features_df):
        for composer in features_df['composer'].unique():
            self.fit_composer(composer, features_df)
        print(f"{len(self.profiles)} besteci profili oluşturuldu.")
    
    def musical_distance(self, composer1, composer2, features=None):
        """
        İki besteci arasındaki 'matematiksel mesafe'.
        KL divergence yaklaşımı.
        """
        p1 = self.profiles.get(composer1, {})
        p2 = self.profiles.get(composer2, {})
        
        if features is None:
            features = list(set(p1.keys()) & set(p2.keys()))
        
        distances = []
        for feat in features:
            if feat in p1 and feat in p2:
                # Normalized mean difference
                std_pool = (p1[feat]['std'] + p2[feat]['std']) / 2 + 1e-8
                d = abs(p1[feat]['mean'] - p2[feat]['mean']) / std_pool
                distances.append(d)
        
        return float(np.mean(distances))
    
    def closest_composer(self, feature_vector, feature_names):
        """
        Bir feature vektörüne en yakın besteci kim?
        """
        distances = {}
        for composer, profile in self.profiles.items():
            dists = []
            for feat, val in zip(feature_names, feature_vector):
                if feat in profile:
                    std = profile[feat]['std'] + 1e-8
                    d = abs(val - profile[feat]['mean']) / std
                    dists.append(d)
            distances[composer] = np.mean(dists)
        
        return min(distances, key=distances.get), distances
    
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.profiles, f, indent=2)
    
    def load(self, filepath):
        with open(filepath, 'r') as f:
            self.profiles = json.load(f)
```

### Müzikal Mesafe Matrisi

```python
# Tüm besteciler arası mesafe matrisi
model = ComposerDistributionModel()
model.fit_all(df)

composers = list(model.profiles.keys())
n = len(composers)
distance_matrix = np.zeros((n, n))

for i, c1 in enumerate(composers):
    for j, c2 in enumerate(composers):
        distance_matrix[i][j] = model.musical_distance(c1, c2)

# Görselleştir
plt.figure(figsize=(14, 10))
sns.heatmap(distance_matrix, 
            xticklabels=composers, yticklabels=composers,
            cmap='YlOrRd', annot=True, fmt='.2f')
plt.title("Besteciler Arası Matematiksel Mesafe Matrisi")
plt.tight_layout()
plt.savefig('results/figures/composer_distance_matrix.png', dpi=150)

# ⭐ Bu grafik: "Kim kime en yakın?"
# Haydn-Mozart yakın mı? Bach-Handel yakın mı?
```

---

## 🔗 4.3 MARKOV GEÇİŞ MODELİ

Müziğin "nasıl aktığını" modelleyen olasılık matrisleri.

```python
# src/model/markov_model.py
import numpy as np

class MusicMarkovModel:
    """
    Çok katmanlı Markov geçiş modeli.
    """
    
    def __init__(self, order=1):
        self.order = order  # 1. order: sadece önceki nota
        self.pitch_transitions = {}    # pitch class geçişleri
        self.interval_transitions = {} # interval geçişleri
        self.duration_transitions = {} # süre geçişleri
    
    def fit(self, notes_data_list):
        """notes_data_list: Her eser için notalar listesi"""
        
        pc_matrix = np.zeros((12, 12))  # pitch class
        dur_matrix = np.zeros((8, 8))   # quantized duration
        
        for notes_data in notes_data_list:
            pitches = [n['pitch'] % 12 for n in notes_data]
            durations = [self._quantize_duration(n['duration']) for n in notes_data]
            
            # Pitch class geçişleri
            for i in range(len(pitches) - self.order):
                pc_matrix[pitches[i]][pitches[i+1]] += 1
            
            # Duration geçişleri  
            for i in range(len(durations) - self.order):
                if durations[i] < 8 and durations[i+1] < 8:
                    dur_matrix[durations[i]][durations[i+1]] += 1
        
        # Normalize (satır bazında)
        self.pitch_transitions = self._normalize_matrix(pc_matrix)
        self.duration_transitions = self._normalize_matrix(dur_matrix)
        
        return self
    
    def _quantize_duration(self, dur, levels=8):
        """Süreyi 8 seviyeye kuantize et"""
        quantized = min(int(dur * 2), levels - 1)  # 1/8 birimler
        return quantized
    
    def _normalize_matrix(self, matrix):
        """Satır-normalize et"""
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return matrix / row_sums
    
    def generate_sequence(self, length=64, start_pitch=60, temperature=1.0):
        """
        Markov modelinden nota dizisi üret.
        temperature: >1 = daha rastgele, <1 = daha deterministik
        """
        sequence = [start_pitch % 12]
        
        for _ in range(length - 1):
            current_pc = sequence[-1]
            probs = self.pitch_transitions[current_pc].copy()
            
            # Temperature scaling
            if temperature != 1.0:
                probs = np.power(probs + 1e-8, 1/temperature)
                probs = probs / probs.sum()
            
            next_pc = np.random.choice(12, p=probs)
            sequence.append(next_pc)
        
        return sequence
    
    def transition_entropy(self):
        """
        Geçiş matrisinin ortalama entropisi.
        Bu, bestecinin harmonik 'tahmin edilebilirliği'ni gösterir.
        """
        entropies = []
        for row in self.pitch_transitions:
            row = row[row > 0]
            if len(row) > 0:
                ent = -np.sum(row * np.log2(row))
                entropies.append(ent)
        return float(np.mean(entropies))
    
    def similarity(self, other_model):
        """İki Markov modelinin benzerliği (matrix cosine similarity)"""
        flat1 = self.pitch_transitions.flatten()
        flat2 = other_model.pitch_transitions.flatten()
        dot = np.dot(flat1, flat2)
        norm = np.linalg.norm(flat1) * np.linalg.norm(flat2)
        return float(dot / (norm + 1e-8))
```

### Besteci Markov Modellerini Karşılaştır

```python
# Her besteci için ayrı Markov modeli eğit
composer_models = {}

for composer in df['composer'].unique():
    composer_files = df[df['composer'] == composer]['filepath'].values
    notes_list = [parse_midi_to_notes(f) for f in composer_files if f]
    notes_list = [n for n in notes_list if n is not None]
    
    model = MusicMarkovModel(order=1)
    model.fit(notes_list)
    composer_models[composer] = model
    
    ent = model.transition_entropy()
    print(f"{composer}: Transition entropy = {ent:.3f}")

# Hangi bestecinin geçişleri daha tahmin edilebilir?
# Bach < Mozart < Chopin < Debussy ? (hipotez)
```

---

## 🎯 4.4 GÜZELLIK / ETKİ FONKSIYONU

Projenin en cesur hipotezi: Matematiksel bir "güzellik" var mı?

### Teorik Çerçeve

```
Müzikal Etki = f(
    Entropi,        # Çeşitlilik
    Tahminsellik,   # Beklentiyi karşılama
    Gerilim,        # Harmonik enerji
    Çözülme,        # Gerilimin düşüşü
    Öz-benzerlik    # Yapısal tutarlılık
)
```

### İlk Formülesyon (Test Edilecek)

```python
def musical_impact_score(features):
    """
    Teorik müzikal etki skoru.
    Bu formül FAZ 6 insan deneyi sonrası kalibre edilecek.
    
    NOT: Bu bir hipotez, gerçek doğrulama sonra gelecek.
    """
    
    # Optimal entropi bandı: 2.0 - 2.8 bits
    # (Bu aralık FAZ 3 bulgularına göre güncellenecek)
    ENTROPY_OPTIMAL_LOW = 2.0
    ENTROPY_OPTIMAL_HIGH = 2.8
    
    entropy = features.get('pitch_entropy', 0)
    consonance = features.get('consonance_score', 0)
    repetition = features.get('repetition_index', 0)
    fractal_dim = features.get('fractal_dimension', 1.5)
    
    # Entropi skoru: Optimal bandın ne kadar yakınında?
    if ENTROPY_OPTIMAL_LOW <= entropy <= ENTROPY_OPTIMAL_HIGH:
        entropy_score = 1.0
    else:
        deviation = min(abs(entropy - ENTROPY_OPTIMAL_LOW),
                       abs(entropy - ENTROPY_OPTIMAL_HIGH))
        entropy_score = max(0, 1 - deviation)
    
    # Konsonans skoru: Orta konsonans en iyi?
    consonance_score = 1 - abs(consonance - 0.65)  # 0.65 optimal?
    
    # Tekrar skoru: Orta tekrar (ne çok ne az)
    repetition_score = 1 - abs(repetition - 0.4)
    
    # Fraktal skor: ~1.5 optimal?
    fractal_score = 1 - abs(fractal_dim - 1.5)
    
    # Ağırlıklı ortalama (katsayılar FAZ 6'da kalibre edilecek)
    weights = {
        'entropy': 0.35,
        'consonance': 0.25,
        'repetition': 0.20,
        'fractal': 0.20,
    }
    
    score = (weights['entropy'] * entropy_score +
             weights['consonance'] * consonance_score +
             weights['repetition'] * repetition_score +
             weights['fractal'] * fractal_score)
    
    return float(score)

# Test: En yüksek skoru hangi besteci alıyor?
df['impact_score'] = df.apply(
    lambda row: musical_impact_score(row.to_dict()), axis=1
)

print(df.groupby('composer')['impact_score'].mean().sort_values(ascending=False))
```

---

## 🔧 4.5 KISIT (CONSTRAINT) MODELİ

Üretim için kullanılacak matematiksel kısıtlar.

```python
# src/model/constraints.py

class MusicalConstraints:
    """
    Müzikal üretim için matematiksel kısıtlar.
    Her parametre FAZ 3 bulgularından gelecek.
    """
    
    PRESETS = {
        'baroque': {
            'entropy_range': (1.8, 2.4),
            'consonance_min': 0.7,
            'step_ratio_min': 0.6,    # Çok adım, az atlama
            'repetition_target': 0.5,  # Yüksek tekrar (füg yapısı)
            'fractal_dim_range': (1.2, 1.6),
        },
        'classical': {
            'entropy_range': (2.0, 2.6),
            'consonance_min': 0.6,
            'step_ratio_min': 0.5,
            'repetition_target': 0.4,
            'fractal_dim_range': (1.3, 1.6),
        },
        'romantic': {
            'entropy_range': (2.3, 3.0),
            'consonance_min': 0.4,
            'step_ratio_min': 0.4,
            'repetition_target': 0.3,
            'fractal_dim_range': (1.4, 1.8),
        },
        'late_romantic': {
            'entropy_range': (2.5, 3.2),
            'consonance_min': 0.3,
            'step_ratio_min': 0.35,
            'repetition_target': 0.25,
            'fractal_dim_range': (1.5, 2.0),
        }
    }
    
    def __init__(self, style='classical'):
        self.params = self.PRESETS.get(style, self.PRESETS['classical'])
    
    def is_valid(self, features):
        """Üretilen bir pasajın kısıtları karşılayıp karşılamadığını kontrol et"""
        
        checks = []
        
        # Entropi kontrolü
        e = features.get('pitch_entropy', 0)
        low, high = self.params['entropy_range']
        checks.append(low <= e <= high)
        
        # Konsonans kontrolü
        c = features.get('consonance_score', 0)
        checks.append(c >= self.params['consonance_min'])
        
        # Adım oranı
        sr = features.get('step_ratio', 0)
        checks.append(sr >= self.params['step_ratio_min'])
        
        return all(checks), checks
    
    def distance_from_target(self, features):
        """Hedef kısıtlardan ne kadar uzak?"""
        distances = []
        
        e = features.get('pitch_entropy', 0)
        low, high = self.params['entropy_range']
        target_e = (low + high) / 2
        distances.append(abs(e - target_e))
        
        # ... diğer kısıtlar
        
        return float(np.mean(distances))
```

---

## 📐 4.6 MODEL VALİDASYONU

Model gerçekçi mi?

```python
# Validation: Orijinal eserlerin skoru ne?
# Model, orijinal eserleri tanıyabiliyor mu?

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df['era'], test_size=0.2, random_state=42
)

# 1. Distribution model: Test setindeki eserleri doğru döneme atayor mu?
# 2. Markov model: Generated sekanslar orijinallerle benzer feature'lara sahip mi?

# Feature distribution karşılaştırması
def compare_distributions(original_features, generated_features, feature_name):
    """
    Orijinal ve üretilmiş müziğin feature dağılımları ne kadar benzer?
    KL Divergence kullan.
    """
    from scipy.special import kl_div
    
    orig = np.array(original_features)
    gen = np.array(generated_features)
    
    # Normalize histogram
    bins = 20
    orig_hist, edges = np.histogram(orig, bins=bins, density=True)
    gen_hist, _ = np.histogram(gen, bins=edges, density=True)
    
    # KL divergence
    orig_hist = orig_hist + 1e-8
    gen_hist = gen_hist + 1e-8
    kl = np.sum(kl_div(orig_hist, gen_hist))
    
    return float(kl)
```

---

## ⚠️ FAZ 4 RİSKLERİ

| Risk | İhtimal | Çözüm |
|------|---------|-------|
| Markov modeli overfit | Orta | Daha büyük dataset, smoothing ekle |
| Güzellik fonksiyonu tamamen teorik kalır | Yüksek | Bu beklenen — FAZ 6'da kalibre edilecek |
| Kısıtlar çok kısıtlayıcı olur | Orta | Kısıtları bayesian range olarak tanımla |

---

## 🏁 FAZ 4 TAMAMLANDI SAYILIR WHEN

- [ ] `ComposerDistributionModel` eğitilmiş ve test edilmiş
- [ ] `MusicMarkovModel` her besteci için fit edilmiş
- [ ] Transition entropy karşılaştırması yapılmış
- [ ] Güzellik fonksiyonu taslağı yazılmış
- [ ] Constraint preset'leri tanımlanmış
- [ ] Model validation tamamlanmış
- [ ] Besteciler arası mesafe matrisi görselleştirilmiş

---

## 🚀 FAZ 5'E GEÇİŞ KOŞULU

> Markov modeli belirli bir stil için en az 16 bar uzunluğunda tutarlı nota dizisi üretebiliyorsa → FAZ 5'e geç.

---

*Sonraki: [FAZ 5 — Generatif Müzik](FAZ_5_Generatif_Muzik.md)*
