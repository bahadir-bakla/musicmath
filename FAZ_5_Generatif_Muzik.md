# 🔴 FAZ 5 — GENERATİF MÜZİK ÜRETİMİ

> **Süre:** 4–6 hafta  
> **Önceki Faz:** FAZ 4 — Matematiksel Model  
> **Sonraki Faz:** FAZ 6 — İnsan Deneyi

---

## 🎯 FAZ AMACI

FAZ 4'te kurduğumuz matematiksel modeli kullanarak **gerçekten dinlenebilir müzik üretmek**. Üretilen müzik hem matematiksel kısıtları karşılamalı hem de müzikal açıdan mantıklı olmalı.

---

## ✅ FAZ ÇIKTILARI

- [ ] Çalışan generatif pipeline
- [ ] 4 farklı dönem stilinde üretim yapabilen sistem
- [ ] 50–100 generated MIDI dosyası
- [ ] Her üretimin matematiksel "sertifikası"
- [ ] Kalite filtresi (otomatik)
- [ ] FAZ 6 insan deneyi için hazır örnekler

---

## 🏗️ 5.1 ÜRETİM MİMARİSİ

Sistem iki modül üzerine kurulu:

```
GENERATOR
├── PitchGenerator    → Hangi nota?
├── RhythmGenerator   → Ne kadar uzun?
├── HarmonyChecker    → Müzikal olarak geçerli mi?
└── ConstraintFilter  → Matematiksel kısıtları karşılıyor mu?

PIPELINE
Input: (stil, uzunluk, tonal_merkez, entropi_hedefi)
  ↓
Başlangıç durumu seç
  ↓
Nota nota üret (Markov + sampling)
  ↓
Kısıt kontrolü (her 8 barda bir)
  ↓
Kabul veya yeniden dene
  ↓
MIDI'ya yaz
  ↓
Feature extract + "matematiksel sertifika"
Output: MIDI dosyası
```

---

## 🎹 5.2 TEMEL GENERATOR

```python
# src/generation/generator.py
import numpy as np
from music21 import stream, note, midi, tempo, key
import random

class ClassicalMusicGenerator:
    """
    Matematiksel model tabanlı klasik müzik üreticisi.
    """
    
    def __init__(self, markov_model, distribution_model, constraints):
        self.markov = markov_model
        self.dist = distribution_model
        self.constraints = constraints
    
    def generate(self, 
                 style='classical',
                 length_bars=32,
                 tonic=60,           # C4 = MIDI 60
                 time_signature=4,
                 target_entropy=2.3,
                 temperature=1.0,
                 seed=None):
        """
        Ana üretim fonksiyonu.
        
        Returns: music21 Stream objesi
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Stream oluştur
        score = stream.Score()
        part = stream.Part()
        
        # Tempo ve tonalite ayarla
        part.append(tempo.MetronomeMark(number=self._get_tempo(style)))
        part.append(key.Key(self._midi_to_key(tonic)))
        
        # Nota üretimi
        notes = []
        current_pitch = tonic
        beats_per_bar = time_signature
        total_beats = length_bars * beats_per_bar
        current_beat = 0
        
        max_attempts = total_beats * 3
        attempts = 0
        
        while current_beat < total_beats and attempts < max_attempts:
            attempts += 1
            
            # Sonraki notayı üret
            pitch_class = self._sample_pitch(current_pitch % 12, temperature)
            octave = self._choose_octave(pitch_class, current_pitch, style)
            midi_pitch = pitch_class + (octave * 12)
            
            duration = self._sample_duration(style, current_beat, beats_per_bar)
            
            # Sınır kontrolü
            if current_beat + duration > total_beats:
                duration = total_beats - current_beat
            
            if duration <= 0:
                break
            
            # Nota ekle
            n = note.Note(midi=midi_pitch)
            n.duration.quarterLength = duration
            part.append(n)
            
            notes.append({
                'pitch': midi_pitch,
                'duration': duration,
                'beat': current_beat
            })
            
            current_pitch = midi_pitch
            current_beat += duration
        
        score.append(part)
        return score, notes
    
    def _sample_pitch(self, current_pc, temperature=1.0):
        """Markov modelinden sonraki pitch class'ı örnekle"""
        probs = self.markov.pitch_transitions[current_pc].copy()
        
        # Temperature uygula
        if temperature != 1.0:
            probs = np.power(probs + 1e-8, 1.0 / temperature)
        
        probs = probs / (probs.sum() + 1e-8)
        return int(np.random.choice(12, p=probs))
    
    def _choose_octave(self, pitch_class, prev_pitch, style):
        """
        Tutarlı oktav seçimi.
        Önceki notadan çok uzaklaşma.
        """
        prev_octave = prev_pitch // 12
        
        # Tercih edilen oktav aralığı (solo piyano için)
        min_octave = 3
        max_octave = 6
        
        # En yakın oktavı seç
        candidate_octaves = range(min_octave, max_octave + 1)
        closest = min(candidate_octaves, 
                     key=lambda o: abs(pitch_class + o*12 - prev_pitch))
        return closest
    
    def _sample_duration(self, style, current_beat, beats_per_bar):
        """Stil'e uygun nota süresi seç"""
        if style == 'baroque':
            # Barok: Genellikle kısa, düzenli
            options = [0.5, 1.0, 1.5, 2.0]
            weights = [0.3, 0.4, 0.15, 0.15]
        elif style == 'romantic':
            # Romantik: Daha çeşitli, rubato hissi
            options = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
            weights = [0.1, 0.2, 0.1, 0.25, 0.15, 0.1, 0.05, 0.05]
        else:
            # Klasik: Dengeli
            options = [0.5, 1.0, 1.5, 2.0]
            weights = [0.25, 0.4, 0.2, 0.15]
        
        return np.random.choice(options, p=weights)
    
    def _get_tempo(self, style):
        tempos = {
            'baroque': random.randint(60, 100),
            'classical': random.randint(80, 130),
            'romantic': random.randint(50, 110),
            'late_romantic': random.randint(55, 100),
        }
        return tempos.get(style, 90)
    
    def _midi_to_key(self, midi_note):
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                     'F#', 'G', 'G#', 'A', 'A#', 'B']
        return key_names[midi_note % 12]
```

---

## 🎼 5.3 HARMONİ DESTEKLEYICI

Üretimi daha müzikal yapan kural tabanlı sistem.

```python
# src/generation/harmony_support.py

class HarmonySupport:
    """
    Temel müzik teorisi kurallarını uygular.
    Bu modül müzikolog danışmanla geliştirilmeli.
    """
    
    # Majör tonalite için diyatonik notalar
    DIATONIC_SCALE = {
        'C_major': [0, 2, 4, 5, 7, 9, 11],
        'G_major': [7, 9, 11, 0, 2, 4, 6],
        'F_major': [5, 7, 9, 10, 0, 2, 4],
        'A_minor': [9, 11, 0, 2, 4, 5, 7],
        'D_minor': [2, 4, 5, 7, 9, 10, 0],
    }
    
    # Güçlü kadans çiftleri (dominant → tonic)
    STRONG_CADENCES = [
        (7, 0),  # G → C
        (2, 7),  # D → G
        (9, 2),  # A → D
    ]
    
    def __init__(self, key='C_major'):
        self.key = key
        self.scale = self.DIATONIC_SCALE.get(key, self.DIATONIC_SCALE['C_major'])
    
    def is_diatonic(self, pitch_class):
        """Nota tonaliteye uyuyor mu?"""
        return pitch_class in self.scale
    
    def diatonic_probability_boost(self, base_probs, boost_factor=1.5):
        """
        Diyatonik notalara ekstra ağırlık ver.
        Üretimi daha tonal yapar.
        """
        boosted = base_probs.copy()
        for pc in self.scale:
            boosted[pc] *= boost_factor
        return boosted / boosted.sum()
    
    def suggest_resolution(self, tension_pc, key_pc=0):
        """
        Gerilimli bir nota için çözüm öner.
        Dominant → tonic, leading tone → tonic, vb.
        """
        # Yarım ton yukarı/aşağı çözüm
        resolutions = {
            11: 0,   # B → C (leading tone)
            4: 5,    # E → F (subdominant)
            6: 7,    # F# → G
        }
        return resolutions.get(tension_pc, key_pc)
    
    def phrase_ending_note(self):
        """Fraz bitişi için tonic notası öner"""
        return self.scale[0]  # Tonic
```

---

## 🔄 5.4 KISIT KONTROLLÜ ÜRETİM PIPELINE

```python
# src/generation/pipeline.py
from .generator import ClassicalMusicGenerator
from ..features.extractor import extract_all_features
from ..model.constraints import MusicalConstraints
import os
import tempfile

def generate_with_constraints(generator, constraints, 
                               style, length_bars=32,
                               max_attempts=10):
    """
    Kısıtları karşılayana kadar yeniden üret.
    """
    
    for attempt in range(max_attempts):
        # Üret
        temperature = 1.0 + (attempt * 0.1)  # Her denemede biraz daha rastgele
        score, notes = generator.generate(
            style=style,
            length_bars=length_bars,
            temperature=temperature
        )
        
        if len(notes) < 10:
            continue
        
        # Feature çıkar
        pitches = [n['pitch'] for n in notes]
        features = {
            'pitch_entropy': pitch_entropy(pitches),
            'consonance_score': consonance_score(pitches),
            'step_ratio': step_ratio(pitches),
        }
        
        # Kısıt kontrolü
        valid, checks = constraints.is_valid(features)
        
        if valid:
            print(f"✓ {attempt+1}. denemede geçerli üretim bulundu")
            return score, notes, features
        else:
            if attempt < max_attempts - 1:
                print(f"✗ Deneme {attempt+1}: Kısıtlar karşılanmadı, tekrar...")
    
    print(f"⚠ {max_attempts} denemede de tam kısıt sağlanamadı, en iyisi alındı")
    return score, notes, features


def batch_generate(style, n_samples=20, output_dir='results/generated_midi/'):
    """
    Belirli bir stil için toplu üretim.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Model yükle
    from ..model.markov_model import MusicMarkovModel
    from ..model.distribution_model import ComposerDistributionModel
    
    # (Önceden eğitilmiş modelleri yükle)
    markov = MusicMarkovModel()
    markov.load(f'results/models/markov_{style}.pkl')
    
    constraints = MusicalConstraints(style=style)
    generator = ClassicalMusicGenerator(markov, None, constraints)
    
    results = []
    
    for i in range(n_samples):
        print(f"\nÜretim {i+1}/{n_samples}...")
        
        score, notes, features = generate_with_constraints(
            generator, constraints, style
        )
        
        # MIDI'ya kaydet
        filename = f"{output_dir}{style}_{i+1:03d}.mid"
        score.write('midi', fp=filename)
        
        # Matematiksel sertifika
        certificate = {
            'filename': filename,
            'style': style,
            'n_notes': len(notes),
            'features': features,
            'generation_id': f"{style}_{i+1:03d}",
        }
        results.append(certificate)
        
        print(f"  Entropy: {features['pitch_entropy']:.3f}")
        print(f"  Consonance: {features['consonance_score']:.3f}")
        print(f"  Saved: {filename}")
    
    # Sertifikaları kaydet
    import json
    with open(f"{output_dir}{style}_certificates.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ {n_samples} eser üretildi: {output_dir}")
    return results
```

---

## 🧪 5.5 KALİTE FİLTRESİ

Üretilen eserlerin otomatik kalite değerlendirmesi.

```python
# src/generation/quality_filter.py
import numpy as np

def quality_score(notes, style, distribution_model, composer_models):
    """
    Üretilen müziğin otomatik kalite skoru (0-1).
    
    Metrikler:
    1. Matematiksel tutarlılık: Stil kısıtlarına uyum
    2. Dağılım benzerliği: Hedef bestecilere ne kadar yakın?
    3. Müzikal geçerlilik: Basit müzik teorisi kontrolleri
    """
    pitches = [n['pitch'] for n in notes]
    
    scores = {}
    
    # 1. Entropi skoru
    ent = pitch_entropy(pitches)
    style_ranges = {
        'baroque': (1.8, 2.4),
        'classical': (2.0, 2.6),
        'romantic': (2.3, 3.0),
    }
    low, high = style_ranges.get(style, (2.0, 2.6))
    if low <= ent <= high:
        scores['entropy'] = 1.0
    else:
        scores['entropy'] = max(0, 1 - min(abs(ent - low), abs(ent - high)) / 0.5)
    
    # 2. Aralık çeşitliliği
    intervals = np.abs(np.diff(pitches))
    interval_variety = len(np.unique(intervals)) / 13  # max 13 unique semitone
    scores['interval_variety'] = float(min(interval_variety, 1.0))
    
    # 3. Pitch aralığı kontrolü
    pitch_range = max(pitches) - min(pitches)
    if 12 <= pitch_range <= 36:
        scores['range'] = 1.0
    else:
        scores['range'] = max(0, 1 - abs(pitch_range - 24) / 24)
    
    # 4. Dağılım benzerliği (hedef stil composer'larına)
    style_composers = {
        'baroque': ['Bach', 'Handel', 'Vivaldi'],
        'classical': ['Mozart', 'Haydn', 'Beethoven'],
        'romantic': ['Chopin', 'Brahms', 'Schumann'],
    }
    target_composers = style_composers.get(style, ['Bach'])
    
    # Eğer composer modeli varsa
    if distribution_model and hasattr(distribution_model, 'profiles'):
        total_sim = 0
        count = 0
        for c in target_composers:
            if c in distribution_model.profiles:
                # Feature benzerliğini hesapla
                total_sim += 0.7  # placeholder
                count += 1
        scores['style_match'] = total_sim / max(count, 1)
    else:
        scores['style_match'] = 0.5
    
    # Ağırlıklı toplam
    weights = {'entropy': 0.3, 'interval_variety': 0.25, 
               'range': 0.2, 'style_match': 0.25}
    
    total = sum(weights[k] * scores[k] for k in weights)
    
    return float(total), scores

def filter_by_quality(generated_list, threshold=0.65):
    """Kalite filtresinden geçir"""
    passed = []
    rejected = []
    
    for item in generated_list:
        score, details = quality_score(item['notes'], item['style'], None, None)
        item['quality_score'] = score
        item['quality_details'] = details
        
        if score >= threshold:
            passed.append(item)
        else:
            rejected.append(item)
    
    print(f"Kalite filtresi: {len(passed)} geçti, {len(rejected)} elendi")
    return passed, rejected
```

---

## 🎭 5.6 STİL TRANSFER DENEYİ

```python
# Romantik bir tema → Barok matematiksel yapıyla yeniden üret
# Bu müzikolog danışmanla birlikte değerlendirilmeli

def style_transfer(source_midi, target_style, markov_models):
    """
    Kaynak eserin melodic konturunu koruyarak
    hedef stilin matematiksel kalıplarını uygula.
    """
    
    # Kaynak eseri parse et
    source_notes = parse_midi_to_notes(source_midi)
    source_intervals = np.diff([n['pitch'] for n in source_notes])
    
    # Hedef stil Markov modelini yükle
    target_markov = markov_models[target_style]
    
    # Interval konturunu koruyarak yeniden üret
    new_pitches = [source_notes[0]['pitch']]
    
    for interval in source_intervals:
        prev = new_pitches[-1]
        prev_pc = prev % 12
        
        # Hedef Markov'dan önerileri al
        probs = target_markov.pitch_transitions[prev_pc]
        
        # En yakın interval'e sahip notayı seç
        candidates = []
        for pc in range(12):
            proposed_pitch = prev - (prev % 12) + pc
            if abs(proposed_pitch - prev - interval) <= 2:
                candidates.append((pc, probs[pc]))
        
        if candidates:
            # En yüksek olasılıklı uygun notayı seç
            best_pc = max(candidates, key=lambda x: x[1])[0]
            octave = prev // 12
            new_pitch = best_pc + octave * 12
        else:
            # Sadece Markov'u takip et
            new_pitch = int(np.random.choice(12, p=probs)) + (prev // 12) * 12
        
        new_pitches.append(new_pitch)
    
    return new_pitches
```

---

## 📊 5.7 ÜRETİM SONUÇLARI ANALİZİ

```python
# Üretilen müziklerin istatistiksel analizi
import pandas as pd

def analyze_generated_set(certificates_file):
    """
    Üretilen eserlerin toplu analizi.
    """
    with open(certificates_file) as f:
        certs = json.load(f)
    
    df = pd.DataFrame([{
        'style': c['style'],
        'entropy': c['features']['pitch_entropy'],
        'consonance': c['features']['consonance_score'],
        'n_notes': c['n_notes'],
    } for c in certs])
    
    print("=== Üretim İstatistikleri ===")
    print(df.groupby('style').agg({
        'entropy': ['mean', 'std'],
        'consonance': ['mean', 'std'],
        'n_notes': 'mean'
    }).round(3))
    
    # Orijinal dataset ile karşılaştır
    print("\n=== Orijinal vs Üretilmiş ===")
    for style in df['style'].unique():
        gen_entropy = df[df['style']==style]['entropy'].mean()
        # orig_entropy = original_df[original_df['era']==style_to_era[style]]['pitch_entropy'].mean()
        print(f"{style}: Generated={gen_entropy:.3f}")
```

---

## ⚠️ FAZ 5 RİSKLERİ

| Risk | İhtimal | Çözüm |
|------|---------|-------|
| Üretilen müzik kulağa "yapay" gelir | Yüksek | Bu normal, FAZ 6'da ölçülür |
| Kısıt döngüsü → hiç üretim yapılamaz | Orta | Kısıtları gevşet, fallback ekle |
| MIDI dönüşüm hataları | Düşük | music21 genellikle iyi çalışır |
| Tonalite kaybı | Orta | Diyatonik boost ekle |

---

## 🏁 FAZ 5 TAMAMLANDI SAYILIR WHEN

- [ ] Generator çalışıyor (bug yok)
- [ ] 4 stil için üretim yapılabiliyor
- [ ] 50+ kalite filtreli MIDI üretilmiş
- [ ] Matematiksel sertifikalar çıkarılmış
- [ ] Stil transfer deneyi yapılmış
- [ ] Üretimler müzikolog danışmana dinletilmiş
- [ ] FAZ 6 için 30 örnek seçilmiş

---

## 🚀 FAZ 6'YA GEÇİŞ KOŞULU

> 30 generated MIDI hazır, müzikolog "müzikal olarak geçerli" demiş → FAZ 6'ya geç.

---

*Sonraki: [FAZ 6 — İnsan Deneyi](FAZ_6_Insan_Deneyi.md)*
