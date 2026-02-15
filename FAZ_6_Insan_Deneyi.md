# 🟠 FAZ 6 — İNSAN DEĞERLENDİRME DENEYİ

> **Süre:** 6–8 hafta  
> **Önceki Faz:** FAZ 5 — Generatif Müzik  
> **Sonraki Faz:** FAZ 7 — Görselleştirme

---

## 🎯 FAZ AMACI

"Matematiksel model gerçekten işe yarıyor mu?" sorusunu **bilimsel olarak** yanıtlamak. Üretilen müzik insanları orijinal müzik gibi etkiliyor mu?

Bu faz aynı zamanda projenin "güzellik fonksiyonunu" kalibre etmek için veri toplar.

---

## ✅ FAZ ÇIKTILARI

- [ ] Deney protokolü dokümanı
- [ ] Online anket (Google Forms veya özel)
- [ ] 50-100 katılımcı verisi
- [ ] İstatistiksel analiz
- [ ] Güzellik fonksiyonu kalibrasyon güncellemesi
- [ ] Paper'ın Results & Discussion bölümü

---

## 🔬 6.1 DENEY TASARIMI

### Araştırma Soruları

- **RQ-A:** Üretilen müzik, orijinal klasik müzikle karşılaştırıldığında beğeni puanı açısından istatistiksel olarak farklı mı?
- **RQ-B:** Hangi matematiksel özellikler (entropi, konsonans, tekrar) insan beğenisiyle korelasyonlu?
- **RQ-C:** Katılımcılar hangi eserin "bilgisayar yapımı" olduğunu tahmin edebiliyor mu?

### Deney Düzeni

```
3 Koşul (within-subjects, counterbalanced):
  A) Orijinal klasik eserler (ground truth)
  B) Model-generated eserler (test)
  C) Random/unstructured baseline (kontrol)

Her koşulda: 10 snippet (30-45 saniye)
Toplam stimulus: 30 snippet
```

### Katılımcı Profili

```
Hedef: 60-100 kişi
Gruplar:
  - Müzik eğitimi almış (20-30 kişi)
  - Sıradan dinleyici (20-30 kişi)
  - Müzik teorisi bilen akademisyen (10-20 kişi)

Recruitement:
  - Tanıdık çevre
  - Üniversite duyuruları
  - Online (Reddit classical music, müzik forumları)
```

---

## 📋 6.2 ÖLÇÜM ARAÇLARI

### Ölçek 1: Beğeni Puanı (Aesthetics)

```
Q1: Bu müziği ne kadar beğendiniz?
    1 (Hiç beğenmedim) — 2 — 3 — 4 — 5 — 6 — 7 (Çok beğendim)
    
Q2: Bu müzik ne kadar "güzel" hissettirdi?
    1 — 2 — 3 — 4 — 5 — 6 — 7
```

### Ölçek 2: Duygusal Etki (Valence-Arousal)

```
Q3: Bu müzik sizi duygusal olarak nasıl etkiledi?
    [SAM (Self-Assessment Manikin) görseli]
    Valence: -3 (çok olumsuz) → +3 (çok olumlu)
    Arousal: -3 (çok sakin) → +3 (çok heyecanlı)
    
Q4: Bu müziği dinlerken aşağıdakilerden hangisini hissettiniz?
    □ Hüzün  □ Sevinç  □ Huzur  □ Merak  □ Sıkıntı
    □ Heyecan  □ Nostalji  □ Hiçbir şey
```

### Ölçek 3: Algısal Değerlendirme

```
Q5: Bu müziğin hangi döneme ait olduğunu düşünüyorsunuz?
    □ 1600-1750 (Barok)  □ 1750-1820 (Klasik)
    □ 1820-1900 (Romantik)  □ 1900+ (Modern)
    □ Bilmiyorum

Q6: Bu müzik insan tarafından mı yoksa bilgisayar tarafından mı yapıldı?
    □ Kesinlikle insan  □ Büyük ihtimalle insan
    □ Emin değilim  □ Büyük ihtimalle bilgisayar
    □ Kesinlikle bilgisayar

Q7: Bu müziği daha önce duydunuz mu?
    □ Evet  □ Hayır  □ Emin değilim
```

### Ölçek 4: Matematiksel Korelat (Post-Test)

```
Q8-10 (sadece müzik eğitimi alanlara):
  "Bu eserde harmonik karmaşıklık ne kadar yüksekti?"  (1-7)
  "Bu eser ne kadar tahmin edilebilirdi?"  (1-7)
  "Bu eserde ne kadar tekrar vardı?"  (1-7)
```

---

## 🎧 6.3 STİMULİ HAZIRLAMA

### Snippet Seçim Kriterleri

```python
def select_stimuli(original_df, generated_list, n_per_condition=10):
    """
    Deney için dengeli stimulus seti oluştur.
    """
    
    # Orijinal eserler
    # - Her dönemden 2-3 tane
    # - Tanınmış eserlerden kaçın! (Q7 bunu kontrol ediyor)
    # - 30-45 saniye snippet (en dramatik bölüm değil, orta kısım)
    
    originals_baroque = original_df[original_df['era']=='Baroque'].sample(3)
    originals_classical = original_df[original_df['era']=='Classical'].sample(3)
    originals_romantic = original_df[original_df['era']=='Romantic'].sample(2)
    originals_late = original_df[original_df['era']=='Late Romantic'].sample(2)
    
    # Generated eserler
    # - Kalite filtresi geçmiş olanlardan
    # - Farklı stillerden
    
    # Baseline (rastgele)
    # - Aynı pitch range
    # - Aynı uzunluk
    # - Ama yapısız
    
    return {
        'originals': [...],
        'generated': [...],
        'baseline': [...],
    }
```

### Audio Dönüşümü

```python
# MIDI → WAV/MP3
# MuseScore, Fluidsynth veya Python kütüphaneleri

import subprocess

def midi_to_audio(midi_file, soundfont, output_wav):
    """
    Fluidsynth ile MIDI'yı gerçek enstrüman sesiyle çevir.
    """
    cmd = [
        'fluidsynth',
        '-ni',
        soundfont,  # Steinway piyano soundfont (MuseScore'dan)
        midi_file,
        '-F', output_wav,
        '-r', '44100'
    ]
    subprocess.run(cmd)

# Tüm stimuli'yi hazırla
for stimulus_file in all_stimuli:
    output = stimulus_file.replace('.mid', '.wav')
    midi_to_audio(stimulus_file, 'steinway.sf2', output)
    
    # 30-45 saniyeye kırp
    crop_audio(output, start=10, duration=35)  # başlangıcı atla
```

---

## 💻 6.4 DENEY PLATFORMU

### Google Forms Yöntemi (Basit)

```
Form Yapısı:
  - Bölüm 1: Demografik + müzik geçmişi
  - Bölüm 2-4: Her stimulus için ölçekler
    (Her snippet ayrı bölümde, embedded audio player)
  - Bölüm 5: Genel değerlendirme

Avantajları: Hızlı kurulum, veri otomatik toplanır
Dezavantajları: Audio embed sınırlı
```

### Özel Web Uygulaması (Önerilen)

```python
# Flask + JavaScript ile basit deney arayüzü

# Temel özellikler:
# - Rastgele sıralama (counterbalancing)
# - Zorunlu tam dinleme (replay sayısı log)
# - Responsetime ölçümü
# - Mobil uyumlu

# Yapı:
# GET /experiment → Stimulus sırası ata, session başlat
# POST /response → Her stimulus için yanıt kaydet
# GET /done → Teşekkür sayfası
```

---

## 📊 6.5 İSTATİSTİKSEL ANALİZ

### Temel Analizler

```python
import scipy.stats as stats
import pandas as pd
import numpy as np

def analyze_experiment_results(results_csv):
    df = pd.read_csv(results_csv)
    
    print("=== TEMEL İSTATİSTİKLER ===")
    
    # 1. Koşul bazında ortalama beğeni
    for condition in ['original', 'generated', 'baseline']:
        mask = df['condition'] == condition
        mean = df[mask]['likability'].mean()
        std = df[mask]['likability'].std()
        print(f"{condition}: mean={mean:.2f} ± {std:.2f}")
    
    # 2. ANOVA: Koşullar arasında fark var mı?
    orig = df[df['condition']=='original']['likability']
    gen = df[df['condition']=='generated']['likability']
    base = df[df['condition']=='baseline']['likability']
    
    f_stat, p_value = stats.f_oneway(orig, gen, base)
    print(f"\nANOVA: F={f_stat:.3f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        print("✓ Koşullar arasında anlamlı fark var (p < 0.05)")
    
    # 3. Post-hoc: Generated vs Original
    t_stat, p_t = stats.ttest_ind(gen, orig)
    effect_size = (gen.mean() - orig.mean()) / np.sqrt(
        (gen.std()**2 + orig.std()**2) / 2
    )  # Cohen's d
    
    print(f"\nGenerated vs Original:")
    print(f"  t={t_stat:.3f}, p={p_t:.4f}")
    print(f"  Cohen's d (effect size): {effect_size:.3f}")
    
    # 4. Turing Test analizi
    turing_df = df[df['question']=='turing_test']
    computer_rate = (turing_df['answer'] >= 4).mean()  # "Büyük ihtimalle bilgisayar" veya "Kesinlikle bilgisayar"
    print(f"\nTuring Test: Katılımcıların {computer_rate:.1%}'i üretimi 'bilgisayar' olarak tanımladı")
    
    return df

def correlation_analysis(results_df, features_df):
    """
    Matematiksel feature'lar ile insan beğenisi korelasyonu.
    Bu güzellik fonksiyonunu kalibre eder.
    """
    
    # Her stimulus için feature'ları eşleştir
    merged = results_df.merge(features_df, on='stimulus_id')
    
    features_to_test = [
        'pitch_entropy', 'consonance_score', 'repetition_index',
        'fractal_dimension', 'interval_entropy', 'rhythmic_entropy'
    ]
    
    print("=== FEATURE-BEĞENİ KORELASYONLARI ===")
    correlations = {}
    
    for feat in features_to_test:
        if feat in merged.columns:
            r, p = stats.pearsonr(merged[feat], merged['likability'])
            correlations[feat] = (r, p)
            significance = "✓" if p < 0.05 else "✗"
            print(f"{significance} {feat}: r={r:.3f}, p={p:.4f}")
    
    # Entropi ile beğeni ilişkisi - bu kritik!
    # Optimal entropi bandı var mı?
    entropy_range = np.linspace(
        merged['pitch_entropy'].min(),
        merged['pitch_entropy'].max(), 20
    )
    
    # Scatter plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(merged['pitch_entropy'], merged['likability'], 
               alpha=0.4, s=30)
    plt.xlabel("Pitch Entropy (bits)")
    plt.ylabel("Beğeni Puanı (1-7)")
    plt.title("Entropi-Güzellik İlişkisi")
    plt.grid(alpha=0.3)
    
    # Polynomial fit
    z = np.polyfit(merged['pitch_entropy'], merged['likability'], 2)
    p = np.poly1d(z)
    x_line = np.linspace(merged['pitch_entropy'].min(), 
                         merged['pitch_entropy'].max(), 100)
    plt.plot(x_line, p(x_line), 'r-', linewidth=2)
    plt.savefig('results/figures/entropy_beauty_correlation.png', dpi=150)
    
    return correlations

def update_beauty_function(correlations):
    """
    İnsan deneyi bulgularına göre güzellik fonksiyonunu güncelle.
    """
    print("\n=== GÜZELLIK FONKSİYONU KALİBRASYONU ===")
    
    # En güçlü korelasyonları bul
    sorted_corr = sorted(
        [(feat, r, p) for feat, (r, p) in correlations.items()],
        key=lambda x: abs(x[1]), reverse=True
    )
    
    print("Güzellikle en güçlü korelasyonlu feature'lar:")
    for feat, r, p in sorted_corr[:5]:
        direction = "↑" if r > 0 else "↓"
        print(f"  {direction} {feat}: r={r:.3f}")
    
    # Güncellenen ağırlıklar
    print("\nÖnerilen güncelleme: FAZ 4 güzellik fonksiyonuna bu ağırlıkları yansıt")
```

---

## 👥 6.6 ETİK VE KATILIMCI HAKLARI

### Bilgilendirilmiş Onay
```
Deney başlamadan önce katılımcılara bildir:
  - Çalışmanın amacı (genel olarak)
  - Ses dosyaları dinleteceğiz
  - Kişisel veri toplanmıyor
  - İstedikleri zaman çıkabilirler
  - Sonuçlar akademik amaçlı
  
IRB (Institutional Review Board) gerekebilir
→ Üniversite bağlantısın varsa danış
```

### Veri Anonimizasyon
```python
# Katılımcı ID rastgele ata, isim alma
participant_id = generate_random_id()

# Demografik veri sadece:
# - Yaş grubu (10'lu aralıklar)
# - Müzik geçmişi (Evet/Hayır)
# - Toplam müzik eğitimi süresi (yıl)
```

---

## ⚠️ FAZ 6 RİSKLERİ

| Risk | İhtimal | Çözüm |
|------|---------|-------|
| Yeterli katılımcı bulamazsın | Orta | Önce küçük pilot (15-20 kişi) yap |
| Tanınmış eserler deneyimi bozar | Orta | Q7 ile kontrol et, tanınan verileri filtrele |
| Üretilen müzik tamamen reddedilir | Düşük | Bu da bir bulgu olur! "Neden reddedildi?" analiz et |
| Audio kalitesi kötü olur | Düşük | İyi soundfont kullan, headphone öner |

---

## 🏁 FAZ 6 TAMAMLANDI SAYILIR WHEN

- [ ] Pilot çalışma (15 kişi) tamamlanmış
- [ ] Anket aracı düzeltilmiş
- [ ] Tam deney (60+ kişi) tamamlanmış
- [ ] İstatistiksel analizler yapılmış
- [ ] Feature-beğeni korelasyonları hesaplanmış
- [ ] Güzellik fonksiyonu kalibre edilmiş
- [ ] Bulgular müzikolog danışmanla tartışılmış
- [ ] Paper Results bölümü yazılmaya başlandı

---

## 🚀 FAZ 7'YE GEÇİŞ KOŞULU

> Deney verileri analiz edilmiş ve en az 1 anlamlı hipotez test sonucu (p < 0.05) elde edilmişse → FAZ 7'ye geç.

---

*Sonraki: [FAZ 7 — Görselleştirme](FAZ_7_Gorsellestirme.md)*
