## Klasik Müziğin Matematiksel DNA'sı

Bu repo, klasik müziğin **matematiksel yapısını** analiz eden, bu yapıdan **generatif müzik** üreten ve sonuçları **insan deneyi** ile test eden araştırma projesinin kod tabanını içerir.

### Amaç

- Çok katmanlı feature engineering ile her eseri ~80–100 boyutlu matematiksel vektöre dönüştürmek.
- Besteci ve dönem bazında **matematiksel imzaları** ortaya çıkarmak.
- Bu modelleri kullanarak **stil kontrollü generatif müzik** üretmek.
- Üretilen müziğin insan beğenisi ve algısı ile ilişkisini ölçmek.

Genel yol haritası ve araştırma vizyonu için `GENEL_PROJE_PLANI.md` ve faz dokümanlarına bakabilirsiniz.

### Kurulum

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U pip
pip install .
# geliştirme bağımlılıkları için:
pip install .[dev]
```

Geliştirme sırasında editable kurulum isterseniz:

```bash
pip install -e .[dev]
```

### Dizin Yapısı

- `music_math/` – ana Python paketi
  - `core/` – konfig, logging, ortak tipler
  - `data/` – veri ingestion, metadata, temizleme
  - `features/` – 6 katmanlı feature extraction
  - `model/` – dağılım modelleri, Markov, güzellik fonksiyonu, kısıtlar
  - `generation/` – generatif müzik üretimi ve kalite filtresi
  - `experiment/` – insan deneyi veri modeli ve analiz araçları
  - `viz/` – görselleştirme ve paper figürleri
  - `cli/` – komut satırı entrypoint’leri
- `data/raw/` – ham MIDI / MusicXML vs.
- `data/clean/` – filtrelenmiş / normalize MIDI
- `notebooks/` – keşif amaçlı Jupyter notebook’ları
- `results/` – figürler, üretilmiş MIDI’ler, istatistikler, interaktif HTML
- `configs/` – yol ve hiperparametre konfig dosyaları
- `tests/` – ünite ve entegrasyon testleri
- `paper/` – makale taslakları ve figürler

### Komut Örnekleri (ileride)

Kod tabanı geliştikçe aşağıdaki CLI komutları eklenecektir:

- `mm-clean-data` – metadata’dan temiz dataset oluşturur.
- `mm-build-features` – feature matrix (`feature_matrix.csv`) üretir.
- `mm-train-models` – matematiksel modelleri eğitir.
- `mm-generate` – belirli bir stil için MIDI üretir.
- `mm-make-figures` – paper figür setini üretir.

