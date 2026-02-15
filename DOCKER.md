# MusicMath DNA – Docker

Frontend (React) ve backend (FastAPI) birlikte çalıştırılır. Backend, proje kökündeki `results/stats` ve `metadata_clean.csv` dosyalarını okur.

## Gereksinimler

- Proje kökünde en az şunlar olmalı:
  - `results/stats/feature_matrix.csv` veya `results/stats/feature_matrix_with_numeric_patterns.csv`
  - İsteğe bağlı: `results/stats/signature_scores.csv` (imza metrikleri için)

## Çalıştırma

Proje kökünden:

```bash
docker compose up --build
```

- **Frontend:** http://localhost:3002  
- **Backend API:** http://localhost:8002  
- **API docs:** http://localhost:8002/docs  

Portlar 8002 / 3002 (8000 ve 3000 baska uygulama tarafindan kullaniliyorsa). Tarayıcıda http://localhost:3002 açıldığında istekler aynı host üzerinden gider; nginx `/api` isteklerini backend’e proxy eder.

## Ortam değişkenleri

### Backend

| Değişken       | Açıklama |
|----------------|----------|
| `DATA_DIR`     | Proje kökü (CSV’lerin bulunduğu dizin). Compose’ta `.:/data` mount edilir. |
| `CORS_ORIGINS` | İzin verilen origin’ler (virgülle ayrılmış). Varsayılan: `*` |
| `MONGO_URL`    | Opsiyonel. Tanımlıysa `/api/status` MongoDB kullanır. |

### Frontend (build sırasında)

| Değişken                  | Açıklama |
|---------------------------|----------|
| `REACT_APP_API_URL`       | Boş = aynı origin (nginx proxy). Ayrı backend için örn. `http://localhost:8000` |
| `REACT_APP_USE_REAL_API`  | `true` = her zaman API kullan. Yoksa production build’de API, dev’de mock kullanılır. |

## API endpoint’leri (ozet)

| Endpoint | Açıklama |
|----------|----------|
| `GET /api/tracks` | Parça listesi (Playground) |
| `GET /api/analysis/{track_id}` | Tek parça analiz sonucu |
| `GET /api/health` | Saglik + data_dir |
| `GET /api/patterns/mathematical` | 100 matematiksel oruntu (ozet veya full) |
| `GET /api/patterns/mathematical/{id}` | Tek oruntu deger serisi |
| `GET /api/gallery/visualizations` | Gallery: tsne, confusion, signatureByEra, numericTimeline |
| `GET /api/artwork/config` | Paletler + render modlari + varsayilan boyut |
| `GET /api/artwork/palettes` | Renk paletleri (muzik -> gorsel sanat) |
| `GET /api/artwork/render-modes` | kalman, spectral, waveform, phi_arc, fractal |
| `GET /api/artwork/preview/{track_id}` | Parça icin cizim onerisi (stub; ileride gorsel) |
| **Experiment (algoritma + oruntu deneyleri)** | |
| `GET /api/experiment/algorithms` | Kalman, EMA, Gaussian, Savitzky-Golay, Rolling median, Double exp, Wiener + parametre semalari |
| `GET /api/experiment/features` | Track ozelligi secimi icin sayisal kolon isimleri |
| `GET /api/experiment/patterns` | Matematiksel oruntu listesi (pattern_id) |
| `POST /api/experiment/run` | Kaynak (track/pattern/custom) + algoritma + params; input/output seri + ozet doner |

## Lokal geliştirme (Docker’sız)

- **Build sorun giderme**: Frontend OOM ise Dockerfile'da NODE_OPTIONS=8192; scipy kurulamazsa requirements'tan scipy satirini sil (Wiener fallback var). Lock yoksa da npm install kullaniliyor; CI=false ile ESLint uyarilari build'i kirilmaz.
- Backend: `cd frontend/backend && pip install -r requirements.txt && DATA_DIR=../.. uvicorn server:app --reload`
- Frontend: `cd frontend/frontend && yarn start` (mock veri)
- Gerçek API ile: `REACT_APP_API_URL=http://localhost:8002 REACT_APP_USE_REAL_API=true yarn start`
