# Music Analysis Projesi — Dizin Yapısı (Claude için)

Bu doküman proje dizin yapısını özetler. Backend dockerize, worker+queue+redis, frontend build sorunları için referans.

## Kök Yapı

```
music_analysisi/
├── metadata_clean.csv          # Eser metadata (file_path, composer, era, vb.)
├── docker-compose.yml          # Backend + Frontend container
├── DOCKER.md
├── README.md
├── ADVANCED_ANALYSIS_GUIDE.md
├── PROJECT_STRUCTURE.md        # Bu dosya
│
├── music_math/                 # Ana Python paketi
│   ├── analysis/               # pattern_miner, known_patterns, golden_ratio, prime_harmony
│   ├── core/                   # config, logging, types
│   ├── data/                   # loader, ingest, pipeline, quality
│   ├── features/               # extractor, harmony, interval, pitch, rhythm, spectral
│   ├── generation/             # generator, pipeline, quality_filter
│   ├── model/                  # beauty, markov, distribution
│   └── viz/                    # heatmap, ssm, interactive
│
├── scripts/                    # CLI ve analiz scriptleri
│   ├── brute_force_patterns.py # Ana giriş (motif + export-patterns)
│   ├── brute_force_motif.py    # Motif analizi (chunk, paralel, merge)
│   ├── known_pattern_discovery.py
│   ├── advanced_analysis.py
│   ├── download_maestro.py
│   └── ...
│
├── results/
│   └── stats/                  # CSV çıktıları
│       ├── brute_force_pattern_summary.csv
│       ├── brute_force_pattern_summary_checkpoint.csv  # Kurtarma
│       └── brute_force_patterns_detail.csv
│
├── backend/                     # FastAPI backend
│   ├── Dockerfile
│   ├── server.py
│   ├── algorithms.py
│   ├── artwork.py
│   └── requirements.txt
│
├── frontend/                   # React + Vite
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│
├── data/                       # MIDI dosyaları
│   └── raw/
│       └── piano_midi/maestro-v3.0.0/...
│
└── pyproject.toml / requirements.txt
```

## Önemli Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `scripts/brute_force_motif.py` | Motif analizi. `--chunk N --total-chunks M`, `--merge-chunks` |
| `scripts/brute_force_patterns.py` | Ana giriş. Motif veya `--export-patterns` |
| `music_math/analysis/pattern_miner.py` | `mine_patterns_one_piece()` — interval n-gram motifleri |
| `music_math/data/loader.py` | `parse_midi_to_note_events()` |
| `backend/server.py` | FastAPI sunucu |
| `docker-compose.yml` | Backend:8002, Frontend:3002 |

## Brute Force Script Kullanımı

```bash
# Tek komutla tüm eserler (paralel)
python scripts/brute_force_patterns.py

# Chunk ile 4 terminalde ayrı çalıştır (daha hızlı)
python scripts/brute_force_motif.py --chunk 0 --total-chunks 4
python scripts/brute_force_motif.py --chunk 1 --total-chunks 4
python scripts/brute_force_motif.py --chunk 2 --total-chunks 4
python scripts/brute_force_motif.py --chunk 3 --total-chunks 4
# Sonra birleştir
python scripts/brute_force_motif.py --merge-chunks --total-chunks 4

# 100 matematiksel örüntü export
python scripts/brute_force_patterns.py --export-patterns
```

## Yapılacaklar (Özet)

1. **Backend Dockerize + Worker/Queue/Redis**: Celery veya RQ ile brute-force işlerini arka planda çalıştırma
2. **Frontend Docker**: `frontend/Dockerfile` ile build (Vite)
3. **Parametreler**: Şu an `max_length=6`, `n_shuffles=5`, `top_k=20` (hızlı mod)
