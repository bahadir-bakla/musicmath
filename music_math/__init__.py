"""
music_math
==========

Klasik müziğin matematiksel DNA'sını analiz eden ve generatif müzik
üreten araştırma projesinin ana Python paketi.

Alt modüller:
- core: Konfigürasyon, logging ve ortak tipler
- data: Veri ingestion, metadata ve temizleme pipeline'ı
- features: 6 katmanlı feature extraction
- model: Dağılım ve Markov modelleri, güzellik fonksiyonu ve kısıtlar
- generation: Generatif müzik üretimi ve kalite filtresi
- experiment: İnsan deneyi veri modeli ve analiz araçları
- viz: Görselleştirme ve paper figürleri
- cli: Komut satırı arayüzleri
"""

__all__ = [
    "core",
    "data",
    "features",
    "model",
    "generation",
    "experiment",
    "viz",
    "cli",
]

