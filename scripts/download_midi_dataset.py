#!/usr/bin/env python
"""
Piano-midi.de ve diğer kaynaklardan MIDI dataset'i indirme aracı.

Kullanım:
    python scripts/download_midi_dataset.py --composers bach,mozart,chopin --max-per-composer 20
"""

import argparse
import shutil
import time
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from music_math.core.logging import get_logger

logger = get_logger(__name__)


# Piano-midi.de besteci sayfaları
PIANO_MIDI_BASE = "http://www.piano-midi.de/"
COMPOSER_PAGES = {
    "bach": "bach.htm",
    "mozart": "mozart.htm", 
    "chopin": "chopin.htm",
    "debussy": "debussy.htm",
    "beethoven": "beethoven.htm",
    "liszt": "liszt.htm",
    "schubert": "schubert.htm",
    "brahms": "brahms.htm",
    "haydn": "haydn.htm",
    "schumann": "schumann.htm",
}


def download_file(url: str, output_path: Path, timeout: int = 30) -> bool:
    """Tek bir dosyayı indir."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        logger.warning("İndirme hatası: %s (%s)", url, e)
        return False


def scrape_midi_links(composer: str, max_files: int = 20) -> list[str]:
    """Piano-midi.de'den belirtilen besteci için MIDI linklerini scrape et."""
    page_url = PIANO_MIDI_BASE + COMPOSER_PAGES.get(composer.lower(), "")
    
    if not page_url:
        logger.error("Besteci bulunamadı: %s", composer)
        return []
    
    try:
        response = requests.get(page_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # MIDI dosyalarını bul (genelde .mid veya .midi uzantılı)
        midi_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.mid') or href.endswith('.midi'):
                full_url = urljoin(page_url, href)
                midi_links.append(full_url)
                
                if len(midi_links) >= max_files:
                    break
        
        logger.info("%s için %d MIDI linki bulundu", composer, len(midi_links))
        return midi_links
    
    except Exception as e:
        logger.error("Scraping hatası (%s): %s", composer, e)
        return []


def download_composer_midis(
    composer: str, 
    output_dir: Path, 
    max_files: int = 20,
    delay: float = 1.0
) -> int:
    """Belirtilen besteci için MIDI dosyalarını indir."""
    logger.info("İndirme başlıyor: %s (max=%d)", composer, max_files)
    
    links = scrape_midi_links(composer, max_files)
    if not links:
        logger.warning("MIDI linki bulunamadı: %s", composer)
        return 0
    
    composer_dir = output_dir / composer.lower()
    composer_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    for i, url in enumerate(tqdm(links, desc=f"Downloading {composer}")):
        filename = Path(url).name
        output_path = composer_dir / filename
        
        if output_path.exists():
            logger.debug("Dosya zaten mevcut, atlanıyor: %s", filename)
            downloaded += 1
            continue
        
        if download_file(url, output_path):
            downloaded += 1
            logger.debug("İndirildi: %s", filename)
        
        # Rate limiting
        if i < len(links) - 1:
            time.sleep(delay)
    
    logger.info("%s: %d / %d dosya indirildi", composer, downloaded, len(links))
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Piano-midi.de'den MIDI dosyaları indir"
    )
    parser.add_argument(
        "--composers",
        type=str,
        default="bach,mozart,chopin,beethoven,debussy",
        help="Virgülle ayrılmış besteci isimleri (örn: bach,mozart,chopin)"
    )
    parser.add_argument(
        "--max-per-composer",
        type=int,
        default=20,
        help="Her besteci için maksimum dosya sayısı"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/piano_midi",
        help="İndirilen dosyaların kayıt edileceği dizin"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="İstekler arası bekleme süresi (saniye)"
    )
    
    args = parser.parse_args()
    
    composers = [c.strip() for c in args.composers.split(',')]
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("MIDI Dataset İndirme Aracı")
    print("=" * 60)
    print(f"Besteciler: {', '.join(composers)}")
    print(f"Her besteci için max dosya: {args.max_per_composer}")
    print(f"Çıktı dizini: {output_dir}")
    print("=" * 60)
    
    total_downloaded = 0
    
    for composer in composers:
        if composer.lower() not in COMPOSER_PAGES:
            logger.warning("Geçersiz besteci: %s (atlanıyor)", composer)
            continue
        
        count = download_composer_midis(
            composer,
            output_dir,
            max_files=args.max_per_composer,
            delay=args.delay
        )
        total_downloaded += count
        
        # Besteciler arası bekleme
        time.sleep(2.0)
    
    print("\n" + "=" * 60)
    print(f"Toplam {total_downloaded} dosya indirildi!")
    print("=" * 60)
    print("\nŞimdi metadata oluşturmak için:")
    print("  python run_pipeline.py")


if __name__ == "__main__":
    main()
