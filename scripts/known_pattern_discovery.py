#!/usr/bin/env python
"""
Bilinen matematiksel oruntuleri tum eserler uzerinde test eder.
Cikti: results/stats/known_pattern_scores.csv

Kullanim:
    python scripts/known_pattern_discovery.py
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger
from music_math.data.loader import parse_midi_to_note_events
from music_math.analysis.known_patterns import analyze_known_patterns


logger = get_logger(__name__)


def main() -> None:
    meta_clean = CONFIG.paths.root / "metadata_clean.csv"
    if not meta_clean.exists():
        raise FileNotFoundError(f"metadata_clean.csv bulunamadi: {meta_clean}")

    df_meta = pd.read_csv(meta_clean)
    rows = []

    for idx, row in df_meta.iterrows():
        path = CONFIG.paths.root / row["file_path"]
        if not path.exists():
            continue
        try:
            events = parse_midi_to_note_events(path)
            if len(events) < 20:
                continue
            scores = analyze_known_patterns(events)
            record = {
                "filepath": row["file_path"],
                "composer": row.get("composer", ""),
                "era": row.get("era", ""),
                **scores,
            }
            rows.append(record)
        except Exception as e:
            logger.warning("Hata %s: %s", path, e)
            continue

    df = pd.DataFrame(rows)
    out_path = CONFIG.paths.root / "results" / "stats" / "known_pattern_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("known_pattern_scores.csv yazildi: %s (%d eser)", out_path, len(df))

    if "era" in df.columns:
        print("\nEra bazinda ornek ortalama skorlar (Fibonacci aralik, kumulatif asal, climax golden):")
        agg = df.groupby("era")[
            ["fibonacci_interval_ratio", "cumulative_prime_ratio", "climax_near_golden"]
        ].mean()
        print(agg.round(4).to_string())


if __name__ == "__main__":
    main()
