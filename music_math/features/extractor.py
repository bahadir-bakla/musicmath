"""Altı katmanlı feature extraction pipeline'ı."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger
from music_math.core.types import NoteEvent
from music_math.data.loader import parse_midi_to_note_events
from music_math.features.harmony import extract_harmony_features
from music_math.features.interval import extract_interval_features
from music_math.features.pitch import extract_pitch_features
from music_math.features.rhythm import extract_rhythm_features
from music_math.features.spectral import spectral_features_from_pitch
from music_math.features.structural import extract_structural_features

logger = get_logger(__name__)

try:  # opsiyonel, yoksa sadece progress log yazarız
    from tqdm import tqdm
except Exception:  # pragma: no cover - opsiyonel bağımlılık
    tqdm = None


def extract_all_features(filepath: str | Path) -> Dict[str, float] | None:
    """
    Tek bir MIDI dosyası için tüm feature katmanlarını çıkar.
    """
    path = Path(filepath)
    events: List[NoteEvent] = parse_midi_to_note_events(path)
    if len(events) < 20:
        return None

    pitches = [e.pitch for e in events]
    durations = [e.duration for e in events]

    features: Dict[str, float] = {}
    features.update(extract_pitch_features(pitches))
    features.update(extract_interval_features(pitches))
    features.update(extract_harmony_features(pitches, durations))
    features.update(extract_rhythm_features(events))
    features.update(extract_structural_features(pitches))
    features.update(spectral_features_from_pitch(pitches))

    # Ensure both paths are absolute for proper relative_to calculation
    abs_path = Path(filepath).resolve()
    abs_root = CONFIG.paths.root.resolve()
    features["filepath"] = str(abs_path.relative_to(abs_root))
    return features


def build_feature_matrix(
    metadata_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """
    Metadata CSV'deki tüm eserler için feature matrix oluştur.
    """
    meta_path = Path(metadata_csv)
    out_path = Path(output_csv)

    df_meta = pd.read_csv(meta_path)
    if "quality_flag" in df_meta.columns:
        df_meta = df_meta[df_meta["quality_flag"] == 1]

    records: List[Dict[str, float]] = []

    iterable = df_meta.iterrows()
    if tqdm is not None:
        iterable = tqdm(iterable, total=len(df_meta), desc="Extracting features")

    for _, row in iterable:
        rel = row["file_path"]
        full_path = CONFIG.paths.root / rel
        try:
            feats = extract_all_features(full_path)
        except Exception as exc:  # pragma: no cover - robust pipeline
            logger.warning("Feature extraction hatası: %s (%s)", full_path, exc)
            feats = None

        if feats is None:
            continue

        # Metadata'dan ek kolonlar
        for col in ("composer", "era", "form"):
            if col in row:
                feats[col] = row[col]

        records.append(feats)

    df_features = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(out_path, index=False)

    logger.info(
        "Feature matrix oluşturuldu: shape=%s | dosya=%s",
        df_features.shape,
        out_path,
    )

    return df_features


__all__ = ["extract_all_features", "build_feature_matrix"]

