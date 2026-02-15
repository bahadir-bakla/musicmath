"""Kısıt kontrollü generatif üretim pipeline'ı."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger
from music_math.core.types import NoteEvent
from music_math.features.harmony import consonance_score
from music_math.features.interval import step_ratio
from music_math.features.pitch import pitch_entropy
from music_math.generation.generator import ClassicalMusicGenerator, save_score_as_midi
from music_math.generation.quality_filter import QualityConfig, quality_score
from music_math.model.constraints import MusicalConstraints

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    filepath: Path
    style: str
    notes: List[NoteEvent]
    features: Dict[str, float]
    quality_score: float


def _basic_feature_summary(notes: List[NoteEvent]) -> Dict[str, float]:
    pitches = [n.pitch for n in notes]
    return {
        "pitch_entropy": pitch_entropy(pitches),
        "consonance_score": consonance_score(pitches),
        "step_ratio": step_ratio(pitches),
    }


def generate_with_constraints(
    generator: ClassicalMusicGenerator,
    constraints: MusicalConstraints,
    style: str,
    length_bars: int = 32,
    max_attempts: int = 10,
) -> Tuple[ClassicalMusicGenerator, List[NoteEvent], Dict[str, float]]:
    """
    Kısıtları sağlayan bir pasaj bulana kadar tekrar üret.
    """
    last_score = None
    last_notes: List[NoteEvent] = []
    last_feats: Dict[str, float] = {}

    for attempt in range(max_attempts):
        temperature = 1.0 + attempt * 0.1
        score, notes = generator.generate(style=style, length_bars=length_bars, temperature=temperature)
        if len(notes) < 10:
            continue

        feats = _basic_feature_summary(notes)
        valid, _ = constraints.is_valid(feats)
        last_score = score
        last_notes = notes
        last_feats = feats

        if valid:
            logger.info("Kısıtlar %d. denemede sağlandı.", attempt + 1)
            return last_score, last_notes, last_feats

    logger.warning(
        "Maksimum deneme sayısında tam kısıt sağlanamadı, en son üretilen pasaj döndürülüyor."
    )
    if last_score is None:
        score, notes = generator.generate(style=style, length_bars=length_bars)
        feats = _basic_feature_summary(notes)
        return score, notes, feats
    return last_score, last_notes, last_feats


def batch_generate(
    style: str,
    markov_model,
    n_samples: int = 20,
    output_dir: str | Path | None = None,
    quality_cfg: QualityConfig | None = None,
) -> List[GenerationResult]:
    """
    Belirli bir stil için toplu üretim ve kalite filtresi.
    """
    if quality_cfg is None:
        quality_cfg = QualityConfig()

    if output_dir is None:
        output_dir = CONFIG.paths.generated_midi
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    constraints = MusicalConstraints(style=style)
    generator = ClassicalMusicGenerator(markov_model=markov_model)

    raw_results: List[Dict] = []

    for i in range(n_samples):
        logger.info("Üretim %d/%d (%s)", i + 1, n_samples, style)
        score, notes, feats = generate_with_constraints(
            generator, constraints, style=style
        )
        filename = out_root / f"{style}_{i+1:03d}.mid"
        save_score_as_midi(score, filename)

        raw_results.append(
            {
                "filepath": filename,
                "style": style,
                "notes": notes,
                "features": feats,
            }
        )

    passed, _rejected = [], []
    for item in raw_results:
        q, details = quality_score(item["notes"], item["style"], quality_cfg)
        item["quality_score"] = q
        item["quality_details"] = details
        passed.append(
            GenerationResult(
                filepath=item["filepath"],
                style=item["style"],
                notes=item["notes"],
                features=item["features"],
                quality_score=q,
            )
        )

    return passed


__all__ = ["GenerationResult", "generate_with_constraints", "batch_generate"]

