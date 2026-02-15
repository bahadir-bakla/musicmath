"""İnsan deneyi için temel veri şemaları."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Condition = Literal["original", "generated", "baseline"]


@dataclass
class Participant:
    participant_id: str
    age_group: str  # "18-24", "25-34", ...
    music_training_years: int
    is_musician: bool


@dataclass
class Stimulus:
    stimulus_id: str
    condition: Condition
    style: str
    filepath: str


@dataclass
class Response:
    participant_id: str
    stimulus_id: str
    condition: Condition
    likability: float
    beauty: float
    valence: float
    arousal: float
    perceived_era: str
    perceived_origin: str  # "human" / "computer" / "unknown"


__all__ = ["Condition", "Participant", "Stimulus", "Response"]

