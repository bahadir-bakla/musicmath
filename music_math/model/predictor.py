"""
Kalman-Enhanced Note Predictor with Emotional Impact Scoring.

Markov gecis olasiliklari + Kalman filtre trajectory tahmini +
matematiksel oruntu uyumu (Fibonacci, altin oran, asal) +
guzellik/etki fonksiyonu ile "bir sonraki nota" onerisi.

Amac: Insanda duygusal tepki tetikleyen nota dizilerini bulmak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math

import numpy as np

from music_math.core.types import NoteEvent
from music_math.features.harmony import CONSONANCE_MAP


PHI = (1 + math.sqrt(5)) / 2
FIBONACCI_INTERVALS = {1, 2, 3, 5, 8, 13}
PRIME_INTERVALS = {2, 3, 5, 7, 11}


@dataclass
class PredictorConfig:
    w_markov: float = 0.30
    w_kalman: float = 0.15
    w_consonance: float = 0.15
    w_tension: float = 0.10
    w_fibonacci: float = 0.10
    w_golden_position: float = 0.10
    w_beauty: float = 0.10

    kalman_process_var: float = 0.02
    kalman_measure_var: float = 0.08
    tension_curve_peak: float = 0.618
    temperature: float = 1.0
    context_window: int = 16


@dataclass
class NoteSuggestion:
    pitch: int
    score: float
    scores_detail: Dict[str, float]
    predicted_duration: float
    emotional_tag: str


@dataclass
class KalmanState:
    x: float = 60.0
    p: float = 1.0
    process_var: float = 0.02
    measure_var: float = 0.08

    def predict(self) -> Tuple[float, float]:
        return self.x, self.p + self.process_var

    def update(self, measurement: float) -> float:
        x_pred, p_pred = self.predict()
        k = p_pred / (p_pred + self.measure_var)
        self.x = x_pred + k * (measurement - x_pred)
        self.p = (1 - k) * p_pred
        return self.x

    def forecast(self, steps: int = 1) -> List[float]:
        x, p = self.x, self.p
        trajectory = []
        for _ in range(steps):
            x_pred = x
            p = p + self.process_var
            trajectory.append(x_pred)
            x = x_pred
        return trajectory


@dataclass
class NotePredictor:
    config: PredictorConfig = field(default_factory=PredictorConfig)
    kalman: KalmanState = field(default_factory=KalmanState)
    context: List[NoteEvent] = field(default_factory=list)
    _markov_matrix: np.ndarray = field(
        default_factory=lambda: np.ones((12, 12), dtype=float) / 12.0
    )
    _piece_length: float = 0.0
    _current_position: float = 0.0

    def set_markov_matrix(self, matrix: np.ndarray) -> None:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self._markov_matrix = matrix / row_sums

    def feed_context(self, notes: List[NoteEvent], piece_total_length: float = 0.0) -> None:
        self.context = notes[-self.config.context_window:]
        if notes:
            self.kalman = KalmanState(
                x=float(notes[-1].pitch),
                p=1.0,
                process_var=self.config.kalman_process_var,
                measure_var=self.config.kalman_measure_var,
            )
            for n in self.context:
                self.kalman.update(float(n.pitch))
            last = notes[-1]
            self._current_position = last.start + last.duration
        self._piece_length = piece_total_length if piece_total_length > 0 else self._current_position

    def predict_next(self, top_k: int = 5) -> List[NoteSuggestion]:
        if not self.context:
            return []

        last_pitch = self.context[-1].pitch
        last_pc = last_pitch % 12
        last_octave = last_pitch // 12

        kalman_forecast = self.kalman.forecast(steps=1)[0]

        candidates = []
        for candidate_pitch in range(max(36, last_pitch - 24), min(97, last_pitch + 25)):
            pc = candidate_pitch % 12
            detail = {}

            detail["markov"] = float(self._markov_matrix[last_pc, pc])

            pitch_dist = abs(candidate_pitch - kalman_forecast)
            detail["kalman"] = float(math.exp(-0.5 * (pitch_dist / 6.0) ** 2))

            interval = abs(candidate_pitch - last_pitch) % 12
            detail["consonance"] = CONSONANCE_MAP.get(interval, 0.5)

            detail["tension"] = self._tension_score(candidate_pitch)

            detail["fibonacci"] = self._fibonacci_score(candidate_pitch)

            detail["golden_position"] = self._golden_position_score()

            detail["beauty"] = self._local_beauty_score(candidate_pitch)

            cfg = self.config
            total = (
                cfg.w_markov * detail["markov"]
                + cfg.w_kalman * detail["kalman"]
                + cfg.w_consonance * detail["consonance"]
                + cfg.w_tension * detail["tension"]
                + cfg.w_fibonacci * detail["fibonacci"]
                + cfg.w_golden_position * detail["golden_position"]
                + cfg.w_beauty * detail["beauty"]
            )

            if cfg.temperature != 1.0:
                total = total ** (1.0 / max(cfg.temperature, 0.1))

            duration = self._predict_duration()

            candidates.append(NoteSuggestion(
                pitch=candidate_pitch,
                score=total,
                scores_detail=detail,
                predicted_duration=duration,
                emotional_tag=self._emotional_tag(detail),
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def _tension_score(self, candidate_pitch: int) -> float:
        if self._piece_length <= 0:
            return 0.5
        position_ratio = self._current_position / self._piece_length
        peak = self.config.tension_curve_peak

        if position_ratio <= peak:
            target_tension = position_ratio / peak
        else:
            target_tension = 1.0 - (position_ratio - peak) / (1.0 - peak)
        target_tension = max(0.0, min(1.0, target_tension))

        if len(self.context) < 2:
            return target_tension

        pitches = [n.pitch for n in self.context[-8:]]
        local_range = max(pitches) - min(pitches)
        interval = abs(candidate_pitch - self.context[-1].pitch)
        local_tension = min(1.0, (local_range + interval) / 24.0)

        return 1.0 - abs(local_tension - target_tension)

    def _fibonacci_score(self, candidate_pitch: int) -> float:
        interval = abs(candidate_pitch - self.context[-1].pitch)
        if interval in FIBONACCI_INTERVALS:
            return 1.0
        if interval % 12 in FIBONACCI_INTERVALS:
            return 0.7
        closest = min(FIBONACCI_INTERVALS, key=lambda f: abs(interval - f))
        return max(0.0, 1.0 - abs(interval - closest) / 12.0)

    def _golden_position_score(self) -> float:
        if self._piece_length <= 0:
            return 0.5
        pos = self._current_position / self._piece_length
        golden_points = [1.0 / PHI, 1.0 / (PHI ** 2), 1.0 - 1.0 / PHI]
        min_dist = min(abs(pos - gp) for gp in golden_points)
        return max(0.0, 1.0 - min_dist * 5.0)

    def _local_beauty_score(self, candidate_pitch: int) -> float:
        recent = [n.pitch for n in self.context[-8:]] + [candidate_pitch]
        intervals = [abs(recent[i + 1] - recent[i]) % 12 for i in range(len(recent) - 1)]

        consonance_vals = [CONSONANCE_MAP.get(iv, 0.5) for iv in intervals]
        mean_consonance = sum(consonance_vals) / len(consonance_vals) if consonance_vals else 0.5

        unique_intervals = len(set(intervals))
        total_intervals = len(intervals) or 1
        variety = unique_intervals / min(total_intervals, 12)

        repetition = 1.0 - variety

        optimal_consonance = 0.65
        consonance_fit = 1.0 - abs(mean_consonance - optimal_consonance) * 2

        optimal_variety = 0.5
        variety_fit = 1.0 - abs(variety - optimal_variety) * 2

        return max(0.0, min(1.0, 0.4 * consonance_fit + 0.3 * variety_fit + 0.3 * repetition))

    def _predict_duration(self) -> float:
        if not self.context:
            return 1.0
        durations = [n.duration for n in self.context[-8:]]
        if len(durations) < 2:
            return durations[0] if durations else 1.0
        x, p = durations[0], 1.0
        for d in durations[1:]:
            x_pred = x
            p_pred = p + 0.05
            k = p_pred / (p_pred + 0.1)
            x = x_pred + k * (d - x_pred)
            p = (1 - k) * p_pred
        return round(max(0.25, min(4.0, x)), 2)

    def _emotional_tag(self, detail: Dict[str, float]) -> str:
        if detail["tension"] > 0.8 and detail["consonance"] < 0.4:
            return "dramatic_tension"
        if detail["consonance"] > 0.8 and detail["fibonacci"] > 0.7:
            return "golden_harmony"
        if detail["golden_position"] > 0.8:
            return "structural_climax"
        if detail["beauty"] > 0.7 and detail["markov"] > 0.3:
            return "natural_flow"
        if detail["tension"] < 0.3 and detail["consonance"] > 0.7:
            return "peaceful_resolution"
        if detail["fibonacci"] > 0.8:
            return "fibonacci_resonance"
        return "neutral"

    def generate_emotional_sequence(
        self,
        length: int = 32,
        start_notes: Optional[List[NoteEvent]] = None,
        target_emotion: str = "balanced",
    ) -> List[NoteEvent]:
        if start_notes:
            self.feed_context(start_notes, piece_total_length=length * 2.0)

        sequence: List[NoteEvent] = list(self.context)
        self._piece_length = length * 2.0

        for _ in range(length):
            suggestions = self.predict_next(top_k=5)
            if not suggestions:
                break

            chosen = self._select_by_emotion(suggestions, target_emotion)

            new_note = NoteEvent(
                pitch=chosen.pitch,
                duration=chosen.predicted_duration,
                start=self._current_position,
            )
            sequence.append(new_note)
            self.context.append(new_note)
            if len(self.context) > self.config.context_window:
                self.context = self.context[-self.config.context_window:]
            self.kalman.update(float(chosen.pitch))
            self._current_position += chosen.predicted_duration

        return sequence

    def _select_by_emotion(
        self, suggestions: List[NoteSuggestion], target: str
    ) -> NoteSuggestion:
        if target == "dramatic":
            key_fn = lambda s: s.scores_detail["tension"]
        elif target == "peaceful":
            key_fn = lambda s: s.scores_detail["consonance"]
        elif target == "mathematical":
            key_fn = lambda s: (
                s.scores_detail["fibonacci"] + s.scores_detail["golden_position"]
            ) / 2.0
        elif target == "expressive":
            key_fn = lambda s: s.scores_detail["beauty"]
        else:
            key_fn = lambda s: s.score

        probs = np.array([key_fn(s) for s in suggestions], dtype=float)
        probs = np.maximum(probs, 1e-8)
        probs = probs / probs.sum()
        idx = int(np.random.choice(len(suggestions), p=probs))
        return suggestions[idx]

    def analyze_sequence_impact(self, notes: List[NoteEvent]) -> Dict:
        if len(notes) < 2:
            return {"impact_score": 0.0, "segments": []}

        pitches = [n.pitch for n in notes]
        intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]

        fib_count = sum(1 for iv in intervals if iv in FIBONACCI_INTERVALS or iv % 12 in FIBONACCI_INTERVALS)
        fib_ratio = fib_count / len(intervals)

        cons_scores = [CONSONANCE_MAP.get(iv % 12, 0.5) for iv in intervals]
        mean_consonance = sum(cons_scores) / len(cons_scores)

        total_dur = notes[-1].start + notes[-1].duration - notes[0].start
        if total_dur > 0:
            climax_pos = 0
            max_pitch = 0
            for n in notes:
                if n.pitch > max_pitch:
                    max_pitch = n.pitch
                    climax_pos = n.start
            climax_ratio = (climax_pos - notes[0].start) / total_dur
            golden_dist = abs(climax_ratio - 1.0 / PHI)
        else:
            golden_dist = 1.0

        tension_curve = []
        window = 8
        for i in range(0, len(intervals), window):
            chunk = intervals[i:i + window]
            if chunk:
                tension_curve.append(sum(chunk) / len(chunk) / 12.0)

        impact = (
            0.25 * fib_ratio
            + 0.25 * mean_consonance
            + 0.25 * max(0.0, 1.0 - golden_dist * 3)
            + 0.25 * (1.0 - abs(mean_consonance - 0.65) * 2)
        )

        segments = []
        seg_size = max(1, len(notes) // 4)
        for i in range(0, len(notes), seg_size):
            seg = notes[i:i + seg_size]
            if len(seg) < 2:
                continue
            seg_intervals = [abs(seg[j + 1].pitch - seg[j].pitch) for j in range(len(seg) - 1)]
            seg_cons = [CONSONANCE_MAP.get(iv % 12, 0.5) for iv in seg_intervals]
            seg_fib = sum(1 for iv in seg_intervals if iv in FIBONACCI_INTERVALS) / max(len(seg_intervals), 1)
            pos = (seg[0].start - notes[0].start) / max(total_dur, 1.0)
            segments.append({
                "position": round(pos, 3),
                "tension": round(sum(seg_intervals) / len(seg_intervals) / 12.0, 3),
                "consonance": round(sum(seg_cons) / len(seg_cons), 3),
                "fibonacci_ratio": round(seg_fib, 3),
                "note_count": len(seg),
            })

        return {
            "impact_score": round(max(0.0, min(1.0, impact)), 3),
            "fibonacci_ratio": round(fib_ratio, 3),
            "mean_consonance": round(mean_consonance, 3),
            "golden_climax_distance": round(golden_dist, 3),
            "tension_curve": [round(t, 3) for t in tension_curve],
            "segments": segments,
            "emotional_profile": self._classify_emotion_profile(
                fib_ratio, mean_consonance, golden_dist, tension_curve
            ),
        }

    @staticmethod
    def _classify_emotion_profile(
        fib_ratio: float, consonance: float, golden_dist: float, tension_curve: List[float]
    ) -> str:
        if fib_ratio > 0.4 and golden_dist < 0.1:
            return "mathematically_sublime"
        if consonance > 0.75 and (not tension_curve or max(tension_curve) < 0.4):
            return "serene_contemplative"
        if tension_curve and max(tension_curve) > 0.7:
            return "dramatic_passionate"
        if consonance < 0.4:
            return "dark_dissonant"
        if fib_ratio > 0.3:
            return "naturally_flowing"
        return "balanced_classical"


__all__ = [
    "PredictorConfig",
    "NoteSuggestion",
    "NotePredictor",
]
