#!/usr/bin/env python
"""
Nota dizilerinin sayisal oruntu analizi.

Fikir:
    - Her eseri zamana gore siralanmis bir pitch serisi (MIDI numaralari) olarak dusun.
    - Bu seriden tureyen cesitli sayisal oruntuleri olc:
        * pitch_class_hist: 0–11 arasinda dagilim (mod 12)
        * kademeli toplamlarda asal sayi sikligi
        * altin oran pozisyonlarinda (phi ~ 0.618) nota yogunlugu
        * kendi kendine benzerlik (interval serisi icin otokorelasyon tabanli bir fraktal benzerlik skoru)

Kullanim:
    python scripts/note_numeric_patterns.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger
from music_math.data.loader import parse_midi_to_note_events


logger = get_logger(__name__)


PHI = (1.0 + 5 ** 0.5) / 2.0


@dataclass
class NumericPatternStats:
    n_notes: int
    pitch_mean: float
    pitch_std: float
    prime_sum_ratio: float
    phi_density_ratio: float
    interval_self_similarity: float
    rw_corr_mean: float
    rw_corr_z: float


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    k = 3
    # Dogru primality testi icin kosul k^2 <= n olmali
    while k * k <= n:
        if n % k == 0:
            return False
        k += 2
    return True


def _running_sum_prime_ratio(pitches: np.ndarray) -> float:
    """Kümülatif pitch toplamlari icinde asala denk gelenlerin orani."""
    if len(pitches) == 0:
        return 0.0
    cumsum = np.cumsum(pitches.astype(int))
    prime_flags = np.array([_is_prime(int(v)) for v in cumsum], dtype=float)
    return float(prime_flags.mean())


def _phi_density_ratio(pitches: np.ndarray) -> float:
    """
    Altin oran pozisyonlarindaki nota yogunlugu.

    Basit surum: eseri N nota olarak dusunup,
    phi*N, (1-phi)*N, phi^2*N vb. etrafinda kucuk bir
    pencerede (ornegin +-2 nota) nota varligini olc.
    """
    n = len(pitches)
    if n == 0:
        return 0.0

    # Golden ratio noktalarini zaman ekseni uzerinde oran olarak dusun:
    # temel nokta ~ N / PHI (~0.618 N), ayrica daha kucuk/ buyuk
    # olcekler icin 1/PHI^2 ve (1 - 1/PHI), (1 - 1/PHI^2) civari.
    frac_points = {1.0 / PHI, 1.0 / (PHI ** 2), 1.0 - 1.0 / PHI, 1.0 - 1.0 / (PHI ** 2)}
    indices = []
    for f in frac_points:
        if 0.0 <= f <= 1.0:
            idx = int(round(f * (n - 1)))
            indices.append(idx)

    if not indices:
        return 0.0

    # Pencere boyutu: eserin uzunlugunun yuzdesi (ornegin %3),
    # en az 1, en fazla 32 nota.
    window = max(1, int(0.03 * n))
    window = min(window, 32)

    mask_phi = np.zeros(n, dtype=bool)
    for pos in indices:
        lo = max(0, pos - window)
        hi = min(n, pos + window + 1)
        mask_phi[lo:hi] = True

    # Golden ratio bolgelerindeki notalarin orani
    return float(mask_phi.mean())


def _interval_self_similarity(pitches: np.ndarray, max_lag: int = 12) -> float:
    """
    Interval serisinin otokorelasyonuna dayali basit bir
    'fraktal benzerlik' skoru.
    """
    if len(pitches) < 4:
        return 0.0
    intervals = np.diff(pitches.astype(float))
    if intervals.std() == 0:
        return 0.0

    intervals = (intervals - intervals.mean()) / intervals.std()
    n = len(intervals)
    lags = range(1, min(max_lag, n // 2))
    ac_values = []
    for lag in lags:
        x1 = intervals[:-lag]
        x2 = intervals[lag:]
        ac = float(np.dot(x1, x2) / (len(x1)))
        ac_values.append(abs(ac))

    if not ac_values:
        return 0.0
    # Daha yavas sönüm (ortalama yuksek) -> daha self-similar
    return float(np.mean(ac_values))


def _interval_nonrandomness_z(
    pitches: np.ndarray, max_lag: int = 8, n_shuffles: int = 20
) -> tuple[float, float]:
    """
    Interval serisinin "rastgele yuruyus"ten sapmasini olcen z-skoru.

    Adimlar:
        - Interval'leri normalize et.
        - Kucuk lag'ler icin (1..max_lag) otokorelasyon mutlak degerlerinin
          ortalamasini al (gercek seri icin).
        - Interval'leri bircok kez karistir (shuffle) ederek ayni metriği
          hesapla ve null dagilim olustur.
        - Gozlenen degeri bu null dagilima gore z-skora cevir.
    """
    if len(pitches) < 6:
        return 0.0, 0.0

    intervals = np.diff(pitches.astype(float))
    if intervals.std() == 0:
        return 0.0, 0.0

    intervals = (intervals - intervals.mean()) / intervals.std()
    n = len(intervals)
    max_lag = min(max_lag, n // 2)
    if max_lag < 1:
        return 0.0, 0.0

    def _mean_abs_corr(ints: np.ndarray) -> float:
        vals: list[float] = []
        for lag in range(1, max_lag + 1):
            x1 = ints[:-lag]
            x2 = ints[lag:]
            if x1.size == 0:
                continue
            ac = float(np.dot(x1, x2) / len(x1))
            vals.append(abs(ac))
        return float(np.mean(vals)) if vals else 0.0

    obs_mean = _mean_abs_corr(intervals)

    if n_shuffles <= 0:
        return obs_mean, 0.0

    null_vals = []
    rng = np.random.default_rng(42)
    for _ in range(n_shuffles):
        shuf = np.array(intervals, copy=True)
        rng.shuffle(shuf)
        null_vals.append(_mean_abs_corr(shuf))

    null_arr = np.array(null_vals, dtype=float)
    mu = float(null_arr.mean())
    sigma = float(null_arr.std())
    if sigma == 0.0:
        return obs_mean, 0.0

    z = (obs_mean - mu) / sigma
    return obs_mean, float(z)


def analyze_single_midi(path: Path) -> NumericPatternStats | None:
    events = parse_midi_to_note_events(path)
    if not events:
        return None

    # events: (start, duration, pitch) varsayimi ile sadece pitch'leri al
    pitches = np.array([e.pitch for e in events])
    if pitches.size == 0:
        return None

    n_notes = int(pitches.size)
    pitch_mean = float(pitches.mean())
    pitch_std = float(pitches.std())

    prime_ratio = _running_sum_prime_ratio(pitches)
    phi_ratio = _phi_density_ratio(pitches)
    self_sim = _interval_self_similarity(pitches)
    rw_mean, rw_z = _interval_nonrandomness_z(pitches)

    return NumericPatternStats(
        n_notes=n_notes,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        prime_sum_ratio=prime_ratio,
        phi_density_ratio=phi_ratio,
        interval_self_similarity=self_sim,
        rw_corr_mean=rw_mean,
        rw_corr_z=rw_z,
    )


def build_numeric_pattern_table() -> pd.DataFrame:
    """
    metadata_clean.csv uzerinden tum eserler icin sayisal oruntu
    metriklerini hesapla.
    """
    meta_clean = CONFIG.paths.root / "metadata_clean.csv"
    if not meta_clean.exists():
        raise FileNotFoundError(f"metadata_clean.csv bulunamadi: {meta_clean}")

    df_meta = pd.read_csv(meta_clean)
    rows: List[Dict] = []

    for _, row in df_meta.iterrows():
        midi_path = CONFIG.paths.root / row["file_path"]
        stats = analyze_single_midi(midi_path)
        if stats is None:
            continue

        rows.append(
            {
                "filepath": row["file_path"],
                "composer": row.get("composer", ""),
                "era": row.get("era", ""),
                "n_notes": stats.n_notes,
                "pitch_mean": stats.pitch_mean,
                "pitch_std": stats.pitch_std,
                "prime_sum_ratio": stats.prime_sum_ratio,
                "phi_density_ratio": stats.phi_density_ratio,
                "interval_self_similarity": stats.interval_self_similarity,
                "rw_corr_mean": stats.rw_corr_mean,
                "rw_corr_z": stats.rw_corr_z,
            }
        )

    df_stats = pd.DataFrame(rows)
    return df_stats


def main() -> None:
    print("=" * 60)
    print("Nota Dizileri – Sayisal Oruntu Analizi")
    print("=" * 60)

    df_stats = build_numeric_pattern_table()

    stats_dir = CONFIG.paths.root / "results" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    out_csv = stats_dir / "note_numeric_patterns.csv"
    df_stats.to_csv(out_csv, index=False)

    print(f"\nToplam eser: {len(df_stats)}")
    print(f"Çıktı tablo: {out_csv}")

    # Basit özetler
    if "era" in df_stats.columns:
        print("\nEra bazında ortalama prime_sum_ratio:")
        print(df_stats.groupby("era")["prime_sum_ratio"].mean().sort_values())

        print("\nEra bazında ortalama phi_density_ratio:")
        print(df_stats.groupby("era")["phi_density_ratio"].mean().sort_values())

        print("\nEra bazında ortalama interval_self_similarity:")
        print(
            df_stats.groupby("era")["interval_self_similarity"]
            .mean()
            .sort_values()
        )

        print("\nEra bazında ortalama rw_corr_z (interval nonrandomness z-skoru):")
        print(
            df_stats.groupby("era")["rw_corr_z"]
            .mean()
            .sort_values()
        )


if __name__ == "__main__":
    main()

