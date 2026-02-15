"""
Oyun Teorisi tabanli muzik analizi.

Muzigi "strateji oyunu" olarak modeller:
- Her nota/ses bir "oyuncu" gibi davranir
- Konsonans/dissonans bir payoff matrisi olusturur
- Nash equilibrium'a yakinlik olcer
- Bestecinin "stratejik" tercihlerini analiz eder

Teorik cerceve:
1. Voice Interaction Game: Soprano-bass veya melody-harmony etkilesimi
2. Temporal Strategy Game: Ardisik nota secimlerinde "iterated game" 
3. Consonance Payoff Matrix: Aralik-bazli kazanc/kayip tablosu
4. Nash Equilibrium Distance: Melodik dizinin NE'ye yakinligi
5. Minimax Analizi: En kotu durumda bile kaliteyi koruyan stratejiler
6. Prisoner's Dilemma analojisi: Tension vs Resolution oyunu
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from music_math.core.types import NoteEvent


# ---------------------------------------------------------------------------
# Konsonans Payoff Matrisi (12x12 pitch-class interaction)
# ---------------------------------------------------------------------------

CONSONANCE_PAYOFF = np.array([
    # C   C#  D   Eb  E   F   F#  G   Ab  A   Bb  B
    [1.0, -0.8, -0.3, 0.2, 0.6, 0.7, -0.9, 0.9, 0.3, 0.5, -0.2, -0.6],  # C
    [-0.8, 1.0, -0.8, -0.3, 0.2, 0.6, 0.7, -0.9, 0.9, 0.3, 0.5, -0.2],  # C#
    [-0.3, -0.8, 1.0, -0.8, -0.3, 0.2, 0.6, 0.7, -0.9, 0.9, 0.3, 0.5],  # D
    [0.2, -0.3, -0.8, 1.0, -0.8, -0.3, 0.2, 0.6, 0.7, -0.9, 0.9, 0.3],  # Eb
    [0.6, 0.2, -0.3, -0.8, 1.0, -0.8, -0.3, 0.2, 0.6, 0.7, -0.9, 0.9],  # E
    [0.7, 0.6, 0.2, -0.3, -0.8, 1.0, -0.8, -0.3, 0.2, 0.6, 0.7, -0.9],  # F
    [-0.9, 0.7, 0.6, 0.2, -0.3, -0.8, 1.0, -0.8, -0.3, 0.2, 0.6, 0.7],  # F#
    [0.9, -0.9, 0.7, 0.6, 0.2, -0.3, -0.8, 1.0, -0.8, -0.3, 0.2, 0.6],  # G
    [0.3, 0.9, -0.9, 0.7, 0.6, 0.2, -0.3, -0.8, 1.0, -0.8, -0.3, 0.2],  # Ab
    [0.5, 0.3, 0.9, -0.9, 0.7, 0.6, 0.2, -0.3, -0.8, 1.0, -0.8, -0.3],  # A
    [-0.2, 0.5, 0.3, 0.9, -0.9, 0.7, 0.6, 0.2, -0.3, -0.8, 1.0, -0.8],  # Bb
    [-0.6, -0.2, 0.5, 0.3, 0.9, -0.9, 0.7, 0.6, 0.2, -0.3, -0.8, 1.0],  # B
], dtype=float)


def _pitch_classes(events: Sequence[NoteEvent]) -> np.ndarray:
    return np.array([e.pitch % 12 for e in events], dtype=int)


# ---------------------------------------------------------------------------
# Nash Equilibrium hesaplama (2-oyunculu sifir toplamli oyun)
# ---------------------------------------------------------------------------

def compute_nash_equilibrium(payoff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    2-oyunculu simetrik oyun icin mixed-strategy Nash equilibrium.
    Linear programming yerine iterative fictitious play kullanir (daha robust).
    """
    n = payoff.shape[0]
    counts_row = np.ones(n) / n
    counts_col = np.ones(n) / n

    for _ in range(500):
        best_row = np.argmax(payoff @ counts_col)
        best_col = np.argmax(payoff.T @ counts_row)
        lr = 2.0 / (_ + 2)
        new_row = np.zeros(n)
        new_row[best_row] = 1.0
        counts_row = (1 - lr) * counts_row + lr * new_row
        new_col = np.zeros(n)
        new_col[best_col] = 1.0
        counts_col = (1 - lr) * counts_col + lr * new_col

    return counts_row, counts_col


NASH_EQ_ROW, NASH_EQ_COL = compute_nash_equilibrium(CONSONANCE_PAYOFF)


def nash_distance(events: Sequence[NoteEvent]) -> float:
    """
    Eserin pitch-class dagiliminin Nash equilibrium'a KL-divergence'i.
    Dusuk = besteci "optimal strateji" oynuyor, yuksek = surprisal/innovation.
    """
    if len(events) < 5:
        return 0.0

    pcs = _pitch_classes(events)
    hist = np.zeros(12)
    for pc in pcs:
        hist[pc] += 1
    hist = hist / (hist.sum() + 1e-12)
    hist = np.maximum(hist, 1e-12)

    ne = np.maximum(NASH_EQ_ROW, 1e-12)
    kl = float(np.sum(hist * np.log(hist / ne)))
    return round(max(0.0, kl), 4)


# ---------------------------------------------------------------------------
# Iterated Game analizi (ardisik nota cifti etkilesimi)
# ---------------------------------------------------------------------------

def iterated_game_payoffs(events: Sequence[NoteEvent]) -> Dict[str, float]:
    """
    Ardisik nota ciftlerini "repeated game" hamleleri olarak degerlendirir.
    Cooperation = konsonans, Defection = dissonans.
    """
    if len(events) < 2:
        return {
            "mean_payoff": 0.0,
            "cooperation_rate": 0.0,
            "defection_rate": 0.0,
            "tit_for_tat_score": 0.0,
            "payoff_variance": 0.0,
        }

    pcs = _pitch_classes(events)
    payoffs = []
    for i in range(len(pcs) - 1):
        payoffs.append(CONSONANCE_PAYOFF[pcs[i], pcs[i + 1]])

    payoffs = np.array(payoffs)
    coop = float(np.mean(payoffs > 0))
    defect = float(np.mean(payoffs < 0))

    tft_score = 0.0
    if len(payoffs) > 2:
        matches = 0
        for i in range(1, len(payoffs)):
            if (payoffs[i] > 0) == (payoffs[i - 1] > 0):
                matches += 1
        tft_score = matches / (len(payoffs) - 1)

    return {
        "mean_payoff": round(float(np.mean(payoffs)), 4),
        "cooperation_rate": round(coop, 4),
        "defection_rate": round(defect, 4),
        "tit_for_tat_score": round(tft_score, 4),
        "payoff_variance": round(float(np.var(payoffs)), 4),
    }


# ---------------------------------------------------------------------------
# Minimax analizi
# ---------------------------------------------------------------------------

def minimax_score(events: Sequence[NoteEvent], window: int = 8) -> float:
    """
    Sliding-window uzerinde minimax stratejisi:
    Her pencerede "en kotu durumda bile ne kadar iyi?"
    Yuksek = besteci risk-averse, dusuk = risk-seeking.
    """
    if len(events) < window:
        return 0.0

    pcs = _pitch_classes(events)
    scores = []
    for i in range(len(pcs) - window + 1):
        w = pcs[i:i + window]
        window_payoffs = []
        for j in range(len(w) - 1):
            window_payoffs.append(CONSONANCE_PAYOFF[w[j], w[j + 1]])
        if window_payoffs:
            scores.append(min(window_payoffs))

    if not scores:
        return 0.0
    raw = float(np.mean(scores))
    return round((raw + 1.0) / 2.0, 4)


# ---------------------------------------------------------------------------
# Prisoner's Dilemma: Tension vs Resolution
# ---------------------------------------------------------------------------

def tension_resolution_game(events: Sequence[NoteEvent]) -> Dict[str, float]:
    """
    Tension (dissonans) = Defect, Resolution (konsonans) = Cooperate
    Prisoner's Dilemma benzeri analiz:
    - Mutual cooperation (CC): Iki ardisik konsonans = +3
    - Temptation to defect (DC): Dissonans sonrasi konsonans = +5 (surprisal!)
    - Sucker's payoff (CD): Konsonans sonrasi dissonans = 0
    - Mutual defection (DD): Iki ardisik dissonans = +1
    """
    if len(events) < 3:
        return {
            "cc_rate": 0.0, "cd_rate": 0.0,
            "dc_rate": 0.0, "dd_rate": 0.0,
            "pd_total_payoff": 0.0,
            "surprise_resolution_rate": 0.0,
        }

    PD_PAYOFF = {"CC": 3.0, "CD": 0.0, "DC": 5.0, "DD": 1.0}

    pcs = _pitch_classes(events)
    states = []
    for i in range(len(pcs) - 1):
        p = CONSONANCE_PAYOFF[pcs[i], pcs[i + 1]]
        states.append("C" if p > 0 else "D")

    cc = cd = dc = dd = 0
    payoff_sum = 0.0
    for i in range(len(states) - 1):
        key = states[i] + states[i + 1]
        payoff_sum += PD_PAYOFF[key]
        if key == "CC": cc += 1
        elif key == "CD": cd += 1
        elif key == "DC": dc += 1
        else: dd += 1

    total = max(cc + cd + dc + dd, 1)
    max_possible = total * 5.0

    return {
        "cc_rate": round(cc / total, 4),
        "cd_rate": round(cd / total, 4),
        "dc_rate": round(dc / total, 4),
        "dd_rate": round(dd / total, 4),
        "pd_total_payoff": round(payoff_sum / max_possible, 4) if max_possible > 0 else 0.0,
        "surprise_resolution_rate": round(dc / total, 4),
    }


# ---------------------------------------------------------------------------
# Evolutionary Stable Strategy (ESS) testi
# ---------------------------------------------------------------------------

def evolutionary_stability(events: Sequence[NoteEvent], window: int = 32) -> Dict[str, float]:
    """
    Eserin farkli bolumleri arasinda strateji tutarliligi olcer.
    ESS: Baskasi (mutant strateji) tarafindan istila edilemeyen strateji.
    Muziksel yorum: Bestecinin tonal tutarliligi / stilistik kararliligi.
    """
    if len(events) < window * 2:
        return {"strategy_stability": 0.0, "invasion_resistance": 0.0}

    pcs = _pitch_classes(events)
    n_windows = len(pcs) // window

    distributions = []
    for i in range(n_windows):
        chunk = pcs[i * window: (i + 1) * window]
        hist = np.zeros(12)
        for pc in chunk:
            hist[pc] += 1
        hist = hist / (hist.sum() + 1e-12)
        distributions.append(hist)

    if len(distributions) < 2:
        return {"strategy_stability": 0.0, "invasion_resistance": 0.0}

    mean_dist = np.mean(distributions, axis=0)
    deviations = [np.sum(np.abs(d - mean_dist)) / 2.0 for d in distributions]
    stability = 1.0 - float(np.mean(deviations))

    invasion_scores = []
    for i, d in enumerate(distributions):
        payoff_own = float(d @ CONSONANCE_PAYOFF @ mean_dist)
        random_strat = np.ones(12) / 12.0
        payoff_mutant = float(random_strat @ CONSONANCE_PAYOFF @ mean_dist)
        invasion_scores.append(1.0 if payoff_own >= payoff_mutant else 0.0)

    return {
        "strategy_stability": round(max(0, stability), 4),
        "invasion_resistance": round(float(np.mean(invasion_scores)), 4),
    }


# ---------------------------------------------------------------------------
# Shapley Value -- her pitch class'in "contribution"u
# ---------------------------------------------------------------------------

def shapley_pitch_values(events: Sequence[NoteEvent], top_k: int = 5) -> Dict[str, float]:
    """
    Her pitch class'in ortalama marjinal katkisini olcer (Shapley benzeri).
    Tam Shapley 2^12 subset gerektirir; burada Monte Carlo approx. kullaniyoruz.
    """
    if len(events) < 10:
        return {"shapley_top_pc": -1, "shapley_gini": 0.0}

    pcs = _pitch_classes(events)
    unique_pcs = np.unique(pcs)
    if len(unique_pcs) < 2:
        return {"shapley_top_pc": int(unique_pcs[0]) if len(unique_pcs) > 0 else -1, "shapley_gini": 0.0}

    rng = np.random.default_rng(42)
    n_samples = 200
    contributions = np.zeros(12)
    counts = np.zeros(12)

    payoffs_full = []
    for i in range(len(pcs) - 1):
        payoffs_full.append(CONSONANCE_PAYOFF[pcs[i], pcs[i + 1]])
    full_value = float(np.mean(payoffs_full)) if payoffs_full else 0.0

    for _ in range(n_samples):
        perm = rng.permutation(unique_pcs)
        coalition = set()
        prev_value = 0.0
        for pc in perm:
            coalition.add(pc)
            mask = np.array([p in coalition for p in pcs])
            idx = np.where(mask)[0]
            if len(idx) < 2:
                marginal = 0.0
            else:
                filtered = pcs[idx]
                payoffs = [CONSONANCE_PAYOFF[filtered[j], filtered[j + 1]]
                           for j in range(len(filtered) - 1)]
                val = float(np.mean(payoffs)) if payoffs else 0.0
                marginal = val - prev_value
                prev_value = val
            contributions[pc] += marginal
            counts[pc] += 1

    valid = counts > 0
    shapley = np.zeros(12)
    shapley[valid] = contributions[valid] / counts[valid]

    top_pc = int(np.argmax(shapley))
    vals = np.sort(np.abs(shapley[shapley != 0]))
    if len(vals) > 1:
        n_v = len(vals)
        gini = float(np.sum((2 * np.arange(1, n_v + 1) - n_v - 1) * vals) / (n_v * np.sum(vals) + 1e-12))
    else:
        gini = 0.0

    return {
        "shapley_top_pc": top_pc,
        "shapley_gini": round(abs(gini), 4),
    }


# ---------------------------------------------------------------------------
# Ana analiz fonksiyonu
# ---------------------------------------------------------------------------

def analyze_game_theory(events: Sequence[NoteEvent]) -> Dict[str, float]:
    """
    Tek bir eser icin oyun teorisi metriklerini hesaplar.
    """
    if len(events) < 5:
        return {k: 0.0 for k in [
            "nash_distance", "mean_payoff", "cooperation_rate", "defection_rate",
            "tit_for_tat_score", "payoff_variance", "minimax_score",
            "cc_rate", "cd_rate", "dc_rate", "dd_rate",
            "pd_total_payoff", "surprise_resolution_rate",
            "strategy_stability", "invasion_resistance",
            "shapley_top_pc", "shapley_gini",
            "game_theory_composite",
        ]}

    out: Dict[str, float] = {}

    out["nash_distance"] = nash_distance(events)

    iterated = iterated_game_payoffs(events)
    out.update(iterated)

    out["minimax_score"] = minimax_score(events)

    pd = tension_resolution_game(events)
    out.update(pd)

    evo = evolutionary_stability(events)
    out.update(evo)

    shapley = shapley_pitch_values(events)
    out["shapley_top_pc"] = float(shapley["shapley_top_pc"])
    out["shapley_gini"] = shapley["shapley_gini"]

    composite = (
        0.2 * (1.0 - min(out["nash_distance"], 2.0) / 2.0) +
        0.2 * (out["cooperation_rate"]) +
        0.15 * out["minimax_score"] +
        0.15 * out["pd_total_payoff"] +
        0.15 * out["strategy_stability"] +
        0.15 * out["invasion_resistance"]
    )
    out["game_theory_composite"] = round(float(np.clip(composite, 0, 1)), 4)

    return out


__all__ = [
    "compute_nash_equilibrium",
    "nash_distance",
    "iterated_game_payoffs",
    "minimax_score",
    "tension_resolution_game",
    "evolutionary_stability",
    "shapley_pitch_values",
    "analyze_game_theory",
]
