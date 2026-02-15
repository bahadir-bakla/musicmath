"""
Kaos Teorisi tabanli muzik analizi.

Melodilerdeki kaotik / deterministik yapilari ortaya cikarir:
- Lyapunov ustu (trajectory divergence hizi)
- Fraktal boyut (box-counting & correlation dimension)
- Phase-space reconstruction (Takens embedding)
- Recurrence Quantification Analysis (RQA)
- Hurst ustu (long-range dependence)

Temel fikir: Muzik ne tamamen rastgele ne de tamamen deterministiktir.
"Edge of chaos" bolgesi -- en ilginc eserler burada bulunur.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

from music_math.core.types import NoteEvent


def _pitch_series(events: Sequence[NoteEvent]) -> np.ndarray:
    return np.array([e.pitch for e in events], dtype=float)


def _interval_series(events: Sequence[NoteEvent]) -> np.ndarray:
    pitches = _pitch_series(events)
    if len(pitches) < 2:
        return np.array([], dtype=float)
    return np.diff(pitches)


# ---------------------------------------------------------------------------
# Phase-space reconstruction (Takens embedding)
# ---------------------------------------------------------------------------

def takens_embedding(
    series: np.ndarray, dim: int = 3, tau: int = 1
) -> np.ndarray:
    """
    Zaman serisinden phase-space vektorleri olusturur.
    dim: embedding boyutu, tau: gecikme (delay).
    Doner: (N - (dim-1)*tau, dim) boyutunda matris.
    """
    n = len(series)
    rows = n - (dim - 1) * tau
    if rows <= 0:
        return np.empty((0, dim))
    out = np.empty((rows, dim))
    for d in range(dim):
        out[:, d] = series[d * tau: d * tau + rows]
    return out


def optimal_delay(series: np.ndarray, max_lag: int = 20) -> int:
    """
    Auto-mutual-information ile optimal delay (tau) tahmini.
    Ilk minimumu bulur; bulamazsa max_lag/4 doner.
    """
    n = len(series)
    if n < max_lag * 2:
        return 1

    bins = max(10, int(np.sqrt(n)))
    hist_x, edges = np.histogram(series, bins=bins, density=True)
    hist_x = hist_x + 1e-12

    ami_vals = []
    for lag in range(1, max_lag + 1):
        x = series[:-lag]
        y = series[lag:]
        h2d, _, _ = np.histogram2d(x, y, bins=bins, density=True)
        h2d = h2d + 1e-12
        hx = np.histogram(x, bins=bins, density=True)[0] + 1e-12
        hy = np.histogram(y, bins=bins, density=True)[0] + 1e-12
        mi = np.sum(h2d * np.log(h2d / np.outer(hx, hy)))
        ami_vals.append(mi)

    for i in range(1, len(ami_vals) - 1):
        if ami_vals[i] < ami_vals[i - 1] and ami_vals[i] < ami_vals[i + 1]:
            return i + 1
    return max(1, max_lag // 4)


# ---------------------------------------------------------------------------
# Lyapunov ustu (Maximum Lyapunov Exponent -- Rosenstein methodu)
# ---------------------------------------------------------------------------

def lyapunov_exponent(
    series: np.ndarray,
    dim: int = 3,
    tau: int = 1,
    min_separation: int = 5,
    max_iter: int = 20,
) -> float:
    """
    Rosenstein algoritmasi ile maximum Lyapunov exponent tahmini.
    Pozitif = kaotik, ~0 = periyodik, negatif = sabit noktaya cekiliyor.
    """
    embedded = takens_embedding(series, dim=dim, tau=tau)
    n_pts = len(embedded)
    if n_pts < max_iter + min_separation:
        return 0.0

    dists = squareform(pdist(embedded))
    np.fill_diagonal(dists, np.inf)

    divergence = np.zeros(max_iter)
    count = np.zeros(max_iter)

    for i in range(n_pts - max_iter):
        row = dists[i].copy()
        for k in range(max(0, i - min_separation), min(n_pts, i + min_separation + 1)):
            row[k] = np.inf
        j = int(np.argmin(row))
        if row[j] == np.inf:
            continue

        for step in range(max_iter):
            i2, j2 = i + step, j + step
            if i2 >= n_pts or j2 >= n_pts:
                break
            d = np.linalg.norm(embedded[i2] - embedded[j2])
            if d > 0:
                divergence[step] += math.log(d)
                count[step] += 1

    valid = count > 0
    if not np.any(valid):
        return 0.0

    avg_div = np.zeros_like(divergence)
    avg_div[valid] = divergence[valid] / count[valid]

    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 3:
        return 0.0

    x = valid_idx.astype(float)
    y = avg_div[valid_idx]
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])


# ---------------------------------------------------------------------------
# Fraktal Boyut (Correlation Dimension -- Grassberger-Procaccia)
# ---------------------------------------------------------------------------

def correlation_dimension(
    series: np.ndarray,
    dim: int = 3,
    tau: int = 1,
    n_radii: int = 15,
) -> float:
    """
    Grassberger-Procaccia korelasyon boyutu.
    Melodik karmasikligin fraktal olcusu.
    """
    embedded = takens_embedding(series, dim=dim, tau=tau)
    n_pts = len(embedded)
    if n_pts < 30:
        return 0.0

    if n_pts > 500:
        idx = np.random.default_rng(42).choice(n_pts, 500, replace=False)
        embedded = embedded[idx]
        n_pts = 500

    dists = pdist(embedded)
    dists = dists[dists > 0]
    if len(dists) == 0:
        return 0.0

    r_min = np.percentile(dists, 5)
    r_max = np.percentile(dists, 80)
    if r_min <= 0 or r_max <= r_min:
        return 0.0

    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
    log_r = np.log(radii)
    log_c = np.zeros(n_radii)

    n_pairs = len(dists)
    for i, r in enumerate(radii):
        c = np.sum(dists < r) / n_pairs
        if c > 0:
            log_c[i] = np.log(c)
        else:
            log_c[i] = -np.inf

    valid = np.isfinite(log_c)
    if np.sum(valid) < 4:
        return 0.0

    coeffs = np.polyfit(log_r[valid], log_c[valid], 1)
    return float(max(0.0, coeffs[0]))


# ---------------------------------------------------------------------------
# Hurst ustu (Rescaled Range analizi)
# ---------------------------------------------------------------------------

def hurst_exponent(series: np.ndarray) -> float:
    """
    R/S analizi ile Hurst ustu.
    H > 0.5: long-range positive correlation (trending/persistent)
    H = 0.5: random walk
    H < 0.5: anti-persistent (mean-reverting)

    Muziksel yorum: H > 0.5 = melodic coherence, H < 0.5 = atonal/surprisal
    """
    n = len(series)
    if n < 20:
        return 0.5

    max_k = int(np.log2(n))
    if max_k < 2:
        return 0.5

    rs_list = []
    ns_list = []

    for k in range(2, max_k + 1):
        size = n // (2 ** k)
        if size < 4:
            break
        n_chunks = n // size
        rs_vals = []
        for i in range(n_chunks):
            chunk = series[i * size: (i + 1) * size]
            mean_c = np.mean(chunk)
            dev = chunk - mean_c
            cum_dev = np.cumsum(dev)
            r = np.max(cum_dev) - np.min(cum_dev)
            s = np.std(chunk, ddof=1) if np.std(chunk, ddof=1) > 0 else 1e-12
            rs_vals.append(r / s)
        if rs_vals:
            rs_list.append(np.mean(rs_vals))
            ns_list.append(size)

    if len(rs_list) < 3:
        return 0.5

    log_n = np.log(ns_list)
    log_rs = np.log(np.array(rs_list) + 1e-12)
    coeffs = np.polyfit(log_n, log_rs, 1)
    return float(np.clip(coeffs[0], 0.0, 1.0))


# ---------------------------------------------------------------------------
# Recurrence Quantification Analysis (RQA)
# ---------------------------------------------------------------------------

def recurrence_matrix(
    series: np.ndarray, dim: int = 3, tau: int = 1, threshold: float = 0.0
) -> np.ndarray:
    """
    Phase-space'te recurrence matrisi olusturur.
    threshold=0 ise otomatik olarak %10 percentile kullanir.
    """
    embedded = takens_embedding(series, dim=dim, tau=tau)
    n = len(embedded)
    if n < 5:
        return np.zeros((0, 0), dtype=bool)

    if n > 800:
        # Random sampling destroys temporal structure needed for RQA (diagonal lines).
        # We use a sequential slice instead.
        embedded = embedded[:800]
        n = 800

    dist_mat = squareform(pdist(embedded))
    if threshold <= 0:
        threshold = np.percentile(dist_mat[dist_mat > 0], 10) if np.any(dist_mat > 0) else 1.0
    return dist_mat <= threshold


def recurrence_rate(rec_mat: np.ndarray) -> float:
    if rec_mat.size == 0:
        return 0.0
    n = rec_mat.shape[0]
    return float(np.sum(rec_mat) - n) / (n * (n - 1)) if n > 1 else 0.0


def determinism(rec_mat: np.ndarray, min_line: int = 2) -> float:
    """
    DET: Recurrence noktalarinin diagonal cizgiler uzerindeki orani.
    Yuksek DET = deterministik yapi, dusuk DET = stokastik.
    """
    n = rec_mat.shape[0]
    if n < min_line + 1:
        return 0.0

    total_rec = 0
    line_rec = 0

    for offset in range(1, n):
        diag = np.array([rec_mat[i, i + offset] for i in range(n - offset)])
        total_rec += np.sum(diag)
        run_len = 0
        for val in diag:
            if val:
                run_len += 1
            else:
                if run_len >= min_line:
                    line_rec += run_len
                run_len = 0
        if run_len >= min_line:
            line_rec += run_len

    return float(line_rec) / max(total_rec, 1)


def laminarity(rec_mat: np.ndarray, min_line: int = 2) -> float:
    """
    LAM: Dikey cizgilerdeki recurrence orani.
    Yuksek LAM = muziksel "duraksamalar" veya tekrar eden durumlar.
    """
    n = rec_mat.shape[0]
    if n < min_line + 1:
        return 0.0

    total_rec = 0
    vert_rec = 0

    for col in range(n):
        column = rec_mat[:, col]
        total_rec += np.sum(column)
        run_len = 0
        for val in column:
            if val:
                run_len += 1
            else:
                if run_len >= min_line:
                    vert_rec += run_len
                run_len = 0
        if run_len >= min_line:
            vert_rec += run_len

    return float(vert_rec) / max(total_rec, 1)


# ---------------------------------------------------------------------------
# Logistic map benzerlik testi
# ---------------------------------------------------------------------------

def logistic_map_similarity(series: np.ndarray, r_range: Tuple[float, float] = (3.5, 4.0), n_r: int = 20) -> Dict[str, float]:
    """
    Melodik seriyi logistic map x_{n+1} = r * x_n * (1 - x_n) ile karsilastirir.
    Muziksel kaosun "logistic map" tipi kaos ile uyumunu olcer.
    """
    if len(series) < 10:
        return {"best_r": 0.0, "correlation": 0.0}

    norm = (series - series.min()) / (series.max() - series.min() + 1e-12)
    n = len(norm)

    best_r = 0.0
    best_corr = -1.0

    for r in np.linspace(r_range[0], r_range[1], n_r):
        x = np.zeros(n)
        x[0] = norm[0] if 0 < norm[0] < 1 else 0.5
        for i in range(1, n):
            x[i] = r * x[i - 1] * (1 - x[i - 1])
        corr = float(np.corrcoef(norm, x)[0, 1])
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_r = r

    return {"best_r": round(best_r, 4), "correlation": round(best_corr, 4)}


# ---------------------------------------------------------------------------
# Entropy of recurrence (Shannon entropy of diagonal line lengths)
# ---------------------------------------------------------------------------

def recurrence_entropy(rec_mat: np.ndarray, min_line: int = 2) -> float:
    """
    Diagonal cizgi uzunluklarinin Shannon entropisi.
    Yuksek = karmasik recurrence yapisi, dusuk = basit/periyodik.
    """
    n = rec_mat.shape[0]
    if n < min_line + 1:
        return 0.0

    line_lengths = []
    for offset in range(1, n):
        diag = np.array([rec_mat[i, i + offset] for i in range(n - offset)])
        run_len = 0
        for val in diag:
            if val:
                run_len += 1
            else:
                if run_len >= min_line:
                    line_lengths.append(run_len)
                run_len = 0
        if run_len >= min_line:
            line_lengths.append(run_len)

    if not line_lengths:
        return 0.0

    counts = {}
    for l in line_lengths:
        counts[l] = counts.get(l, 0) + 1
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


# ---------------------------------------------------------------------------
# Ana analiz fonksiyonu
# ---------------------------------------------------------------------------

def analyze_chaos(
    events: Sequence[NoteEvent],
    embedding_dim: int = 3,
) -> Dict[str, float]:
    """
    Tek bir eser icin kaos teorisi metriklerini hesaplar.
    CSV/feature birlestirmeye uygun dict doner.
    """
    if len(events) < 20:
        return {
            "lyapunov_exponent": 0.0,
            "correlation_dimension": 0.0,
            "hurst_exponent": 0.5,
            "recurrence_rate": 0.0,
            "determinism": 0.0,
            "laminarity": 0.0,
            "recurrence_entropy": 0.0,
            "logistic_r": 0.0,
            "logistic_correlation": 0.0,
            "chaos_score": 0.0,
            "edge_of_chaos_score": 0.0,
        }

    pitches = _pitch_series(events)
    intervals = _interval_series(events)

    tau = optimal_delay(pitches) if len(pitches) > 40 else 1
    dim = embedding_dim

    le = lyapunov_exponent(pitches, dim=dim, tau=tau)
    cd = correlation_dimension(pitches, dim=dim, tau=tau)
    he = hurst_exponent(intervals)

    rec_mat = recurrence_matrix(pitches, dim=dim, tau=tau)
    rr = recurrence_rate(rec_mat)
    det = determinism(rec_mat)
    lam = laminarity(rec_mat)
    rent = recurrence_entropy(rec_mat)

    logistic = logistic_map_similarity(pitches)

    chaos_score = float(np.clip(
        0.3 * min(le, 1.0) +
        0.2 * min(cd / 3.0, 1.0) +
        0.2 * (1.0 - det) +
        0.15 * (1.0 - abs(he - 0.5) * 2) +
        0.15 * rent / max(rent, 3.0),
        0.0, 1.0
    ))

    h_dist = abs(he - 0.5)
    le_edge = 1.0 - abs(le - 0.1) * 5 if 0 < le < 0.2 else 0.0
    det_edge = 1.0 - abs(det - 0.6) * 3 if 0.3 < det < 0.9 else 0.0
    edge_of_chaos_score = float(np.clip(
        0.3 * max(0, 1.0 - h_dist * 4) +
        0.3 * max(0, le_edge) +
        0.2 * max(0, det_edge) +
        0.2 * min(rent / 3.0, 1.0),
        0.0, 1.0
    ))

    return {
        "lyapunov_exponent": round(le, 4),
        "correlation_dimension": round(cd, 4),
        "hurst_exponent": round(he, 4),
        "recurrence_rate": round(rr, 4),
        "determinism": round(det, 4),
        "laminarity": round(lam, 4),
        "recurrence_entropy": round(rent, 4),
        "logistic_r": logistic["best_r"],
        "logistic_correlation": logistic["correlation"],
        "chaos_score": round(chaos_score, 4),
        "edge_of_chaos_score": round(edge_of_chaos_score, 4),
    }


__all__ = [
    "takens_embedding",
    "optimal_delay",
    "lyapunov_exponent",
    "correlation_dimension",
    "hurst_exponent",
    "recurrence_matrix",
    "recurrence_rate",
    "determinism",
    "laminarity",
    "recurrence_entropy",
    "logistic_map_similarity",
    "analyze_chaos",
]
