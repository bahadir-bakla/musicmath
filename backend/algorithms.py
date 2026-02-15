"""
Zaman serisi / sinyal isleme algoritmalari: analiz ve harmoni deneyleri icin.
Kullanici istedigi oruntu/parametre ile deney yapabilsin diye hepsi parametreli ve liste halinde sunuluyor.
"""

from __future__ import annotations

from typing import List, Dict, Any, Callable, Optional
import math


def _to_float_list(series: List[float] | List[int]) -> List[float]:
    if not series:
        return []
    return [float(x) for x in series]


# ---------------------------------------------------------------------------
# 1. Kalman filter (1D)
# ---------------------------------------------------------------------------

def kalman(
    series: List[float],
    process_var: float = 0.01,
    measure_var: float = 0.1,
) -> List[float]:
    """
    Tek boyutlu Kalman filtre. Gürültülü ölçümleri düzgünleştirir;
    process_var küçük = daha yumuşak, measure_var büyük = ölçüme daha az güven.
    """
    s = _to_float_list(series)
    if not s:
        return []
    x, p = s[0], 1.0
    out = [x]
    for i in range(1, len(s)):
        x_pred, p_pred = x, p + process_var
        k = p_pred / (p_pred + measure_var)
        x = x_pred + k * (s[i] - x_pred)
        p = (1 - k) * p_pred
        out.append(x)
    return out


# ---------------------------------------------------------------------------
# 2. Exponential Moving Average (EMA)
# ---------------------------------------------------------------------------

def ema(series: List[float], alpha: float = 0.3) -> List[float]:
    """
    Üstel hareketli ortalama. alpha büyük = son değerlere daha çok ağırlık (daha az yumuşak).
    alpha in [0.05, 0.5] genelde iyi; 0.3 dengeli.
    """
    s = _to_float_list(series)
    if not s:
        return []
    out = [s[0]]
    for i in range(1, len(s)):
        out.append(alpha * s[i] + (1 - alpha) * out[-1])
    return out


# ---------------------------------------------------------------------------
# 3. Gaussian smoothing (convolution)
# ---------------------------------------------------------------------------

def gaussian_smooth(
    series: List[float],
    window_size: int = 5,
    sigma: Optional[float] = None,
) -> List[float]:
    """
    Gauss çekirdeği ile konvolüsyon. window_size tek sayı olmalı (5, 7, 9...).
    sigma None ise window_size'a göre seçilir (yaklaşık 0.25*window).
    """
    s = _to_float_list(series)
    if not s:
        return []
    n = len(s)
    w = max(3, window_size if window_size % 2 == 1 else window_size + 1)
    if sigma is None:
        sigma = 0.25 * w
    half = w // 2
    # Gaussian kernel
    kernel = []
    for i in range(-half, half + 1):
        kernel.append(math.exp(-0.5 * (i / sigma) ** 2))
    k_sum = sum(kernel)
    kernel = [k / k_sum for k in kernel]
    # Convolve (pad with edge values)
    padded = [s[0]] * half + s + [s[-1]] * half
    out = []
    for i in range(n):
        out.append(sum(padded[i + j] * kernel[j] for j in range(w)))
    return out


# ---------------------------------------------------------------------------
# 4. Savitzky-Golay (polynomial smoothing, numpy only)
# ---------------------------------------------------------------------------

def savitzky_golay(
    series: List[float],
    window_length: int = 5,
    polyorder: int = 2,
) -> List[float]:
    """
    Savitzky-Golay: pencere içinde polinom fit, merkez değeri kullan.
    Tepe/dip korunur, gürültü azalır. window_length tek, polyorder < window_length.
    """
    import numpy as np
    s = _to_float_list(series)
    if not s:
        return []
    n = len(s)
    w = max(3, window_length if window_length % 2 == 1 else window_length + 1)
    order = min(polyorder, w - 1)
    half = w // 2

    x = np.arange(-half, half + 1, dtype=float)
    A = np.column_stack([x ** p for p in range(order + 1)])
    e_center = np.zeros(w)
    e_center[half] = 1.0
    kernel = np.linalg.solve(A.T @ A, A.T @ e_center)
    padded = np.array([s[0]] * half + s + [s[-1]] * half, dtype=float)
    out = np.convolve(padded, kernel, mode="valid")
    return out.tolist()


# ---------------------------------------------------------------------------
# 5. Rolling median (robust to outliers)
# ---------------------------------------------------------------------------

def rolling_median(series: List[float], window_size: int = 5) -> List[float]:
    """
    Kayan medyan. Aykırı değerlere dayanıklı; sert geçişleri yumuşatır.
    """
    s = _to_float_list(series)
    if not s:
        return []
    n = len(s)
    w = max(1, window_size if window_size % 2 == 1 else window_size + 1)
    half = w // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = sorted(s[lo:hi])
        out.append(window[len(window) // 2])
    return out


# ---------------------------------------------------------------------------
# 6. Double exponential (Holt) – trend + yumuşaklık
# ---------------------------------------------------------------------------

def double_exponential(
    series: List[float],
    alpha: float = 0.3,
    beta: float = 0.1,
) -> List[float]:
    """
    Holt çift üstel düzeltme: seviye + trend. alpha (seviye), beta (trend) küçük = daha yumuşak.
    """
    s = _to_float_list(series)
    if not s:
        return []
    level, trend = s[0], 0.0
    out = [level]
    for i in range(1, len(s)):
        prev_level = level
        level = alpha * s[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        out.append(level + trend)
    return out


# ---------------------------------------------------------------------------
# 7. Wiener filter (scipy optional)
# ---------------------------------------------------------------------------

def wiener(series: List[float], mysize: int = 5) -> List[float]:
    """
    Wiener filtresi: yerel varyansa göre adaptif yumuşatma.
    Gürültü tahmini pencereden; sinyal korunur.
    """
    try:
        from scipy.signal import wiener as scipy_wiener
        import numpy as np
        arr = np.array(_to_float_list(series), dtype=float)
        if arr.size == 0:
            return []
        out = scipy_wiener(arr, mysize=max(3, mysize))
        return out.tolist()
    except ImportError:
        # Fallback: Gaussian with small window
        return gaussian_smooth(series, window_size=max(3, mysize), sigma=1.0)


# ---------------------------------------------------------------------------
# Registry: id -> (fn, param_schema) for API
# ---------------------------------------------------------------------------

def _param(name: str, type_: str, default: Any, min_: Optional[float] = None, max_: Optional[float] = None, description: str = "") -> Dict[str, Any]:
    p = {"name": name, "type": type_, "default": default, "description": description}
    if min_ is not None:
        p["min"] = min_
    if max_ is not None:
        p["max"] = max_
    return p


ALGORITHMS: List[Dict[str, Any]] = [
    {
        "id": "kalman",
        "name": "Kalman Filter",
        "description": "1D Kalman: ölçüm gürültüsünü azaltır, düzgün geçişler. process_var/measure_var ile hassasiyet ayarı.",
        "params": [
            _param("process_var", "float", 0.01, 0.001, 0.5, "Süreç gürültüsü (küçük = daha yumuşak)"),
            _param("measure_var", "float", 0.1, 0.01, 2.0, "Ölçüm gürültüsü (büyük = ölçüme daha az güven)"),
        ],
        "fn": kalman,
    },
    {
        "id": "ema",
        "name": "Exponential Moving Average",
        "description": "Üstel hareketli ortalama. Basit ve hızlı; alpha ile yumuşaklık.",
        "params": [
            _param("alpha", "float", 0.3, 0.05, 0.95, "Yeni değerin ağırlığı (büyük = daha az yumuşak)"),
        ],
        "fn": ema,
    },
    {
        "id": "gaussian",
        "name": "Gaussian Smoothing",
        "description": "Gauss çekirdeği ile konvolüsyon. Pencere büyüdükçe daha yumuşak.",
        "params": [
            _param("window_size", "int", 5, 3, 31, "Pencere (tek sayı, 5–31)"),
            _param("sigma", "float", 0.0, 0.0, 20.0, "0 = otomatik (pencereye göre)"),
        ],
        "fn": gaussian_smooth,
    },
    {
        "id": "savitzky_golay",
        "name": "Savitzky-Golay",
        "description": "Polinom yumuşatma; tepe/dipleri korur, gürültüyü azaltır.",
        "params": [
            _param("window_length", "int", 5, 3, 25, "Pencere (tek sayı)"),
            _param("polyorder", "int", 2, 1, 5, "Polinom derecesi"),
        ],
        "fn": savitzky_golay,
    },
    {
        "id": "rolling_median",
        "name": "Rolling Median",
        "description": "Kayan medyan; aykırı değerlere dayanıklı.",
        "params": [
            _param("window_size", "int", 5, 3, 31, "Pencere (tek sayı)"),
        ],
        "fn": rolling_median,
    },
    {
        "id": "double_exponential",
        "name": "Double Exponential (Holt)",
        "description": "Seviye + trend; zaman serisinde trendi takip eder.",
        "params": [
            _param("alpha", "float", 0.3, 0.05, 0.95, "Seviye düzeltme"),
            _param("beta", "float", 0.1, 0.01, 0.5, "Trend düzeltme"),
        ],
        "fn": double_exponential,
    },
    {
        "id": "wiener",
        "name": "Wiener Filter",
        "description": "Adaptif yumuşatma; yerel gürültü tahmini (scipy gerekir, yoksa Gaussian fallback).",
        "params": [
            _param("mysize", "int", 5, 3, 21, "Pencere boyutu"),
        ],
        "fn": wiener,
    },
]


def get_algorithms_list() -> List[Dict[str, Any]]:
    """API için: id, name, description, params (schema)."""
    return [
        {
            "id": a["id"],
            "name": a["name"],
            "description": a["description"],
            "params": a["params"],
        }
        for a in ALGORITHMS
    ]


def run_algorithm(
    algorithm_id: str,
    series: List[float],
    params: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Algoritmayı çalıştırır; parametreler verilmezse varsayılan kullanılır."""
    params = params or {}
    for a in ALGORITHMS:
        if a["id"] == algorithm_id:
            kwargs = {}
            for p in a["params"]:
                name = p["name"]
                kwargs[name] = params.get(name, p["default"])
            # Gaussian: sigma=0 means None (auto)
            if algorithm_id == "gaussian" and kwargs.get("sigma") == 0:
                kwargs["sigma"] = None
            return a["fn"](series, **kwargs)
    raise ValueError(f"Unknown algorithm: {algorithm_id}")
