"""Model doğrulama ve dağılım karşılaştırma yardımcıları."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.special import kl_div


def kl_divergence_histograms(
    original_values: Iterable[float],
    generated_values: Iterable[float],
    bins: int = 20,
) -> float:
    """
    Orijinal ve üretilmiş feature dağılımları arasındaki KL diverjansı.
    """
    orig = np.asarray(list(original_values), dtype=float)
    gen = np.asarray(list(generated_values), dtype=float)
    if orig.size == 0 or gen.size == 0:
        return float("nan")

    orig_hist, edges = np.histogram(orig, bins=bins, density=True)
    gen_hist, _ = np.histogram(gen, bins=edges, density=True)

    orig_hist = orig_hist + 1e-8
    gen_hist = gen_hist + 1e-8

    return float(np.sum(kl_div(orig_hist, gen_hist)))


def compare_feature_distributions(
    original_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    feature_name: str,
) -> float:
    """Belirli bir feature için orijinal vs üretilmiş KL diverjansını döndür."""
    return kl_divergence_histograms(original_df[feature_name], generated_df[feature_name])


def basic_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Basit bir train/test bölmesi (sklearn bağımlılığı olmadan).
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)

    split = int(n * (1.0 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


__all__ = [
    "kl_divergence_histograms",
    "compare_feature_distributions",
    "basic_train_test_split",
]

