#!/usr/bin/env python
"""
Era ve besteci siniflandirma analizleri.

Kullanim:
    python scripts/classification_analysis.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from music_math.core.config import CONFIG
from music_math.core.logging import get_logger


logger = get_logger(__name__)


META_COLS = {"filepath", "composer", "era", "form"}


def load_feature_matrix() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """results/stats/feature_matrix.csv dosyasini yukle ve standardize et."""
    feat_path = CONFIG.paths.root / "results" / "stats" / "feature_matrix.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_matrix.csv bulunamadi: {feat_path}")

    df = pd.read_csv(feat_path)
    logger.info("Feature matrix yuklendi: shape=%s", df.shape)

    feat_cols = [
        c
        for c in df.columns
        if c not in META_COLS and np.issubdtype(df[c].dtype, np.number)
    ]
    X = df[feat_cols].fillna(0.0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, feat_cols


def _filter_min_samples(
    labels: pd.Series, min_per_class: int = 8
) -> Tuple[pd.Series, np.ndarray]:
    """Az ornekli siniflari filtrele ve mask dondur."""
    counts = labels.value_counts()
    keep = counts[counts >= min_per_class].index
    mask = labels.isin(keep)
    return labels[mask], mask.to_numpy()


def evaluate_classification(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    target_col: str,
    min_per_class: int,
    title_prefix: str,
) -> None:
    """Verilen hedef kolonu icin RandomForest ile siniflandirma yap."""
    if target_col not in df.columns:
        logger.warning("'%s' kolonu yok, atlaniyor.", target_col)
        return

    y_raw = df[target_col]
    # NaN ve 'Unknown'lari temizle
    y = y_raw.replace("Unknown", np.nan).dropna()
    if y.empty:
        logger.warning("'%s' icin yeterli etiket yok, atlaniyor.", target_col)
        return

    y_filtered, mask = _filter_min_samples(y, min_per_class=min_per_class)
    if y_filtered.nunique() < 2:
        logger.warning(
            "'%s' icin min_per_class=%d sonrasinda en az iki sinif kalmadi, atlaniyor.",
            target_col,
            min_per_class,
        )
        return

    X_use = X_scaled[mask]
    y_use = y_filtered.astype("category")
    class_names = list(y_use.cat.categories)
    y_codes = y_use.cat.codes.to_numpy()

    logger.info(
        "%s siniflandirma: %d ornek, %d sinif: %s",
        target_col,
        len(y_codes),
        len(class_names),
        class_names,
    )

    clf = RandomForestClassifier(
        n_estimators=300, random_state=42, max_depth=None, n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=min(5, len(np.unique(y_codes))), shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_use, y_codes, cv=cv, n_jobs=-1)

    acc = accuracy_score(y_codes, y_pred)
    print("\n" + "=" * 60)
    print(f"{title_prefix} – hedef: {target_col}")
    print("=" * 60)
    print(f"Toplam ornek: {len(y_codes)}  |  Sinif sayisi: {len(class_names)}")
    print(f"Cross-val accuracy (RandomForest): {acc:.3f}\n")

    print("Sinif bazinda istatistikler:")
    print(classification_report(y_codes, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_codes, y_pred, labels=range(len(class_names)))
    figures_dir = CONFIG.paths.figures
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(include_values=True, cmap="Blues", xticks_rotation=45, colorbar=True)
    plt.title(f"{title_prefix} – Confusion Matrix ({target_col})")
    plt.tight_layout()

    out_path = figures_dir / f"confusion_{target_col}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Confusion matrix kaydedildi: %s", out_path)


def main() -> None:
    print("=" * 60)
    print("Era & Besteci Siniflandirma Analizleri")
    print("=" * 60)

    df, X_scaled, _ = load_feature_matrix()

    # Era siniflandirma
    evaluate_classification(
        df=df,
        X_scaled=X_scaled,
        target_col="era",
        min_per_class=12,
        title_prefix="Dönem Siniflandirma",
    )

    # Besteci siniflandirma
    evaluate_classification(
        df=df,
        X_scaled=X_scaled,
        target_col="composer",
        min_per_class=8,
        title_prefix="Besteci Siniflandirma",
    )

    print("\nAnalizler tamamlandi. Confusion matrix figurleri 'results/figures' altinda.")


if __name__ == "__main__":
    main()

