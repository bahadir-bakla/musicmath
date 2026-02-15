"""
Merkezi konfigürasyon yönetimi.

Basit ama genişleyebilir bir yapı: varsayılan yolları ve temel
hiperparametreleri tek noktadan tanımlar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Proje boyunca kullanılan temel dizin yolları."""

    root: Path
    data_raw: Path
    data_clean: Path
    notebooks: Path
    results: Path
    figures: Path
    generated_midi: Path
    stats: Path
    interactive: Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Deney ve modelleme ile ilgili temel hiperparametreler."""

    random_seed: int = 42


@dataclass(frozen=True)
class AppConfig:
    """Genel uygulama konfigürasyonu."""

    paths: Paths
    experiment: ExperimentConfig = ExperimentConfig()


def load_config(root: str | Path | None = None) -> AppConfig:
    """
    Konfigürasyonu oluştur.

    Şimdilik yalnızca dizin yapılarını ve birkaç sabiti içeriyor;
    ileride `configs/*.yaml` dosyalarından okunacak şekilde genişletilebilir.
    """
    root_path = Path(root) if root is not None else Path.cwd()

    paths = Paths(
        root=root_path,
        data_raw=root_path / "data" / "raw",
        data_clean=root_path / "data" / "clean",
        notebooks=root_path / "notebooks",
        results=root_path / "results",
        figures=root_path / "results" / "figures",
        generated_midi=root_path / "results" / "generated_midi",
        stats=root_path / "results" / "stats",
        interactive=root_path / "results" / "interactive",
    )

    return AppConfig(paths=paths)


CONFIG = load_config()

