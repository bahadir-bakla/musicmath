"""
Proje genelinde kullanılacak logging yapılandırması.

Kullanım:
    from music_math.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Mesaj")
"""

from __future__ import annotations

import logging
from logging import Logger


def _configure_root_logger() -> None:
    """Basit ama okunabilir bir logging formatı ayarla."""
    if logging.getLogger().handlers:
        # Zaten yapılandırılmış
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str | None = None) -> Logger:
    """Verilen isim için bir logger döndür."""
    _configure_root_logger()
    return logging.getLogger(name)


__all__ = ["get_logger"]

