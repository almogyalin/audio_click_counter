"""Onset detection and click counting for dose-pen recordings."""

from .detector import ClickDetector, CountResult, count_clicks, count_file
from .audio import load_audio, AudioLoadError

__all__ = [
    "ClickDetector",
    "CountResult",
    "count_clicks",
    "count_file",
    "load_audio",
    "AudioLoadError",
]

__version__ = "0.1.0"
