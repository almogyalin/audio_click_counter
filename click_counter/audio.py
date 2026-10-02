"""Audio loading and format handling.

Scope note: this module handles uncompressed PCM WAV only. The dataset also
contains AAC-in-3GP (``.m4a``) files from a phone microphone; those require an
external decoder (ffmpeg / libsndfile with AAC support) and are NOT supported.
See README "Known limitations".
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass

import numpy as np
from scipy import signal


class AudioLoadError(Exception):
    """Raised when an audio file cannot be decoded."""


@dataclass(frozen=True)
class Audio:
    """Mono float64 samples in [-1, 1]."""

    samples: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    def resample(self, target_rate: int) -> "Audio":
        """Resample to `target_rate` using polyphase FIR filtering."""
        if target_rate == self.sample_rate:
            return self
        from math import gcd

        g = gcd(self.sample_rate, target_rate)
        up = target_rate // g
        down = self.sample_rate // g
        y = signal.resample_poly(self.samples, up, down)
        return Audio(samples=y, sample_rate=target_rate)


def load_audio(path: str | os.PathLike) -> Audio:
    """Load an uncompressed PCM WAV file as mono float64.

    Multi-channel input is downmixed by averaging. Sample width 8/16/32-bit PCM
    is supported; anything else (float WAV, mu-law, compressed) raises
    AudioLoadError.
    """
    path = os.fspath(path)
    try:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except EOFError as exc:
        raise AudioLoadError(
            f"{path}: truncated or empty WAV file (no header data). "
            "Compressed formats (.m4a/.mp3/.aac) need an external decoder."
        ) from exc
    except wave.Error as exc:
        raise AudioLoadError(
            f"{path}: not a supported PCM WAV file ({exc}). "
            "Compressed formats (.m4a/.mp3/.aac) need an external decoder."
        ) from exc
    except OSError as exc:
        raise AudioLoadError(f"{path}: cannot read ({exc})") from exc

    if sampwidth == 1:  # unsigned 8-bit
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:  # signed 16-bit, the dataset's format
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sampwidth == 4:  # signed 32-bit
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise AudioLoadError(
            f"{path}: unsupported sample width {sampwidth * 8}-bit "
            "(supported: 8/16/32-bit PCM)"
        )

    if n_channels > 1:
        usable = (len(data) // n_channels) * n_channels
        data = data[:usable].reshape(-1, n_channels).mean(axis=1)

    return Audio(samples=data, sample_rate=framerate)
