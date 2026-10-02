"""Click counting via spectral-flux novelty with temporal background subtraction.

Method
------
Clicks from a dose pen are short broadband transients. A plain energy envelope
does not work on this dataset: loud recordings clip, and steady ambient noise
(street, TV) produces envelope peaks indistinguishable from clicks by
amplitude alone. An early energy-based baseline scored 3/33 files exactly.

The approach here is instead:

1. Compute a log-magnitude spectrogram restricted to the band where click
   energy concentrates (empirically 2-10 kHz).
2. Subtract a per-frequency *temporal median* background. This removes
   stationary noise -- the dominant failure mode -- while leaving transient
   onsets intact, because a click occupies a few frames and cannot move a
   median over a ~0.35 s window.
3. Sum the positive residuals across frequency to get a novelty curve.
4. Peak-pick above `median + k * MAD` of the novelty curve, with a refractory
   period to suppress sub-clicks from a single mechanical event.

Thresholding is relative (median + k*MAD), never absolute, so it adapts to
recording level automatically.

IMPORTANT: see README "Honest limitations" -- the default constants were
selected by evaluating against all 33 dataset files, so the reported accuracy
is optimistic and is NOT a held-out benchmark. Do not treat `k`/`min_gap_ms`
as tuned for new data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from .audio import Audio, load_audio


@dataclass(frozen=True)
class ClickDetector:
    """Detector configuration.

    Defaults correspond to the best configuration found on the 33-file
    development corpus (see module docstring caveat).
    """

    band_low_hz: float = 2000.0
    band_high_hz: float = 10000.0
    n_fft: int = 1024
    hop_length: int = 64
    background_frames: int = 31
    threshold_k: float = 25.0
    min_gap_ms: float = 30.0
    skip_ms: float = 400.0
    log_scale: float = 1e4
    # Absolute novelty floor. Without it, a recording whose novelty curve is
    # almost entirely exact zeros -- a steady tone, a DC offset, digital silence
    # that survived filtering -- has mad == 0, so the relative threshold collapses
    # onto numerical noise and dozens of phantom "clicks" get counted.
    #
    # Justification: measured peak novelty of steady tonal/DC content is <0.004,
    # while the weakest genuine click onset across the 33-file development corpus
    # is >=0.8 (median ~5.6). 0.2 sits ~50x above the noise floor and ~4x below
    # the weakest real click.
    min_novelty: float = 0.2

    def __post_init__(self) -> None:
        if not 0 < self.band_low_hz < self.band_high_hz:
            raise ValueError(
                f"invalid band {self.band_low_hz}-{self.band_high_hz} Hz"
            )
        if self.n_fft < 64:
            raise ValueError("n_fft must be >= 64")
        if self.hop_length < 1 or self.hop_length > self.n_fft:
            raise ValueError("hop_length must be in [1, n_fft]")
        if self.background_frames < 3:
            raise ValueError("background_frames must be >= 3")
        if self.threshold_k <= 0:
            raise ValueError("threshold_k must be > 0")
        if self.min_gap_ms <= 0:
            raise ValueError("min_gap_ms must be > 0")
        if self.min_novelty < 0:
            raise ValueError("min_novelty must be >= 0")


@dataclass
class CountResult:
    """Outcome of counting one recording."""

    count: int
    onset_times_s: np.ndarray
    novelty_threshold: float
    duration_s: float
    detector: ClickDetector = field(repr=False)

    @property
    def clicks_per_minute(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return 60.0 * self.count / self.duration_s

    def inter_onset_intervals_ms(self) -> np.ndarray:
        if len(self.onset_times_s) < 2:
            return np.array([], dtype=float)
        return np.diff(self.onset_times_s) * 1000.0


def novelty_curve(
    audio: Audio, detector: ClickDetector | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_s, novelty) for a recording.

    Novelty is the summed positive deviation of a log-magnitude spectrogram from
    its temporal median background, over the configured band.
    """
    det = detector or ClickDetector()
    x = audio.samples
    sr = audio.sample_rate

    freqs, times, spec = signal.spectrogram(
        x,
        sr,
        nperseg=det.n_fft,
        noverlap=det.n_fft - det.hop_length,
        window="hann",
        mode="magnitude",
    )

    nyquist = sr / 2.0
    hi = min(det.band_high_hz, nyquist * 0.98)
    lo = det.band_low_hz
    if lo >= hi:
        raise ValueError(
            f"band {lo}-{hi} Hz is empty for a {sr} Hz signal "
            "(sample rate too low for the configured band)"
        )
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        raise ValueError(f"no FFT bins fall inside {lo}-{hi} Hz at n_fft={det.n_fft}")

    log_spec = np.log1p(spec[band] * det.log_scale)
    background = median_filter(log_spec, size=(1, det.background_frames))
    novelty = np.maximum(log_spec - background, 0.0).sum(axis=0)
    return times, novelty


def count_clicks(
    audio: Audio, detector: ClickDetector | None = None
) -> CountResult:
    """Count clicks in a loaded recording."""
    det = detector or ClickDetector()
    times, novelty = novelty_curve(audio, det)

    frame_dt = float(times[1] - times[0]) if len(times) > 1 else 0.0
    median = float(np.median(novelty))
    mad = float(np.median(np.abs(novelty - median)))
    # Relative threshold (median + k*MAD) with an absolute floor so a degenerate
    # zero-MAD novelty curve cannot admit numerical noise.
    threshold = max(median + det.threshold_k * mad, det.min_novelty)

    min_gap_frames = max(1, int(round(det.min_gap_ms / 1000.0 / frame_dt))) if frame_dt > 0 else 1
    peaks, _ = signal.find_peaks(novelty, height=threshold, distance=min_gap_frames)

    onset_times = times[peaks]
    onset_times = onset_times[onset_times >= det.skip_ms / 1000.0]

    return CountResult(
        count=int(len(onset_times)),
        onset_times_s=onset_times,
        novelty_threshold=threshold,
        duration_s=audio.duration_s,
        detector=det,
    )


def count_file(
    path: str | os.PathLike, detector: ClickDetector | None = None
) -> CountResult:
    """Load an audio file and count its clicks."""
    return count_clicks(load_audio(path), detector)
