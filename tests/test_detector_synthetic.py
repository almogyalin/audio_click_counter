"""Synthetic-signal tests for the click detector.

These use generated audio with a known ground-truth click count, so they are
independent of the real dataset and of any label assumptions in it.
"""

from __future__ import annotations

import numpy as np
import pytest

from click_counter.audio import Audio
from click_counter.detector import ClickDetector, count_clicks

SR = 44100


def synth_clicks(
    n_clicks: int,
    spacing_s: float = 0.10,
    duration_s: float | None = None,
    noise_rms: float = 0.0,
    seed: int = 0,
    click_kind: str = "sharp",
) -> Audio:
    """Generate `n_clicks` broadband transients at a regular interval.

    `click_kind` controls the transient shape:
      sharp   - ~3 ms damped 6 kHz burst (closest to a real dose-pen click)
      soft    - ~4 ms damped 2.5 kHz burst (quieter, duller click)
      mixed   - alternate sharp/soft (tests robustness to click variability)
      ringing - 25 ms decay, i.e. a long-ringing struck object. A real pen does
                NOT do this; see TestKnownLimitations::test_ringing_double_counts.

    Decay constants are deliberately short. Measured: a 6 kHz burst with >=6 ms
    decay starts producing two novelty peaks per physical click, because the
    ringing extends past the detector's 30 ms refractory period. Real dose-pen
    clicks decay in ~4 ms. Do not lengthen these to "sound more realistic"
    without re-checking TestKnownLimitations::test_ringing_double_counts.

    Amplitude is kept at 0.3 peak so that callers can apply gain up to ~2x
    without clipping in float; see TestInvariances.
    """
    rng = np.random.default_rng(seed)
    if duration_s is None:
        duration_s = 1.0 + n_clicks * spacing_s + 0.5
    n = int(duration_s * SR)
    x = np.zeros(n)

    if noise_rms > 0:
        x += rng.normal(0.0, noise_rms, n)

    t = np.arange(n) / SR
    for i in range(n_clicks):
        onset = 1.0 + i * spacing_s
        kind = click_kind
        if click_kind == "mixed":
            kind = "sharp" if i % 2 == 0 else "soft"
        if kind == "sharp":
            f0, decay = 6000.0, 0.003
        elif kind == "soft":
            f0, decay = 2500.0, 0.004
        else:  # ringing
            f0, decay = 3000.0, 0.025
        local = t - onset
        mask = (local >= 0) & (local < decay * 8)
        env = np.exp(-local[mask] / decay)
        burst = env * np.sin(2 * np.pi * f0 * local[mask])
        # add a harmonic so it is not a pure tone
        burst += 0.4 * env * np.sin(2 * np.pi * 2.7 * f0 * local[mask])
        x[np.where(mask)[0]] += burst * 0.3

    return Audio(samples=x, sample_rate=SR)


class TestSyntheticCounts:
    @pytest.mark.parametrize("n", [1, 5, 10, 25, 50, 75])
    def test_exact_count_clean(self, n):
        result = count_clicks(synth_clicks(n))
        assert result.count == n, f"expected {n}, got {result.count}"

    @pytest.mark.parametrize("n", [10, 75])
    def test_exact_count_with_noise(self, n):
        result = count_clicks(synth_clicks(n, noise_rms=0.01, seed=1))
        assert result.count == n, f"expected {n}, got {result.count}"

    @pytest.mark.parametrize("n", [10, 75])
    def test_exact_count_moderate_noise(self, n):
        # ~30 dB SNR. Measured operating limit is ~25 dB; see
        # TestKnownLimitations::test_accuracy_degrades_below_25db_snr.
        result = count_clicks(synth_clicks(n, noise_rms=0.02, seed=2))
        assert result.count == n, f"expected {n}, got {result.count}"

    def test_soft_clicks(self):
        result = count_clicks(synth_clicks(30, click_kind="soft"))
        assert result.count == 30

    def test_mixed_click_shapes(self):
        result = count_clicks(synth_clicks(40, click_kind="mixed"))
        assert result.count == 40

    def test_varying_spacing(self):
        a = synth_clicks(20, spacing_s=0.06)
        b = synth_clicks(20, spacing_s=0.20)
        assert count_clicks(a).count == 20
        assert count_clicks(b).count == 20


class TestEdgeCases:
    def test_silence_returns_zero(self):
        audio = Audio(samples=np.zeros(SR), sample_rate=SR)
        assert count_clicks(audio).count == 0

    def test_dc_offset_returns_zero(self):
        t = np.arange(SR) / SR
        audio = Audio(samples=0.5 * np.ones(SR) + 0.01 * np.sin(2 * np.pi * 50 * t), sample_rate=SR)
        assert count_clicks(audio).count == 0

    def test_pure_noise_returns_near_zero(self):
        rng = np.random.default_rng(7)
        audio = Audio(samples=rng.normal(0, 0.05, SR * 3), sample_rate=SR)
        assert count_clicks(audio).count == 0

    def test_tonal_noise_returns_zero(self):
        t = np.arange(SR * 3) / SR
        audio = Audio(samples=0.2 * np.sin(2 * np.pi * 440 * t), sample_rate=SR)
        assert count_clicks(audio).count == 0


class TestInvariances:
    """Gain must not change the count: thresholds are relative, not absolute."""

    # Gains stay <= 2.0 because the synthetic peak is 0.3 and float samples are
    # nominally [-1, 1]; beyond that the test would be measuring clipping.
    @pytest.mark.parametrize("gain", [0.01, 0.1, 0.5, 1.0, 2.0])
    def test_gain_invariance(self, gain):
        base = synth_clicks(30)
        scaled = Audio(samples=base.samples * gain, sample_rate=SR)
        assert np.abs(scaled.samples).max() <= 1.0
        assert count_clicks(scaled).count == 30

    @pytest.mark.parametrize("offset", [0.0, 1.0, 3.0])
    def test_time_shift_invariance(self, offset):
        base = synth_clicks(30)
        shifted = Audio(
            samples=np.concatenate([np.zeros(int(offset * SR)), base.samples]),
            sample_rate=SR,
        )
        assert count_clicks(shifted).count == 30


class TestRatePlausibility:
    def test_reported_rate_matches(self):
        result = count_clicks(synth_clicks(30, spacing_s=0.1))
        # 30 clicks over ~4.5s -> roughly 400 clicks/min
        assert 300 < result.clicks_per_minute < 500

    def test_intervals_are_regular(self):
        result = count_clicks(synth_clicks(20, spacing_s=0.12))
        ioi = result.inter_onset_intervals_ms()
        assert len(ioi) == 19
        assert np.std(ioi) < 15.0
        assert abs(np.median(ioi) - 120.0) < 15.0


class TestKnownLimitations:
    """Characterize failure modes explicitly so they stay visible.

    Each test asserts the *observed* behaviour, including where it is wrong.
    If a future change fixes one of these, the test fails and should be updated
    -- that is a good outcome, not a regression.
    """

    def test_ringing_is_tolerated_at_low_level(self):
        # At the generator's nominal 0.3 amplitude, a 25 ms decay is handled
        # correctly: the ring stays below min_novelty.
        assert count_clicks(synth_clicks(10, spacing_s=0.15, click_kind="ringing")).count == 10

    def test_loud_ringing_double_counts(self):
        # Scaling a ringing source up pushes its tail above min_novelty, and the
        # count overshoots by one per event. Real dose-pen clicks decay in ~4 ms
        # and are unaffected; a user recording a bell or loose metal would see
        # inflated counts. Documented rather than fixed -- the fix is a
        # longer refractory period, which costs resolution on fast click trains.
        loud = Audio(samples=synth_clicks(10, spacing_s=0.15, click_kind="ringing").samples * 2.0,
                     sample_rate=SR)
        assert count_clicks(loud).count > 10

    def test_accuracy_degrades_below_25db_snr(self):
        # Measured breakdown point: counts stay exact to noise_rms=0.02 (~30 dB
        # SNR) and collapse by noise_rms=0.08 (~18 dB). This documents the
        # operating envelope; it is not a pass/fail quality gate.
        assert count_clicks(synth_clicks(30, noise_rms=0.02, seed=1)).count == 30
        assert count_clicks(synth_clicks(30, noise_rms=0.08, seed=1)).count < 30

    def test_counts_above_full_dose_are_still_reported(self):
        # The pen holds 75 clicks, but the library does not clamp: a longer
        # recording is counted honestly and the CLI adds a warning.
        assert count_clicks(synth_clicks(90)).count == 90


class TestDetectorConfig:
    def test_invalid_band_rejected(self):
        with pytest.raises(ValueError):
            ClickDetector(band_low_hz=8000, band_high_hz=2000)

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            ClickDetector(threshold_k=0)

    def test_band_above_nyquist_raises_clear_error(self):
        # Nyquist below band_low_hz -> nothing to analyze. Decimate far enough
        # that sample_rate < 2 * band_low_hz.
        det = ClickDetector()
        audio = Audio(
            samples=synth_clicks(10).samples[::22], sample_rate=SR // 22
        )
        assert audio.sample_rate < 2 * det.band_low_hz
        with pytest.raises(ValueError, match="empty"):
            count_clicks(audio)

    def test_band_clipped_to_nyquist_still_counts(self):
        # 8 kHz input: Nyquist 4 kHz, so the band becomes 2-3.9 kHz. The soft
        # 2.5 kHz click is still inside it and must still be counted.
        audio = Audio(samples=synth_clicks(20, click_kind="soft").samples[::5], sample_rate=SR // 5)
        result = count_clicks(audio)
        assert result.count == 20

    def test_low_sample_rate_still_counts(self):
        # 16 kHz keeps Nyquist (8 kHz) above band_low_hz, so counting still works
        audio = Audio(samples=synth_clicks(20).samples[::2], sample_rate=SR // 2)
        result = count_clicks(audio)
        assert result.count == 20

    def test_resample_preserves_count(self):
        base = synth_clicks(30)
        r = count_clicks(base.resample(22050))
        assert abs(r.count - 30) <= 1
