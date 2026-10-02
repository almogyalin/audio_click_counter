"""Tests for audio loading and error handling."""

from __future__ import annotations

import struct
import wave

import numpy as np
import pytest

from click_counter.audio import Audio, AudioLoadError, load_audio

SR = 22050


def write_wav(path, samples, sample_rate=SR, n_channels=1, sampwidth=2):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        if sampwidth == 2:
            q = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 32767 / 32768.0) * 32767.0
            data = np.round(q).astype("<i2").tobytes()
        elif sampwidth == 1:
            data = np.asarray(np.clip(samples, -1, 1) * 127 + 128, dtype=np.uint8).tobytes()
        else:
            data = np.asarray(np.clip(samples, -1, 1) * 2147483647, dtype="<i4").tobytes()
        wf.writeframes(data)
    return path


class TestLoading:
    def test_load_mono_16bit(self, tmp_path):
        x = np.sin(2 * np.pi * 440 * np.arange(SR) / SR) * 0.5
        p = write_wav(tmp_path / "a.wav", x)
        a = load_audio(p)
        assert a.sample_rate == SR
        assert len(a.samples) == SR
        assert abs(a.duration_s - 1.0) < 1e-6
        assert np.max(np.abs(a.samples)) <= 1.0

    def test_stereo_is_downmixed(self, tmp_path):
        # Interleaved float frames, written as a stereo 16-bit WAV. Left and
        # right are equal and opposite, so the downmix must cancel to silence.
        left = np.full(SR, 0.5)
        right = np.full(SR, -0.5)
        inter = np.empty(SR * 2)
        inter[0::2] = left
        inter[1::2] = right
        p = write_wav(tmp_path / "s.wav", inter, n_channels=2)
        a = load_audio(p)
        assert len(a.samples) == SR
        assert np.allclose(a.samples, 0.0, atol=1e-4)

    def test_stereo_duplicate_channels_preserve_signal(self, tmp_path):
        # Both channels identical -> downmix must preserve the waveform, not
        # halve it.
        mono = np.sin(2 * np.pi * 300 * np.arange(SR) / SR) * 0.5
        inter = np.empty(SR * 2)
        inter[0::2] = mono
        inter[1::2] = mono
        p = write_wav(tmp_path / "dup.wav", inter, n_channels=2)
        a = load_audio(p)
        assert len(a.samples) == SR
        # allow one quantization step
        assert np.abs(a.samples - mono).max() < 2e-4

    def test_8bit(self, tmp_path):
        x = np.sin(2 * np.pi * 200 * np.arange(1000) / SR) * 0.5
        p = write_wav(tmp_path / "a8.wav", x, sampwidth=1)
        a = load_audio(p)
        assert len(a.samples) == 1000
        assert np.max(np.abs(a.samples)) <= 1.0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(AudioLoadError, match="cannot read"):
            load_audio(tmp_path / "nope.wav")

    def test_m4a_raises_actionable_error(self, tmp_path):
        # minimal ftyp box, mimicking the dataset's AAC-in-3GP files
        blob = struct.pack(">I", 24) + b"ftyp3gp4" + b"\x00" * 16
        p = tmp_path / "clip.m4a"
        p.write_bytes(blob)
        with pytest.raises(AudioLoadError, match="decoder|P CM|not a supported"):
            load_audio(p)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.wav"
        p.write_bytes(b"")
        with pytest.raises(AudioLoadError):
            load_audio(p)


class TestResampling:
    @pytest.mark.parametrize("target", [8000, 16000, 22050, 48000])
    def test_resample_length_is_correct(self, target):
        x = np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
        a = Audio(samples=x, sample_rate=SR).resample(target)
        assert a.sample_rate == target
        assert abs(a.duration_s - 1.0) < 0.01

    def test_resample_same_rate_is_identity(self):
        x = np.random.default_rng(0).normal(0, 0.1, 1000)
        a = Audio(samples=x, sample_rate=SR)
        assert a.resample(SR) is a

    def test_resample_preserves_tone(self):
        x = np.sin(2 * np.pi * 400 * np.arange(SR) / SR)
        a = Audio(samples=x, sample_rate=SR).resample(11025)
        spec = np.abs(np.fft.rfft(a.samples * np.hanning(len(a.samples))))
        peak_hz = np.fft.rfftfreq(len(a.samples), 1 / 11025)[np.argmax(spec)]
        assert abs(peak_hz - 400) < 40
