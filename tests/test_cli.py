"""Tests for the CLI surface."""

from __future__ import annotations

import json

import numpy as np
import pytest

from click_counter.audio import Audio
from click_counter.cli import main
from click_counter.detector import count_clicks

from .test_detector_synthetic import synth_clicks


def write_wav_bytes(path, samples, sample_rate=44100):
    import wave

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        q = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 32767 / 32768.0) * 32767.0
        wf.writeframes(np.round(q).astype("<i2").tobytes())
    return path


@pytest.fixture
def click_file(tmp_path):
    a = synth_clicks(12)
    return write_wav_bytes(tmp_path / "clicks.wav", a.samples, a.sample_rate)


class TestCLI:
    def test_counts_correctly(self, click_file, capsys):
        rc = main([str(click_file)])
        assert rc == 0
        assert "12 clicks" in capsys.readouterr().out

    def test_json_output(self, click_file, capsys):
        rc = main([str(click_file), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["clicks"] == 12
        assert payload[0]["duration_s"] > 0

    def test_verbose_includes_onsets(self, click_file, capsys):
        main([str(click_file), "--json", "--verbose"])
        payload = json.loads(capsys.readouterr().out)
        assert len(payload[0]["onset_times_s"]) == 12
        assert payload[0]["inter_onset_ms"]["median"] == pytest.approx(100, abs=20)

    def test_multiple_files(self, click_file, tmp_path, capsys):
        b = synth_clicks(5, spacing_s=0.2)
        p2 = write_wav_bytes(tmp_path / "b.wav", b.samples, b.sample_rate)
        main([str(click_file), str(p2), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert [r["clicks"] for r in payload] == [12, 5]

    def test_missing_file_nonzero_exit(self, tmp_path, capsys):
        rc = main([str(tmp_path / "ghost.wav")])
        assert rc == 1
        assert "no such file" in capsys.readouterr().err

    def test_unsupported_format_nonzero_exit(self, tmp_path, capsys):
        p = tmp_path / "x.m4a"
        p.write_bytes(b"\x00\x00\x00\x18ftyp3gp4" + b"\x00" * 16)
        rc = main([str(p)])
        assert rc == 1
        assert "decoder" in capsys.readouterr().err or "not a supported" in capsys.readouterr().err

    def test_warns_above_full_dose(self, tmp_path, capsys):
        a = synth_clicks(80)
        p = write_wav_bytes(tmp_path / "many.wav", a.samples, a.sample_rate)
        main([str(p), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert "warning" in payload[0]

    def test_silence_warns_zero(self, tmp_path, capsys):
        p = write_wav_bytes(tmp_path / "silence.wav", np.zeros(44100), 44100)
        main([str(p), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["clicks"] == 0
        assert "no clicks" in payload[0]["warning"]

    def test_band_override_accepted(self, click_file, capsys):
        rc = main([str(click_file), "--band", "2000", "10000"])
        assert rc == 0

    def test_invalid_band_errors(self, click_file, capsys):
        # exit code 2 distinguishes "bad settings" from "bad/unreadable input"
        rc = main([str(click_file), "--band", "9000", "2000"])
        assert rc == 2
        assert "invalid detector settings" in capsys.readouterr().err

    def test_invalid_threshold_errors(self, click_file, capsys):
        rc = main([str(click_file), "--threshold-k", "-1"])
        assert rc == 2
