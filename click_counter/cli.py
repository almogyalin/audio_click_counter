"""Command-line interface: count clicks in an audio recording."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .audio import AudioLoadError, load_audio
from .detector import ClickDetector, count_clicks

# A dose pen holds 75 clicks per full dose (Wegovy 1 mg). Used only as a
# plausibility warning -- never to correct or clamp the returned count.
FULL_DOSE_CLICKS = 75


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="click-counter",
        description=(
            "Count dose-pen clicks in a WAV recording. "
            "Counts clicks of any direction; does not distinguish forward "
            "(dose-advancing) from backward (dose-retracting) clicks."
        ),
    )
    p.add_argument("audio", nargs="+", help="WAV file(s) to analyze")
    p.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="show onset times and inter-onset intervals",
    )
    p.add_argument(
        "--band", nargs=2, type=float, metavar=("LO", "HI"),
        help="analysis band in Hz (default 2000 10000)",
    )
    p.add_argument(
        "--threshold-k", type=float, metavar="K",
        help="peak threshold as median + K*MAD (default 25)",
    )
    p.add_argument(
        "--min-gap-ms", type=float, metavar="MS",
        help="refractory period between onsets in ms (default 30)",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def detector_from_args(args: argparse.Namespace) -> ClickDetector:
    """Build a detector from CLI overrides.

    Invalid values raise ValueError from the ClickDetector constructor; the
    caller turns that into a clean message and a non-zero exit rather than a
    traceback.
    """
    kwargs: dict[str, float | int] = {}
    if args.band:
        kwargs["band_low_hz"] = args.band[0]
        kwargs["band_high_hz"] = args.band[1]
    if args.threshold_k is not None:
        kwargs["threshold_k"] = args.threshold_k
    if args.min_gap_ms is not None:
        kwargs["min_gap_ms"] = args.min_gap_ms
    return ClickDetector(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        det = detector_from_args(args)
    except ValueError as exc:
        print(f"error: invalid detector settings: {exc}", file=sys.stderr)
        return 2

    records = []
    had_error = False

    for name in args.audio:
        path = Path(name)
        if not path.exists():
            print(f"error: no such file: {name}", file=sys.stderr)
            had_error = True
            continue
        try:
            result = count_clicks(load_audio(path), det)
        except (AudioLoadError, ValueError) as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            had_error = True
            continue

        record = {
            "file": str(path),
            "clicks": result.count,
            "duration_s": round(result.duration_s, 3),
            "clicks_per_minute": round(result.clicks_per_minute, 2),
        }
        if result.count == 0:
            record["warning"] = "no clicks detected"
        elif result.count > FULL_DOSE_CLICKS:
            record["warning"] = (
                f"count exceeds {FULL_DOSE_CLICKS}; pen holds one full dose"
            )
        if args.verbose:
            record["onset_times_s"] = [round(t, 4) for t in result.onset_times_s]
            ioi = result.inter_onset_intervals_ms()
            if len(ioi):
                record["inter_onset_ms"] = {
                    "median": round(float(np.median(ioi)), 1),
                    "min": round(float(ioi.min()), 1),
                    "max": round(float(ioi.max()), 1),
                }
        records.append(record)

    if args.json:
        print(json.dumps(records, indent=2))
    else:
        for rec in records:
            line = f"{rec['file']}: {rec['clicks']} clicks in {rec['duration_s']}s"
            if "warning" in rec:
                line += f"  [warning: {rec['warning']}]"
            print(line)
            if args.verbose and "onset_times_s" in rec:
                print(f"  onsets (s): {rec['onset_times_s']}")
                if "inter_onset_ms" in rec:
                    ioi = rec["inter_onset_ms"]
                    print(
                        f"  inter-onset (ms): median={ioi['median']} "
                        f"min={ioi['min']} max={ioi['max']}"
                    )

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
