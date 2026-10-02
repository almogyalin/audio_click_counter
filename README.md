# audio_click_counter

Count dose-pen clicks in an audio recording.

Given a WAV recording of a **Wegovy 1 mg dose pen** being clicked, this library
returns the number of clicks. The pen holds **75 clicks per full dose**, so a
correct count tells you whether a full dose was administered.

```bash
pip install -e .
click-counter recording.wav
# recording.wav: 75 clicks in 10.75s
```

```python
from click_counter import count_file

result = count_file("recording.wav")
print(result.count)              # 75
print(result.onset_times_s[:3])  # array([1.418, 1.487, 1.561])
print(result.clicks_per_minute)  # 418.6
```

Python 3.10+. Dependencies: `numpy`, `scipy`.

---

## Status: early, and read this before trusting a number

This is a working detector with a **thorough test suite but no held-out
evaluation**. Two things you must know before relying on the accuracy figures:

**1. The reported accuracy is optimistic.** The detector's constants were chosen
by evaluating against the same 33 files the accuracy is then reported on. There
is no held-out split, so these numbers describe fitting, not generalization.

On the 33-file development corpus, with the shipped defaults:

| metric | value |
|---|---|
| exact match | 22/33 (67%) |
| within ±1 click | 27/33 (82%) |
| within ±2 clicks | 31/33 (94%) |
| MAE | 0.58 clicks |
| largest single error | 3 clicks |
| the 8 short segment files | 8/8 exact |
| the 25 full-dose (75-click) files | 14/25 exact, MAE 0.76 |

For reference, an energy-envelope baseline scored **3/33 exact, MAE 24.3** — so
the method matters enormously, but 67% exact is not a production number.

**2. The ground-truth labels are inferred, not human-verified.** They were
derived by reasoning about filenames plus the device's known 75-click capacity.
See [Dataset](#dataset) for exactly how, and [Milestone 1](#milestone-1--ground-truth-is-still-open)
for why this needs human confirmation before the numbers mean anything.

---

## How it works

Clicks are short broadband transients. The hard part is not finding them, it is
not confusing them with everything else.

An **energy envelope** — the obvious first approach — fails badly here. Ambient
noise (street, TV) produces envelope peaks indistinguishable from clicks by
amplitude, and loud recordings clip, which destroys amplitude information
entirely. That baseline over-counted by 472 clicks across 33 files.

The shipped approach is **spectral-flux novelty with temporal background
substitution**:

1. Compute a log-magnitude spectrogram, restricted to **2–10 kHz** where click
   energy concentrates. Measured band sweep: 2–10 kHz gives MAE 0.45; 1–6 kHz
   gives MAE 5.21. The band choice is load-bearing.
2. Subtract a per-frequency **temporal median** background. This is the key
   step and it is what fixes the noise failures. Stationary noise is removed
   because a median over ~0.35 s ignores transients; clicks survive because a
   click occupies a few frames and cannot move a median.
3. Sum positive residuals across frequency → a novelty curve.
4. Peak-pick above `median + 25·MAD`, with a **30 ms refractory period** to
   suppress sub-clicks from one mechanical event.

Thresholds are **relative** (`median + k·MAD`), never absolute, so the detector
adapts to recording level automatically — see the gain-invariance tests.

### Configuration

```python
from click_counter import ClickDetector, count_file

det = ClickDetector(
    band_low_hz=2000.0,      # analysis band
    band_high_hz=10000.0,
    n_fft=1024,
    hop_length=64,           # ~1.5 ms at 44.1 kHz
    background_frames=31,    # temporal median window
    threshold_k=25.0,        # threshold = median + k * MAD
    min_gap_ms=30.0,         # refractory period
    skip_ms=400.0,           # ignore this much file-start
    min_novelty=0.2,         # absolute floor, see below
)
count_file("recording.wav", det)
```

CLI equivalents: `--band LO HI`, `--threshold-k K`, `--min-gap-ms MS`.

**On `min_novelty`.** Without an absolute floor, a recording whose novelty curve
is nearly all exact zeros (a steady tone, a DC offset, silence that survived
filtering) has `MAD == 0`, so the relative threshold collapses onto numerical
noise and dozens of phantom clicks get counted. The floor is set at 0.2 from
measurement: steady tonal/DC content peaks below 0.004, while the weakest
genuine click onset in the corpus is ≥0.8 (median ≈5.6). Without this floor,
`count_clicks` returned 65 "clicks" for a pure 440 Hz sine wave.

---

## Known limitations

These are **measured and asserted in the test suite**
(`TestKnownLimitations`), not speculation. If a future change fixes one, that
test fails and should be updated — that is a good outcome.

| limitation | measured behaviour |
|---|---|
| **SNR floor ~25 dB** | Exact counts hold to ~30 dB SNR. At ~18 dB, counts collapse (30 clicks → 0–6). |
| **Loud ringing double-counts** | A 25 ms-decay source at high level overshoots by one per event. Real pen clicks decay in ~4 ms, so this does not affect them; a user recording a bell would see inflated counts. |
| **`.m4a` unsupported** | Only uncompressed PCM WAV. The dataset's phone recordings are AAC-in-3GP and raise an actionable error. |
| **No direction detection** | Counts forward and backward clicks identically (see below). |
| **Sample rate must exceed ~4 kHz** | Below that the 2 kHz band lower edge exceeds Nyquist; raises `ValueError`. |
| **No uncertainty estimate** | Returns a bare integer. See [Roadmap](#roadmap). |

### Forward vs backward clicks — deliberately not distinguished

On a dose pen, clicking **backward retracts** the dose. A recording containing
25 forward and 5 backward clicks means 20 net, not 30. This library returns 30.

Direction detection was scoped out with the project owner as too much complexity
for the current stage. It is a real correctness gap for any dose-tracking use
case, not a cosmetic one. It needs a labeled forward/backward dataset, which does
not exist yet.

---

## Dataset

**The recordings are not in this repository** (39 MB of WAV plus a redundant
27 MB `data.zip`; both are gitignored). To reproduce the numbers above, place
them at `data/laptop_mic/`.

The device is a Wegovy 1 mg dose pen with **75 clicks per full dose**.

### How ground truth was derived

This matters, because it is the weakest link in the accuracy claim.

**The 8 `N_to_M_forward_click_track.wav` files are trustworthy.** Names state
their own counts (1–10, 11–20, …, 71–75 = 75 total), and a detector tuned on
nothing else reproduces exactly 10, 10, 10, 10, 10, 10, 10, 5. Independent
confirmation.

**The 25 `full_*` / `close_mic_full_*` files are inferred to be 75 clicks each.**
Their names are actively misleading:

- `full_backward_2_click_track.wav` — the `2` is a **take/repetition number, not
  a click count**. Evidence: all 9 `full_forward_N` / `full_backward_N` files
  contain a full dose regardless of `N`, and 75 is the device's known capacity.
- These files were **not** verified to be single continuous takes. Burst
  analysis shows internal 250–600 ms pauses splitting some into 2–5 clusters.

So 25 of 33 labels rest on an inference that has not been confirmed by ear.
If it is wrong, the accuracy figures are wrong. **Confirming this is the single
highest-value next step** and takes about two minutes of listening.

### Not yet checked

The 4 `.m4a` phone recordings (different sample rate, different capture chain)
are entirely unevaluated. They are the real generalization test — see
[Milestone 3](#milestone-3--the-real-generalization-test).

---

## Roadmap

### Milestone 1: ground truth (still open — do this first)

Filenames are not labels, and 25 of 33 labels are inferred. Build a small
annotation tool (waveform + spectrogram, `n`/`d`/arrows to add and move onsets),
have **two people annotate an overlapping subset independently**, and report
inter-annotator agreement. That number sets the floor on achievable accuracy —
if annotators disagree by ±2 clicks, then "67% exact" may be near-perfect.

Until this is done, **treat every accuracy number in this repo as provisional.**

### Milestone 2: freeze a test split

Designate a held-out set **before** further tuning, grouped by capture session
(never by window — with ~30 files, window-level splitting is catastrophic
leakage). Re-measure once. Also add a label-permutation test: shuffle labels and
confirm accuracy collapses to chance. If it doesn't, there is leakage.

Note that `min_gap_ms=20` scored better during tuning (25/33 exact, MAE 0.36)
than the shipped 30 ms. The worse value was kept deliberately — taking the best
number from a sweep over the evaluation set is exactly the overfitting this
project is trying to avoid. Settle it on held-out data.

### Milestone 3: the real generalization test

Train nothing yet, but measure across devices: laptop mic vs phone mic, near vs
far, quiet vs street/TV noise. This gap will be much wider than the aggregate
number, and it is the honest expectation for a user in a new room with a new
phone.

### Milestone 4: uncertainty, then product

Return `73 ± 4` rather than a bare integer — a count app that admits uncertainty
is far more useful. Then a CLI/app wrapper, plus inference logging so real-world
failures become new labeled data.

### Anti-overfitting rules for whoever continues this

- Prefer the simple detector. If a 1-D CNN does not beat it outside noise
  **on held-out data**, ship the DSP detector.
- With ~30 files that are *not* independent (one pen, one room), augmentation
  and a strong DSP prior matter more than model capacity. Mixing verified
  isolated clicks into the provided noise files at controlled SNR is the
  highest-value data path if ML becomes necessary.
- Report per-condition error breakdowns, not just aggregates. A model that
  always answers 75 looks great on this corpus.
- Write down the hyperparameter search space and trial count *before* running it.

---

## Development

```bash
pip install -e ".[dev]"
pytest                    # 61 tests
```

The test suite uses **synthetic signals with known ground truth**, so it is
independent of the real dataset and of its label assumptions. It covers exact
counts across click densities and shapes, SNR sweep, gain invariance (0.01×–2×),
time-shift invariance, silence/DC/tone/white-noise returning zero, WAV format
handling, resampling, and the CLI surface.

`TestKnownLimitations` asserts current *failure* modes explicitly so they cannot
silently change.

### Repo layout

```
click_counter/
  detector.py   # ClickDetector, novelty_curve, count_clicks, count_file
  audio.py      # WAV loading, downmix, resampling
  cli.py        # command-line interface
tests/
  test_detector_synthetic.py
  test_audio.py
  test_cli.py
```

`PLAN.md` is the original project plan, including the reconnaissance that
motivated the current design. **Its §1 findings are partly superseded** — it
describes the recordings as "near-continuous full-scale mass," which turned out
to be an artifact of plotting max-pooled 95 ms columns on clipping files. The
signals are sparse clicks.

## License

MIT
