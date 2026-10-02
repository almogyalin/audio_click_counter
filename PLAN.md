# Click Counter — Project Plan (Draft v0.1)

**Goal:** an app that ingests a recording of a Wegovy 1 mg dose pen and returns the
number of clicks.

**Status of this document:** drafted *after* an initial data reconnaissance pass, so the
milestones below are shaped by what is actually in `data/`. Numbers quoted in §1 are
measured, not assumed.

---

## 1. What the data reconnaissance actually found

I ran a quick characterization of all 33 WAV files (pure-Python, no deps). Findings that
change the plan:

**Corpus shape**
- 33 WAVs, mono, 16-bit, 44.1 kHz, durations 5.2 s – 36.1 s. Uniform format — good.
- 4 additional files in `data/phone_s25_mic/` are AAC-in-3GP (`.m4a`). **No decoder is
  available in this environment** (no ffmpeg, no sox, no pip).
- `data/data.md`: the device is a Wegovy 1 mg dose pen, 75 clicks per full dose. So the
  label space is roughly 0–75+, and the app has a natural upper bound.

**The naive baseline fails badly.** A robust energy-envelope peak detector
(1 ms hop, 4 ms window, MAD-based threshold, 45 ms refractory) scored:

| metric | value |
|---|---|
| exact-match files | **3 / 33 (9 %)** |
| MAE | **24.3 clicks** |
| total predicted vs expected | 1555 vs 1083 (**+472 bias**) |

**The most alarming result — the detector is not tracking click count at all.**
Comparing files whose names imply wildly different counts:

| file | name implies | detector output | median inter-onset gap |
|---|---|---|---|
| `close_mic_full_backwards_click_track` | 75 | 70 peaks | 69 ms |
| `full_backward_2_click_track` | 2 | 65 peaks | 94 ms |

A 2-click file and a 75-click file produce near-identical peak streams. Either the
filename semantics are not what I assumed, or a pervasive mechanical/ambient noise source
is being mistaken for clicks. **Both possibilities must be resolved before modeling.**

**Loud files look saturated and near-continuous.** Amplitude traces of the loud files show
a continuous full-scale mass spanning the whole recording, not a sparse event train:
- peak amplitude pinned at 32768 with 0.06–0.17 % of samples at/near full scale (clipping)
- ~30 % of frame energy below 300 Hz, plus strong sub-20 Hz / near-DC content
- Energy is broadly spread, so there is no obvious narrow bandpass that isolates "clicks"

**Quiet files are a different capture condition.** `21_to_30`, `31_to_40`, `41_to_50`,
`61_to_70` have RMS 166–255 and zero clipping, versus RMS ~2000–3000 with clipping for the
rest. The 10-click segment files are quiet; the "full" 75-click files are loud. A single
global threshold will not serve both.

**Environment is bare.** Python 3.14.4 with no numpy, scipy, pip, ffmpeg, or sox.
(`audioop` was removed in 3.14.) All analysis so far was hand-rolled. This is a hard
blocker for both development and any `.m4a` support.

> **Bottom line:** the difficulty in this project is *not* "detect a click." It is
> (a) establishing trustworthy labels, and (b) separating clicks from a loud, clipping,
> possibly mechanical background. The plan is sequenced accordingly.

---

## 2. Open questions that materially change the design

Please answer before I start Milestone 1:

1. **Label semantics.** For `full_backward_2_click_track.wav`, is "2" the number of
   clicks? Or is it take/repetition #2 of a *full 75-click* backward track? The evidence
   above says one of these is wrong and I don't want to guess.
2. **What is the app's actual job?** (a) count total clicks heard, or (b) count
   *dose-advancing* clicks? On a dose pen, clicks turned backwards retract the dose, so
   "count" may require **direction detection**, which roughly doubles the problem. Which
   do you want?
3. **Counting range.** Is a click track that goes past 75 in one recording in scope
   (multi-dose), or is 0–75 guaranteed?
4. **Deployment.** On-device (phone) or server? This decides model size budget and
   whether the `.m4a` phone files must be supported at all.
5. **More data?** How much more can you record? See §5 — this is the single biggest
   lever on final accuracy, and it is cheap for you to produce.

---

## 3. Roadmap and milestones

### Milestone 0 — Environment and I/O (small, unblocks everything)
- Get a working scientific Python stack (`numpy`, `scipy`, `scikit-learn`, `soundfile`/
  `librosa`). Decide the interpreter — 3.14 is bleeding-edge; **3.11/3.12 is the safer
  target** and avoids `audioop` removal issues.
- Add an audio decode path for AAC/`.m4a` (ffmpeg via `imageio-ffmpeg` wheel, or
  `soundfile`+`libsndfile` if it can read the 3GP/AAC).
- Normalize all inputs to a canonical format (mono, 44.1 kHz, float32) in one loader
  module, with a loudness normalization step that is *estimated on train data only*
  (see §4, leakage).
- **Done when:** every one of the 37 files loads and reports a plausible duration.

### Milestone 1 — Ground truth (critical path, do not skip)
Nothing downstream is trustworthy without this. Filenames are not labels.
- Build a small annotation tool: a script that renders a waveform + spectrogram with
  click onsets marked, and lets you add/correct onsets by ear (keyboard: `n` next,
  `d` delete, arrow keys to nudge). Pure-Python `wave` + a self-contained plotting path is
  enough; no framework needed.
- Two annotators, independently, on an overlapping subset → report **inter-annotator
  agreement**. This is also the first real estimate of label noise, which sets a floor on
  achievable accuracy.
- Reconcile the §2 label-semantics question; write the mapping down as a spec.
- **Near-duplicate audit (important):** the `1_to_10` … `71_to_75` files may be segments of
  one continuous 75-click take. If so, they are *not* 8 independent examples and treating
  them as such is one of the nastiest overfitting traps available here. Check explicitly
  (envelope cross-correlation of file boundaries) and record the grouping in a manifest.
- Build `manifest.csv`: `file, mic, condition, n_clicks, onset_times_ms, annotator,
  is_segment_of, duration_s`.
- **Done when:** ≥8 files fully annotated with agreement ≥0.9 (or documented disagreement),
  and a manifest exists.

### Milestone 2 — DSP baseline (the thing to beat)
Deliberately simple and interpretable, so that any ML model has to earn its complexity.
- Detect **clipping/saturation** explicitly (run-length of near-full-scale samples) and
  expose it as a feature/flag rather than silently normalizing it away.
- Adaptive noise-floor tracking (e.g. running low-quantile of the envelope) so one
  threshold serves both the quiet 10-click files and the loud 75-click files.
- Bandpass to the band that actually carries click energy (to be determined empirically in
  M2, not guessed).
- Onset picking: envelope + local-max + adaptive threshold + **refractory period derived
  from the observed click-rate distribution** (the current fixed 45 ms is clearly wrong).
- **Done when:** it produces a full per-file report and an honest MAE. Record it forever
  as the reference number. (My quick version: MAE 24.3 — but on *unverified* labels, so
  re-measure on the M1 ground truth.)

### Milestone 3 — Robust feature + classical ML detector
- Features per time frame (1–5 ms hop): log-mel or band energies, spectral flux,
  spectral centroid/spread, ZCR, harmonic-to-noise ratio, local envelope stats,
  **clipping indicator**, and rolling noise-floor-relative level.
- Model: gradient-boosted trees or a small logistic regression / HMM. These are
  interpretable and data-efficient — important with ~30 files.
- Decode: per-frame click probability → adaptive peak-picking → integer count.
- **Done when:** it beats the M2 baseline on the M1 ground truth under the M4 protocol, and
  a per-condition error breakdown is written down.

### Milestone 4 — Learned detector (only if M3 plateaus)
- 1-D CNN / small CRNN over log-mel, trained as onset detection + counting.
- **Data augmentation is the whole game here** (see §5): gain, time shift, speed/pitch
  shift, SpecAugment, and **mixing real pen clicks into the provided noise files**
  (`background_street_noise`, `random_tv_noise`) at controlled SNR. If M1 yields even one
  verified isolated click, synthetic training pairs become possible and largely solve the
  30-file starvation problem.
- **Done when:** it beats M3 *on the held-out-condition split*, not just on aggregate.

### Milestone 5 — Calibration, uncertainty, and the app
- Predict a count **plus** a calibrated uncertainty (conformal prediction or a
  count-interval from the bootstrap). A count app that says "73 ± 4" is far more useful and
  more honest than one that always says a bare integer.
- Ship the CLI + optional web/mobile wrapper. Single-file input → single integer +
  interval + debug spectrogram.
- Log every inference (input hash, count, confidence, conditions) so real-world failures
  become new labeled data — the cheapest path to closing the long tail.

---

## 4. Statistical methodology and anti-overfitting guards

This is the part that decides whether the result is real.

**Primary metric:** MAE in clicks. Secondary: exact-match rate, within-±1 rate, and
Pearson/Spearman correlation with truth. Report all four — MAE alone can hide a
saturating model that always says 75.

**The single most important design decision: how you split the data.**
- `GroupKFold` by **recording file** at minimum. Never let two windows from the same
  recording land in different folds. With few files this is the #1 leakage source.
- Better: group by **capture session / condition** (`mic × distance × noise type`), so a
  fold boundary corresponds to a genuinely new recording situation.
- **Honest generalization test:** hold out an entire condition — e.g. train on
  `laptop_mic`, test on `phone_s25_mic` (and vice versa). This is the closest proxy for
  "user records in a new room with a new phone." Report this number prominently; it will
  be much worse than the aggregate CV number, and that gap is the honest expectation.
- The test split is **touched once, at the end**. Pre-register it in the manifest.

**Guard against fitting the benchmark:**
- **Frozen baseline rule:** if the M2 DSP detector is statistically indistinguishable from
  the M3/M4 model, ship the DSP detector. Simplicity is a feature.
- **Label-permutation test:** shuffle labels within folds and confirm accuracy collapses to
  chance. If it doesn't, there is leakage in the pipeline.
- **Near-duplicate detection** across the whole corpus before splitting (perceptual hash
  on log-mel or envelope correlation). If files are segments of one take, the CV number is
  fiction.
- **Normalization statistics computed on train folds only**, never globally.
- **Documented, small hyperparameter budget.** Write down the search space and the number
  of trials *before* running it. An unlogged 500-config search on 30 files is guaranteed
  overfitting even if every fold is clean.
- **Learning curves** (train on 25 %, 50 %, 75 %, 100 % of files). A curve that has
  flattened says the ceiling is data, not modeling — that tells you to stop tuning and go
  record more.
- **Bootstrap confidence intervals** on MAE; a 2-click MAE improvement that is inside the
  CI is noise.
- **Seed everything** and re-run the full pipeline under ≥3 seeds. Report the spread.

**Sanity / invariance checks (these catch more real bugs than any metric):**
- **Gain invariance:** scale input by 0.1× and 3× → count must not change materially.
- **Noise robustness:** mix in the street/TV noise at 0, 10, 20, 30 dB SNR → plot count
  vs SNR and find the breakdown point. This becomes a documented operating envelope.
- **Silence → 0.** All-zeros, and DC-only, must return 0.
- **Pure noise → ~0.** Feed the noise files with no pen: the false-positive rate must be
  near zero. This is a direct, label-free test that uses data we already have.
- **Monotonicity / count-vs-duration:** for a fixed click rate, doubling duration should
  roughly double the count; across files, count should scale with duration after
  controlling for rate.
- **Rate-plausibility guard:** the pen physically cannot click faster than some rate. A
  decoded stream with a 40 ms median inter-onset gap should trigger a warning, not a
  confident number. (The current baseline does exactly this and is wrong.)
- **Consistency:** two files of the same take counted by the forward and backward path
  should agree.
- **Unit/contract tests:** known synthetic signals (N impulses at known times) must return
  exactly N. Cheap, and they catch refactors.

**Regression discipline:** commit the metric value on the frozen test split alongside each
model version, so regressions are visible rather than discovered by users.

---

## 5. Contingencies and pitfalls

**Data and labels**
1. **Labels may be wrong** (highest risk, already evidenced). Mitigation: M1 before any
   modeling; two annotators; write the label spec down.
2. **Too few files.** ~30 files is very few for a learned model, and they are *not*
   independent (one pen, one room, one annotator). Mitigation: augmentation + synthetic
   click/noise mixing; strong DSP prior; prefer M3 over M4 unless M4 clearly wins.
3. **Segment files treated as independent examples** → silent, catastrophic leakage.
   Mitigation: the M1 near-duplicate audit + session grouping.
4. **Class imbalance across counts.** The corpus is heavy at 75 and 1–10. A model can
   score well by predicting "75" most of the time. Mitigation: report per-count-bucket
   error; macro-average across buckets; consider stratified folds.
5. **The 1-to-75 segments may not be uniform.** If they are one take, count correlates
   almost perfectly with time offset within the file — a model could learn "position in
   file" and get a great CV score for the wrong reason. Test by shuffling within-file
   position and confirming the model does not collapse.

**Signal / physics**
6. **Clipping destroys amplitude information.** Confirmed present. Mitigation: detect and
   flag clipping; lean on spectral shape over level; consider features invariant to gain.
7. **Mechanical self-noise is confounded with clicks** (the §1 evidence). Mitigation: this
   is the central research question — spectro-temporal click *shape* vs ambient energy
   distribution; consider a matched-filter / template approach once real onsets are known.
8. **Direction ambiguity.** Backwards clicks may sound like forwards clicks. If M2 question
   2 says direction matters, add a direction head and a labeled forward/backward set.
9. **Reverberation and room variation** smear onsets, which biases counts *downward* via
   merged events. Mitigation: report under/over-count asymmetry separately; it is
   diagnostic.
10. **Very quiet or very close recordings** can push clicks below the noise floor or into
    saturation. Mitigation: the SNR sweep in §4 finds the real operating envelope;
    document it as a product limitation rather than hiding it.

**Engineering / delivery**
11. **No toolchain in this environment** (no numpy/scipy/pip/ffmpeg; `audioop` gone in
    3.14). Fix in M0; do not let it leak into later milestones.
12. **`.m4a`/AAC support is currently impossible here** and the phone files use a different
    sample rate. Either fix in M0 or explicitly scope the app to WAV.
13. **Two "devices"** (laptop mic, phone mic) with different capture chains. Normalize
    carefully; keep the cross-device split as the headline metric.
14. **Model size / on-device latency** if shipping to a phone. Favors M2/M3 over M4.
15. **Silent deployment drift** — users record in conditions we never saw. Mitigation: the
    inference log from M5, and a confidence threshold below which the app declines to
    answer rather than guessing.

**Scope**
16. **Scope creep into a general audio event counter.** Anchor on the 0–75 dose-pen domain
    and say so in the README. If it should generalize, that is a different, larger project
    and it should be decided now, not discovered in M4.

---

## 6. Immediate next step

Answer §2 (especially questions 1 and 2). Then I will:

1. Stand up the M0 environment + unified loader.
2. Build the M1 annotation tool and hand you the first file to label, so we establish real
   ground truth before any modeling work.

Two things worth doing in parallel, since only you can do them: **listen to
`full_backward_2_click_track.wav` and tell me how many clicks you hear**, and **confirm
whether `1_to_10` … `71_to_75` are segments of one continuous take.** Those two answers
will sharpen this plan more than any modeling choice I make in the meantime.
