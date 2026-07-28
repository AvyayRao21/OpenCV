# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [1.0.0] — 2026-07-27

First public release. Collects the hand and colour experiments I had been
writing since May 2025, and adds the facial expression recogniser.

### Added

- **Facial expression recognition.** A convolutional network trained from
  scratch on FER-2013, reaching 66.7% on the held-out test set. No pretrained
  weights; trained entirely on my own laptop, on CPU, in 42 minutes.
  - `live.py` — webcam demo with MediaPipe face detection, temporal smoothing
    so labels do not flicker, a confidence bar for all seven classes, and a
    preview of exactly what the model receives.
  - `train.py` — trains from the FER-2013 folders and exports to TFLite with a
    metadata sidecar. Has `--quick` for proving the pipeline in seconds and
    `--resume` for continuing after an interruption.
  - `verify_tflite.py` — re-scores the exported model on the full test set so
    quantisation cannot silently cost accuracy.
  - The trained model is included at 357 KB and runs a face in 0.64 ms.
- **Calibrated face cropping.** The crop that gets fed to the model is derived
  from a measurement rather than guessed: MediaPipe's detection box covers
  88.8% of a FER-2013 crop, so expanding it by 1.126× matches the training
  framing. Worth 2.8 points of accuracy over the guess it replaced.
- **A live crop tuner** (`[`, `]`, `r`, `w`) with an on-screen preview and real
  FER-2013 faces for comparison, because no single crop setting is best for
  every expression — `sad` and `disgust` both favour a tighter crop than
  overall accuracy does.
- **Hand tracking programs**: finger counting two ways (HSV mask and MediaPipe
  landmarks), volume control by pinch distance, air drawing with and without
  pen-up, pinch-to-drag, and virtual keyboards for one and two hands.
- **Colour detection**: dominant colour naming, and an Arduino version that
  drives an RGB LED over serial to match what the camera sees.
- **HSV tuner** for finding skin thresholds, which is what made the mask-based
  finger counter workable.

### Notes

- Smoothing in the live demo is expressed as a time constant rather than a
  per-frame factor, so it behaves identically whether your webcam manages
  15 fps or 120.
- Contrast normalisation (CLAHE, histogram equalisation) and flip test-time
  augmentation were all implemented, measured, and dropped — the first two cost
  accuracy and the third gained 0.14 points for double the work. Recorded here
  so nobody wastes time re-trying them.

[Unreleased]: https://github.com/AvyayRao21/OpenCV/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/AvyayRao21/OpenCV/releases/tag/v1.0.0
