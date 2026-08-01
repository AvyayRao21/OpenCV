# Changelog

Notable changes to this collection.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each new project area arrives as a minor version. Fixes and improvements to
existing ones are patches.

## [Unreleased]

Planned: augmented reality experiments, and depth and perspective from a single
webcam. See the roadmap in the README.

## [1.0.0] — 2026-07-27

First public release, collecting the experiments I had been writing since May
2025 plus the expression recogniser built on top of them.

### Hand tracking

- Finger counting two ways — one using HSV colour thresholds and convexity
  defects, one using MediaPipe landmarks — kept side by side deliberately,
  because the contrast between them is the most useful thing in the folder.
- `hsv tuner.py`, six sliders for finding the skin thresholds the mask-based
  counter depends on.
- Gesture volume control, mapping thumb-to-index distance onto system volume.
- Air drawing, with and without pinch-to-lower-the-pen.
- Pinch to drag an on-screen object, preserving the grab offset so it does not
  jump to your fingers.
- Virtual keyboards for one hand and for two, with a debounce so a single pinch
  does not register thirty times a second.

### Colour detection

- Dominant colour naming from HSV ranges, including the two-range handling red
  needs because hue wraps around.
- An Arduino version that sends the detected colour over serial to drive an RGB
  LED, with the sketch and wiring included.

### Facial expression recognition

- A convolutional network trained from scratch on FER-2013, reaching 66.7% on
  the held-out test set. No pretrained weights; trained on CPU in 42 minutes.
- Live webcam demo with MediaPipe face detection, temporal smoothing so labels
  do not flicker, a confidence bar for all seven expressions, and a preview of
  exactly what the model receives.
- Training script with `--quick` for proving the pipeline in seconds, and
  `--resume` for continuing after an interruption without losing the best model.
- `verify_tflite.py`, which re-scores the exported model on the full test set so
  quantisation cannot silently cost accuracy.
- The trained model is included at 357 KB and classifies a face in 0.64 ms.
- The face crop is calibrated rather than guessed: MediaPipe's detection box
  covers 88.8% of a FER-2013 crop, so expanding it by 1.126× matches the
  training framing. Worth 2.8 percentage points over the guess it replaced.
- A live crop tuner, because no single crop is best for every expression —
  `sad` and `disgust` both favour a tighter one than overall accuracy does.

### Notes

- Smoothing in the live demo is a time constant rather than a per-frame amount,
  so it behaves the same at any frame rate.
- Contrast normalisation and flip test-time augmentation were both built,
  measured and removed — the first cost accuracy, the second gained 0.14 of a
  point for twice the work. Recorded here so nobody repeats the experiment.

[Unreleased]: https://github.com/AvyayRao21/OpenCV/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/AvyayRao21/OpenCV/releases/tag/v1.0.0
