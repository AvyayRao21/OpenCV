# OpenCV experiments

A growing collection of small computer vision programs I write to learn with.
Each one is self-contained, runs on a normal laptop webcam, and does something
you can see immediately. Everything runs locally — no cloud, no accounts, no
uploads.

Three groups so far: **hand tracking**, **colour detection**, and **facial
expression recognition** with a model I trained myself.

> Before pointing any of this at a person — especially the expression
> recogniser — please read [DISCLAIMER.md](DISCLAIMER.md). It explains what the
> model can and cannot tell you, and it is shorter than it looks.

## Requirements

- **Python 3.11.** Not 3.13 — MediaPipe and TensorFlow do not publish wheels
  for it yet.
- A webcam.
- Windows for the volume control program (it uses `pycaw`). Everything else is
  cross-platform.

```bash
pip install opencv-python mediapipe numpy
pip install pycaw comtypes      # only for volume control
pip install pyserial            # only for the Arduino colour program
pip install tensorflow          # only to train or run the expression model
```

Every program runs the same way — no arguments, no setup:

```bash
python "hand recognition/finger counting - mediapipe.py"
```

Press **q** to quit any of them (a few use **Esc** — noted in the table).

## Hand tracking

| Program | What it does | Keys |
| --- | --- | --- |
| `hsv tuner.py` | Slider tool for finding the HSV range that isolates your skin from your background. I built it to make the program below work, and it prints the final values when you quit. | q |
| `finger counting - hsv mask.py` | Counts fingers held inside a box, using an HSV skin mask and convexity defects. No machine learning — pure geometry. Sensitive to lighting, which is exactly why the tuner exists. | q |
| `finger counting - mediapipe.py` | The same task using MediaPipe hand landmarks. Compares each fingertip against the joint below it. Far more robust than the mask version, and much shorter. | Esc |
| `volume control.py` | Pinch your thumb and index finger to set the system volume. Distance between the two tips maps to a volume level, with a bar showing where you are. **Changes your real volume.** | q |
| `air drawing.py` | Draws on screen following your index fingertip. The pen is always down. | c clear, q |
| `air drawing - pinch to draw.py` | The same, but pinching lowers the pen and releasing lifts it, so you can move without drawing. | c clear, q |
| `drag and drop.py` | Pinch to pick up an on-screen circle, move it, release to drop. Tracks the grab offset so the object does not jump when you grab it. | q |
| `virtual keyboard - one hand.py` | A full on-screen keyboard. Hover a key with your index finger and pinch to type. Open palm is space, extended pinky is backspace. | Esc |
| `virtual keyboard - two hands.py` | The same keyboard tracking both hands at once, so you can type with either. | Esc |

The two finger-counting programs are worth reading side by side. They solve an
identical problem, one with hand-tuned colour thresholds and one with a
landmark model, and the difference in robustness is the whole lesson.

## Colour detection

| Program | What it does | Keys |
| --- | --- | --- |
| `dominant color.py` | Names the most common colour in view from a set of HSV ranges. | q |
| `dominant color with arduino.py` | The same detection, but sends the colour over serial to an Arduino driving an RGB LED, so a real light matches what the camera sees. | q |

The Arduino sketch is in `color recognition/arduino/rgb led.ino`. It reads
`"r,g,b\n"` from serial and writes to pins 9, 10 and 11. Set your port at the
top of the Python file — it defaults to `COM3` — and note the program will not
start without a board connected.

## Facial expression recognition

The largest piece here. A convolutional network trained from scratch on
[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) to sort faces into
seven expressions, paired with MediaPipe face detection for a live webcam demo.

No pretrained weights are used anywhere. The model is trained entirely on my
own laptop, on CPU, in about 42 minutes.

| Program | What it does |
| --- | --- |
| `live.py` | The live demo. Detects faces, classifies each one, and smooths the result over time so labels do not flicker. |
| `train.py` | Trains the model from the FER-2013 image folders and exports it. |
| `verify_tflite.py` | Re-scores the exported model against the whole test set, so quantisation cannot quietly cost accuracy. |

### Running the demo

```bash
python "emotion recognition/live.py"
```

The trained model is included (`model/emotion.tflite`, 357 KB), so this works
straight away.

| Key | Does |
| --- | --- |
| `q` / `Esc` | Quit |
| `s` | Toggle temporal smoothing |
| `b` | Toggle the confidence bars |
| `p` | Toggle the preview of what the model actually sees |
| `[` `]` | Tighten / widen the face crop |
| `r` | Reset the crop to the calibrated default |
| `w` | Save your crop setting for next time |

### Training it yourself

Download FER-2013 and extract it so the folders sit next to the project as
`FER-2013 - data set - 1/train/<emotion>/` and `.../test/<emotion>/`.

```bash
python "emotion recognition/train.py" --quick    # ~15s, proves the pipeline works
python "emotion recognition/train.py"            # the real run, ~42 min on CPU
```

| Flag | Does |
| --- | --- |
| `--quick` | Trains on 300 images per class for 3 epochs. Meaningless accuracy; it exists so you find out the pipeline is broken in seconds rather than an hour. |
| `--epochs N` | Number of epochs. Default 60, or 3 with `--quick`. |
| `--batch-size N` | Default 128. |
| `--resume` | Continue an interrupted run from the last saved epoch. |

Training writes a checkpoint every epoch, so a crash or a power cut costs you
one epoch rather than the whole run.

### How well it works

**66.7% overall on the FER-2013 test set**, which the model never sees during
training. Human agreement with FER-2013's own labels is around 65%, so this is
a decent result — but it also means one prediction in three is wrong.

| Expression | Recall | | Expression | Recall |
| --- | ---: | --- | --- | ---: |
| happy | 86.6% | | angry | 59.9% |
| surprise | 84.2% | | sad | 53.7% |
| disgust | 70.3% | | fear | **37.0%** |
| neutral | 69.6% | | | |

`fear` is the weak point. It shares raised eyebrows with `surprise` and a
downturned mouth with `sad`, and at 48×48 pixels there is not much left to tell
them apart. I would rather publish that number than hide it behind the average.

### The interesting part: framing matters more than lighting

The thing I did not expect. I assumed uneven lighting would be the problem, so
I measured it — and the model turned out to be almost immune to brightness and
contrast changes. Every normalisation I tried made it *worse*.

What actually matters is how tightly the face is cropped:

| Crop scale | Overall | `disgust` |
| --- | ---: | ---: |
| 0.90 (tighter) | 63.2% | 74.8% |
| **1.00 (as trained)** | **66.7%** | 71.2% |
| 1.25 (wider) | 59.6% | 35.1% |
| 1.50 (wider) | 47.2% | **8.1%** |

`disgust` collapses from 71% to 8% purely from bad framing. So rather than
guess the right crop, I calibrated it: running MediaPipe's face detector on the
FER-2013 images themselves shows its box covers 88.8% of a FER-2013 crop, which
means expanding it by 1.126× reproduces the framing the model was trained on.
That is where the default comes from.

There is no single best value, though — overall accuracy peaks at one setting
while `sad` and `disgust` both prefer a tighter crop. That is why you can
adjust it live with `[` and `]`, with a preview of the exact image being fed to
the model and real FER-2013 faces beside it to match against.

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE.md) — free for personal, educational and
other noncommercial use. Not for selling or commercial use.

See [CONTRIBUTING.md](CONTRIBUTING.md) if you want to suggest something, and
[CHANGELOG.md](CHANGELOG.md) for what has changed.
