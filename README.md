# Computer Vision Projects

A growing collection of small programs I build with OpenCV to learn computer
vision properly — by making things that work rather than by reading about them.

Every program is self-contained, runs on an ordinary laptop webcam, and does
something you can see the moment you start it. Everything runs locally: no
cloud services, no accounts, no uploads, no data leaving your machine.

```bash
python "hand recognition/counting - mediapipe.py"
```

That is the whole setup. Every program in here starts the same way.

## Contents

| | |
| --- | --- |
| [Quick start](#quick-start) | What you need and how to run anything here |
| [Hand tracking](#hand-tracking) | Finger counting, air drawing, virtual keyboards, gesture volume control |
| [Colour detection](#colour-detection) | Naming colours in view, and driving an Arduino LED with them |
| [Faces and expressions](#faces-and-expressions) | A CNN trained from scratch to sort facial expressions |
| [Head-tracking parallax window](#head-tracking-parallax-window) | The screen becomes a window into a box behind the glass |
| [What is coming next](#what-is-coming-next) | Where this collection is heading |
| [How this repository is organised](#how-this-repository-is-organised) | The pattern every project follows |
| [Honest limits](#honest-limits) | What these programs are and are not |

## Quick start

You need **Python 3.11**. Not 3.13 — MediaPipe and TensorFlow do not publish
wheels for it yet, and nothing here will install.

```bash
pip install opencv-python mediapipe numpy
```

Three programs need something extra, and only those three:

| Extra | Needed for |
| --- | --- |
| `pip install pycaw comtypes` | Gesture volume control (Windows only) |
| `pip install pyserial` | The Arduino colour program |
| `pip install tensorflow` | Running or training the expression model |

Then run any program directly. None of them take arguments or need
configuration:

```bash
python "<folder>/<program>.py"
```

Press **q** to quit, or **Esc** in a few of them. Each project's README lists
the exact keys.

## Hand tracking

Hand and finger tracking, mostly built on MediaPipe's landmark model. These
were the first things I made here, and they still make the best starting point
if you are reading the code to learn from it.

| Program | What it does |
| --- | --- |
| `hsv tuner.py` | Slider tool for isolating your skin from the background |
| `counting - hsv mask.py` | Counts fingers using colour thresholds and geometry |
| `counting - mediapipe.py` | The same task using landmarks instead |
| `volume control.py` | Pinch to set the system volume |
| `air drawing.py` | Draw on screen with your fingertip |
| `air drawing - pinch to draw.py` | The same, but pinch to lower the pen |
| `drag and drop.py` | Pinch to pick up and move an on-screen object |
| `virtual keyboard - one hand.py` | Type on an on-screen keyboard by pinching |
| `virtual keyboard - two hands.py` | The same keyboard, both hands at once |

The two counting programs solve an identical problem two different ways — one
with hand-tuned colour thresholds, one with a landmark model. Reading them side
by side is the clearest lesson in this repository about why the second approach
won.

**→ [Full details, controls and notes](hand%20recognition/README.md)**

## Colour detection

Naming the dominant colour in view, and pushing that out to real hardware.

| Program | What it does |
| --- | --- |
| `dominant color.py` | Names the most common colour in frame |
| `dominant color with arduino.py` | Sends that colour to an Arduino driving an RGB LED |

The second one needs a board wired up; the sketch is included.

**→ [Full details, wiring and the Arduino sketch](color%20recognition/README.md)**

## Faces and expressions

A convolutional network I trained from scratch on the FER-2013 dataset, which
sorts a face into one of seven expressions, paired with MediaPipe face
detection for a live webcam demo.

No pretrained weights are used anywhere in this — the model is trained entirely
on my own laptop, on CPU, in about 42 minutes. It reaches **66.7%** on the
held-out test set, and the trained model is included so the demo runs
immediately.

| Program | What it does |
| --- | --- |
| `live.py` | The live webcam demo |
| `train.py` | Trains the model from the dataset |
| `verify_tflite.py` | Re-scores the exported model to catch accuracy loss |

The most interesting thing I found building it was that **how tightly you crop
the face matters far more than lighting does** — enough that one expression
drops from 71% to 8% accuracy under a bad crop. That is why the demo lets you
adjust the crop while it runs.

**→ [Full details, accuracy figures and how to train it](emotion%20recognition/README.md)**

> This one deserves a caveat the others do not. It recognises *expressions*, not
> feelings, and it is wrong about one time in three. Please read
> [DISCLAIMER.md](DISCLAIMER.md) before pointing it at anybody.

## Head-tracking parallax window

The biggest thing here. The webcam works out where my eye is in physical space,
and the screen stops being a flat picture and becomes a window into a box that
extends behind the glass. Move your head and you see round the sides of things,
as you would leaning across a real window frame.

The trick is not what most people expect. It is **not** a camera orbiting the
scene — that looks similar in a still frame and is obviously wrong in motion.
The virtual camera never turns at all; its direction stays perpendicular to the
screen and only its *position* follows my eye. What changes is the shape of the
view volume: the projection is sheared into an asymmetric frustum whose four
side planes always pass exactly through the four physical corners of my monitor.

The upshot is that the screen corners are nailed to the world. A test runs 200
random eye positions and checks that a point at a screen corner lands on the
same pixel every time — it does, to within a millionth of a pixel.

| Program | What it does |
| --- | --- |
| `environment check.py` | Verifies the stack before anything else |
| `calibration.py` | Measures the screen, the camera position and the lens |
| `eye position in millimeters.py` | Webcam pixels to a head position in millimetres |
| `scenes.py` | **The finished thing** — four scenes, and where to start |

Everything is measured in real millimetres, because it has to be: the screen is
measured with a ruler, the camera lens position with it, and the lens focal
length is solved by sitting a known distance away. Get those wrong and the scene
still moves, just by the wrong amount.

Eight programs in total, each left working and frozen with the next starting as
a copy, so you can run any two side by side and see what changed. The pair worth
comparing is `grid room.py`, which uses an ordinary camera and can only slide
the picture around, against `mouse parallax.py`, which shears the frustum
properly and makes the room open up.

**→ [Setup order, calibration and full details](head%20tracking/README.md)**

> An honest warning that the README covers properly: this effect is far more
> convincing through a camera than to your own eyes. Two eyes looking at a flat
> panel get stereo depth cues insisting the surface is flat, and at desk
> distance those usually win. Every impressive video of this effect online —
> including the famous ones — is filmed, and a camera has only one eye.

## What is coming next

This is a long-running collection, and the plan is to keep adding to it. What I
am working towards:

- **Augmented reality** — anchoring graphics to real objects and surfaces so
  they hold position as the camera moves.
- **Stereo output for the parallax window** — one frustum per eye instead of
  one for both. The maths already supports it, and it is the one change that
  could beat the flat-screen depth cue described above.
- More small experiments alongside those, in the same self-contained style.

Each will arrive as its own folder following the same pattern as the ones
already here, so the repository grows sideways rather than getting tangled.

## How this repository is organised

One folder per subject area. Each folder holds its programs plus a README
covering what each does, its controls, and anything you need to know before
running it.

```
.
├── hand recognition/       tracking hands and fingers
├── color recognition/      colour detection, plus Arduino output
├── emotion recognition/    expression classification and its model
└── <new areas land here as their own folders>
```

Programs are deliberately standalone. There is no shared framework to learn
before you can read one, and no program depends on another. Open any single
file and it makes sense on its own — that is the point, since these exist to be
read as much as run.

## Honest limits

These are learning projects, not products. They are not robust against unusual
input, not optimised, and not tested across every platform — I build on
Windows, so that path gets the most use.

Two of them reach outside their own window: the volume control changes your
actual system volume, and the Arduino program writes to a serial port. Both are
flagged in their own READMEs.

Full detail, including what the expression recogniser can and cannot tell you,
is in **[DISCLAIMER.md](DISCLAIMER.md)**. It is worth the two minutes.

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE.md) — free to use, study, modify and
share for personal, educational and other noncommercial purposes. Not for
commercial use or resale.

[CHANGELOG.md](CHANGELOG.md) tracks what has changed.
[CONTRIBUTING.md](CONTRIBUTING.md) covers issues and suggestions.
