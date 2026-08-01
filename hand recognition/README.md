# Hand tracking

Nine programs that track hands and fingers through a webcam, and do something
useful with the result. These were the first projects in this collection.

All of them except one use [MediaPipe](https://developers.google.com/mediapipe)
to find 21 landmarks on each hand — knuckles, joints and fingertips — and then
work with those coordinates. The exception is deliberate, and worth explaining.

## The two counting programs

If you read nothing else here, read these two side by side. They solve exactly
the same problem — count how many fingers are held up — with completely
different approaches.

**`counting - hsv mask.py`** takes the old computer vision route. It converts
the frame to HSV colour space, thresholds it to keep only skin-coloured pixels,
cleans that mask up with morphological operations, finds the outline of your
hand, then counts the gaps between fingers using convexity defects. No machine
learning at all — just geometry.

It works. It is also fragile in a way that teaches you something: change the
lighting, the background, or whose hand it is, and the thresholds stop fitting.
That fragility is exactly why `hsv tuner.py` exists.

**`counting - mediapipe.py`** does the same job by asking a landmark model
where the fingers are, then comparing each fingertip against the joint below
it. If the tip sits higher than the joint, the finger is up.

It is a third of the length, and it does not care about your lighting or your
background.

| | Program | What it does | Quit |
| --- | --- | --- | --- |
| 1 | `hsv tuner.py` | Six sliders for the HSV range that isolates your hand. Aim for your hand pure white against a pure black background in the mask window. Prints the values when you quit, ready to paste into the program below. | `q` |
| 2 | `counting - hsv mask.py` | Hold your hand inside the green box. Counts fingers from the hand's outline. | `q` |
| 3 | `counting - mediapipe.py` | The same, anywhere in frame, using landmarks. | `Esc` |

## Interaction

Once you can track a hand reliably, you can drive things with it.

| | Program | What it does | Keys |
| --- | --- | --- | --- |
| 4 | `volume control.py` | Pinch your thumb and index finger together and apart to set the system volume. Distance between the tips maps to a level, with a bar showing where you are. | `q` |
| 5 | `air drawing.py` | Draws a line following your index fingertip. The pen is always down, so it draws whenever your hand is visible. | `c` clear, `q` |
| 6 | `air drawing - pinch to draw.py` | The same, but the pen only draws while your thumb and index finger are touching. Solves the obvious problem with the version above. | `c` clear, `q` |
| 7 | `drag and drop.py` | Pinch near the orange circle to grab it, move your hand to drag, release to drop. It tracks where you grabbed it so the circle does not jump to your fingers. | `q` |

> **`volume control.py` changes your real system volume** while it runs, and
> keeps doing so as long as your hand is in frame. It is Windows only — it uses
> `pycaw`, which needs `pip install pycaw comtypes`.

## Virtual keyboards

The most involved programs here. Both draw a full keyboard over the video and
let you type on it.

| | Program | What it does | Quit |
| --- | --- | --- | --- |
| 8 | `virtual keyboard - one hand.py` | Hover a key with your index finger and pinch to press it. Open palm types a space, an extended pinky is backspace. | `Esc` |
| 9 | `virtual keyboard - two hands.py` | The same keyboard tracking both hands at once, so either hand can type and both are highlighted independently. | `Esc` |

Both have a one-second debounce after each keypress, which is what stops a
single pinch registering thirty times at 30 fps. Typed text appears in a bar
across the top and is printed to the console.

These type **into the program's own window only**. They do not send keystrokes
to other applications, so there is no chance of accidentally typing into
something else on your desktop.

## Running any of them

```bash
pip install opencv-python mediapipe numpy
pip install pycaw comtypes          # only for volume control

python "hand recognition/counting - mediapipe.py"
```

Python 3.11. Nothing takes arguments.

## Notes if you are reading the code

- Frames are mirrored with `cv2.flip` so moving left on screen matches moving
  left in real life. Without it, using any of these feels wrong immediately.
- Pinch detection is just the pixel distance between two landmarks against a
  threshold. It works well, but the threshold is resolution-dependent — the
  same number means different things at 720p and 1080p.
- The keyboards recompute their layout from the frame size every frame, so they
  scale with your camera rather than assuming a fixed resolution.
