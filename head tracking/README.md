# Head-tracking parallax window

A real-time head-coupled perspective renderer. The webcam works out where your
eye is in physical space, and the screen stops being a flat picture and becomes
a window into a box that extends behind the glass.

The effect goes by several names: head-coupled perspective, fish-tank VR, motion
parallax display, off-axis or asymmetric frustum projection, and popularly the
Johnny Chung Lee Wii-remote demo.

Everything runs locally. No cloud, no network, no accounts.

Built in 2026 through spec-driven, AI-assisted development: a written
specification, eight stages each verified before the next began, the
implementation generated with an AI coding assistant, and every camera-facing
stage tested and tuned on my own screen and webcam. See
[How this was built](../README.md#how-this-was-built).

---

## The one idea that makes it work

This is **not** an orbiting camera. If you rotate a camera around a scene as the
viewer moves, you get something that looks superficially similar in a still
frame and is completely wrong in motion.

Instead, the virtual camera's view direction stays **fixed perpendicular to the
screen** at all times. Only its *position* follows your eye. The projection is
then sheared into an **asymmetric frustum** whose four side planes pass exactly
through the four physical corners of the window on your desk.

The consequence worth understanding: **the screen corners are nailed to the
world.** Whatever your head does, the four corners of the window always land on
the four corners of the rendered image, because the frustum is defined by them.
Everything else moves relative to that fixed frame — which is exactly what
looking through a real window is like.

This is Robert Kooima's generalized perspective projection. It arrives in `mouse parallax.py`, and the automated test that proves it is the corner-lock test: for 200
random eye positions, a vertex at a screen corner must land on the same pixel
every time. It does, to within `1.4e-12` pixels. The same test is run against
the ordinary camera and is required to *fail* — otherwise it would prove
nothing.

---

## Setting it up — do these in order

**The program to run is `scenes.py`.** Everything else is either a setup step
you run once, or an earlier stage of the build kept around because it is worth
seeing — the files are named for what they each do, and this is the order to use
them in.

### 1. Check the stack

```bash
python "environment check.py"
```

Prints one PASS/FAIL line per item: Python version, every library, the landmark
model, an OpenGL 4.6 context, and the webcam. It opens the camera for about two
seconds to confirm the iris landmarks come through, and saves nothing. If any
line says FAIL, fix that before going further — everything downstream assumes
these all pass.

### 2. Calibrate to your screen and camera

```bash
python "calibration.py"
```

**This is not optional.** The illusion is built from physical measurements, and
without them the scene still moves — it just moves by the wrong amount, which
reads as "nearly right" and is far harder to diagnose later. It walks you
through three things and writes them into `config.yaml`. Details in
[Calibration](#calibration) below.

Re-run just the camera part any time with `python "calibration.py" --focal`, and
check what is currently stored with `python "calibration.py" --show`.

### 3. Check the signs

```bash
python "eye position in millimeters.py"
```

Lean right — `X` must go positive. Sit up — `Y` must go positive. Lean back —
`Z` must grow. If any is backwards, press `x` or `y` to flip that axis and `w`
to save it. A sign error is invisible in the code and completely obvious in the
finished illusion, where the scene slides the wrong way.

### 4. Tune the smoothing

```bash
python "one euro filter.py"
```

Hold still and lower `min_cutoff` with `[` until the jitter figure settles. Then
move your head and raise `beta` with `=` until the filtered trace stops trailing
the raw one. Press `w` to save. The defaults are a reasonable starting point,
not a tuned answer.

### 5. Run it

```bash
python "scenes.py"
```

---

## The programs, in the order they were written

Each of these is a finished, runnable program, left working and frozen — the
next one started as a copy of it. Run any two side by side and you can see
exactly what changed.

| Order | File | What it adds | Camera? |
| --- | --- | --- | --- |
| 1 | `environment check.py` | Verifies the whole stack | Yes |
| 2 | `calibration.py` | Panel size, webcam offset, lens focal length | Yes |
| 3 | `eye position in millimeters.py` | Webcam pixels to millimetres, on its own thread | Yes |
| 4 | `one euro filter.py` | Adaptive smoothing: jitter gone, no added lag | Yes |
| 5 | `grid room.py` | The render path, with an **ordinary camera** | No |
| 6 | `mouse parallax.py` | The off-axis projection, driven by the mouse | No |
| 7 | `head tracked parallax.py` | Tracker plus projection, and a debug HUD | Yes |
| 8 | **`scenes.py`** | Four scenes, a vignette, smoothing between samples — **run this one** | Yes |

**`grid room.py` is deliberately wrong, and it is the most useful one to run.** It uses
an ordinary symmetric frustum. Move the eye sideways and the whole room slides
across the window as a rigid picture — the walls do not open, near objects do
not sweep across far ones. That is all a normal 3D camera can do. From `mouse parallax.py`
onward, press `TAB` to flip between the two live: same scene, same eye position,
and only one of them has depth.

**`mouse parallax.py` uses the mouse rather than the webcam, on purpose.** The projection
maths and the head tracker can each be wrong, and debugged together a sign error
and a maths error look identical. Proving the projection against a perfect,
noiseless, zero-lag input first means that when the tracker is connected in
`head tracked parallax.py`, anything that misbehaves is the tracker.

---

## Controls

In `scenes.py`, and mostly in the three programs before it too:

| Key | Action |
| --- | --- |
| `1` `2` `3` `4` | The Shaft / Grid Room / Depth Cards / The Aquarium |
| `TAB` | Off-axis projection vs. the ordinary one |
| `D` | Debug HUD — eye position, latency, frame rates, live frustum extents |
| `M` | Hand control to the mouse and back |
| `A` | Hands-off sweep, for recording |
| `V` | Vignette on/off |
| `F` | Fullscreen |
| `R` | Recentre |
| `Q` | Quit |

### Useful flags

```bash
python "scenes.py" --eye-source mouse   # drive it by hand
python "scenes.py" --render-test        # every scene to PNG, no window
python "scenes.py" --selftest           # check the projection maths
python "mouse parallax.py" --selftest   # 14 assertions
```

---

## Calibration

The illusion is only as accurate as its physical measurements, and there is no
way around measuring them. `calibration.py` walks through three things:

1. **The panel** — the width and height of the glass in millimetres. Measure the
   lit area, not the bezel. It also detects your screen's native pixel count, so
   that a floating window can work out where its own corners are in physical
   space.
2. **The webcam** — how far above the top of the glass the lens sits. Leaving
   this out puts a permanent vertical bias into the illusion. It is subtle and
   it ruins the effect.
3. **The focal length** — the one number that cannot be measured with a ruler,
   so it is solved instead. Sit at a distance you can measure, hold still, and
   the known pupil separation gives it up.

Everything lands in `config.yaml`, which is written back **in place** — comments
and any hand edits survive.

### Things worth knowing about the numbers

- **Interpupillary distance** defaults to 63 mm, the adult average. It scales the
  depth estimate linearly, so if the sense of scale feels wrong, that is the
  first value to revisit.
- **Focal length is in pixels, so it depends on the capture resolution.** If you
  change `camera.width` or `camera.height`, recalibrate.
- **Head turn is corrected for.** Turning your head squashes the pupil separation
  the camera sees, by exactly `cos(yaw)`; uncorrected, every glance sideways
  reads as a lurch backwards. The correction uses the 3D distance between the
  irises rather than a pose solve — the ratio of the 3D separation to the flat
  2D one *is* `1/cos(yaw)`, and because cosine is an even function the sign
  convention of the depth axis cannot introduce an error.

### Camera backends

`camera.backend` matters more than it looks. Measured on the development
machine at 1280x720:

| Backend | Delivered |
| --- | ---: |
| DirectShow | **10 fps** — only offers an uncompressed stream, and ignores a request for MJPG |
| Media Foundation | **30.3 fps** |

10 fps means 100 ms between head updates, which is enough on its own to wreck
the illusion. `msmf` is the default for that reason. `dshow` behaves better with
some external USB webcams — if you switch, check the delivered rate in the debug
HUD rather than assuming.

One trap if you modify the capture code: **Media Foundation will not change
resolution once the stream has started.** It goes on reporting the size you
asked for while returning nothing but bad frames. Set the format before reading
the first frame.

---

## Smoothing

Raw landmark output jitters by a few pixels even when you hold perfectly still,
and a few pixels at arm's length is a few millimetres of head movement, which
the renderer faithfully turns into a shivering scene.

The obvious fix is to average over recent frames, and the obvious fix is wrong.
Averaging trades jitter for lag, and in this illusion lag is much the worse of
the two: a slightly shivering scene still reads as solid, but one that arrives a
fifth of a second behind your head reads as sloshing about, and the sense of
depth collapses entirely.

So this uses a **One Euro filter**, which makes the cutoff frequency depend on
how fast the signal is currently moving — filter hard when still, because
anything moving must be noise; barely filter at all when moving fast, because
the lag would cost more than the noise does. Run `one euro filter.py`
to tune it against a live graph of raw versus filtered.

Two notes on the implementation:

- The blend factor is `1 - exp(-dt/tau)`, the exact discretisation, rather than
  the `1/(1 + tau/dt)` given in the original paper. They agree to first order,
  but on a 200 mm/s ramp measured across 15 to 240 fps, the paper's form drifts
  by 6.25 mm and this one by 1.09 mm once the unavoidable half-frame sampling
  delay is accounted for. Same filter, better arithmetic.
- The renderer draws far faster than the camera delivers, so between tracker
  samples the eye is carried forward along its own filtered velocity. Without
  that, each reading is held for several frames and the motion moves in visible
  30 Hz steps even though every frame arrives on time — smooth frames, stepped
  movement, which reads as stutter.

---

## What it actually looks like — read this before you are disappointed

**The effect is dramatically more convincing through a camera than it is to your
own eyes, and that is not a flaw in the implementation.**

Two eyes looking at a flat panel receive stereo disparity that says,
unambiguously and continuously, "this is a flat surface half a metre away".
Motion parallax says "this is a deep box". The two cues conflict, and at desk
distance stereo generally wins. Closing one eye removes stereo but leaves the
focus distance of your eye still reporting a flat screen.

This is why every impressive recording of this effect looks better than the
thing itself: a camera has one viewpoint and no focus cue to argue back, so it
receives the illusion cleanly. If you want to see it properly, hold a phone
beside your head, parallel to your face, and move both together — through the
phone screen it is completely convincing.

To film it, mount the recording camera at head height right next to your face
and move both together, or let the phone be the viewer. A video shot from a
fixed tripod while you move your head shows a scene wobbling for no reason and
looks broken. `--eye-source mouse` exists for clean hand-driven recordings.

The scenes differ in how well they survive this. **The Shaft** is built entirely
from solid surfaces that occlude one another, and occlusion is the strongest
depth cue human vision has — it is the only monocular cue powerful enough to
argue with stereo at all. The wireframe **Grid Room** offers linear perspective
and very little else.

---

## Coordinate system

One frame of reference, in millimetres, obeyed everywhere:

```
        +Y  up
         |
         |
         +--------- +X  right, from where you sit
        /
       /
     +Z  out of the screen, towards you
```

- The origin is the **geometric centre of the display panel**.
- The screen plane is exactly `Z = 0`. Your eye is always at positive Z,
  typically 400–700 mm.
- Anything that should appear inside the box lives at **negative Z**.

The webcam is *not* at the origin — it sits above the top bezel — so every
position measured in camera space is translated by `cam_offset_mm` before use.

**In windowed mode the frustum's screen rectangle is the window, not the
panel.** The corners are derived from where the window sits on the desktop,
converted through a pixels-per-millimetre scale, and recomputed when it is moved
or resized. Fullscreen is simply the case where the two rectangles coincide;
there is no separate code path for it.

---

## Configuration

`config.yaml` holds everything, and is editable without touching code. The
values that matter most:

| Key | What it does |
| --- | --- |
| `display.screen_w_mm` / `screen_h_mm` | Panel size. Measure the glass. |
| `display.native_px` | Real pixel count, not the DPI-scaled desktop size |
| `display.cam_offset_mm` | Where the lens is, relative to the panel centre |
| `tracking.ipd_mm` | Your interpupillary distance |
| `tracking.focal_px` | Solved by calibration — never guess it |
| `tracking.flip_x` / `flip_y` | Fix an inverted axis without touching code |
| `tracking.predict_ms` | Forward extrapolation to hide pipeline latency |
| `filter.min_cutoff` / `beta` | Steadiness when still / responsiveness when moving |
| `render.msaa` | Anti-aliasing. Aliased receding lines cheapen the effect. |
| `render.vignette` | Border shading, so the bezel reads as a window casing |
| `render.max_fps` | Cap on the render loop |

If leaning right moves the scene the wrong way, flip `flip_x`. Sign errors are
invisible in the code and completely obvious on screen.

---

## Requirements

Python 3.11 (MediaPipe and the rest of this project have no 3.13 wheels).

```
opencv-python  mediapipe  numpy  moderngl  moderngl-window  pyglet  PyYAML
```

Plus `models/face_landmarker.task` — MediaPipe's Face Landmarker, Apache 2.0,
from Google. It returns 478 landmarks with iris refinement on, and the two iris
points are what make monocular depth possible at all.

Performance on the development machine — an Intel integrated GPU, not a discrete
one:

| | |
| --- | ---: |
| Render | ~230 fps at 8x MSAA, p99 frame time 5.7 ms |
| Tracker | 30 fps, ~12 ms inference |
| Eye position age at draw time | ~35 ms |

That last figure covers the camera frame arriving to the frame being drawn. It
excludes the camera's own exposure and the display's response, neither of which
a program can see.
