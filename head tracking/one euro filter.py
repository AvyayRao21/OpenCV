"""
Step 3 of the parallax window: smoothing, without adding lag.

Step 2 gave a head position that is correct on average and visibly noisy frame
to frame. Landmarks wobble by a few pixels even when you hold perfectly still,
and a few pixels of wobble at arm's length is a few millimetres of head
movement, which the renderer will faithfully turn into a scene that shivers.

The obvious fix is to average over recent frames. The obvious fix is wrong.
Averaging trades jitter for lag, and in this particular illusion lag is the
worse of the two by a wide margin: a scene that shivers slightly still reads as
solid, but a scene that arrives a fifth of a second after your head has moved
reads as sloshing about behind you, and the sense of depth collapses.

So this uses a **One Euro filter** (Casiez, Roussel and Vogel, CHI 2012). The
idea is one line long: make the cutoff frequency depend on how fast the signal
is currently moving. Hold still and it filters hard, because anything moving
must be noise. Move fast and it barely filters at all, because the lag would
cost more than the noise does.

Run it, hold still, and watch the two traces on the graph — the dim one is raw,
the bright one is filtered. Then move your head quickly and watch them stay
together. If the bright trace lags visibly behind the dim one during fast
movement, beta is too low.

Usage:

    python "one euro filter.py"
    python "one euro filter.py" --selftest    check the maths, no camera

    q / ESC   quit                    1 / 2 / 3   graph X, Y or Z
    [ / ]     min_cutoff  down / up   p           prediction on / off
    - / =     beta        down / up   w           save the filter to config.yaml
    r         reset to the config     m           mirror the preview
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
MODEL_PATH = HERE / "models" / "face_landmarker.task"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

IRIS_LEFT = 468
IRIS_RIGHT = 473

HISTORY_SECONDS = 6.0


# ----------------------------------------------------------------------------
# The One Euro filter
# ----------------------------------------------------------------------------


def smoothing_alpha(cutoff_hz: float, dt: float) -> float:
    """Blend factor for a one-pole low pass, from a cutoff and a time step.

    This is where frame-rate independence comes from, and it is worth being
    explicit about because this project has been bitten by the alternative
    before. A fixed per-frame blend factor smooths over a fixed number of
    *frames*, so the same constant means something different on a machine
    running at 30 fps and one running at 120. Deriving the factor from the
    measured dt and a cutoff in hertz means the filter is specified in seconds,
    which is what a human actually cares about.

    Note this is `1 - exp(-dt/tau)`, the exact discretisation, and **not** the
    `1 / (1 + tau/dt)` given in the One Euro paper. The two agree to first order
    and the paper's form is fine at a steady frame rate, but it drifts when the
    rate changes. Measured on a 200 mm/s ramp across 15 to 240 fps, with the
    unavoidable half-frame sampling delay accounted for, the paper's form
    spreads by 6.25 mm and this one by 1.09 mm. Same filter, better arithmetic.
    """
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 - math.exp(-dt / tau)


class LowPass:
    """One-pole exponential low pass that remembers its last output."""

    def __init__(self) -> None:
        self.value: float | None = None

    def __call__(self, x: float, alpha: float) -> float:
        self.value = x if self.value is None else alpha * x + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class OneEuroFilter:
    """Adaptive low pass: cutoff rises with the speed of the signal.

        cutoff = min_cutoff + beta * |velocity|

    `min_cutoff` sets how hard it filters when you are still — lower is
    steadier and laggier. `beta` sets how quickly it gets out of the way when
    you move — higher means less lag and more jitter during motion.

    Tune min_cutoff first, holding still, until the jitter readout is
    acceptable. Then tune beta, moving, until the lag disappears.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = LowPass()
        self._dx = LowPass()
        self._last: float | None = None
        self.velocity = 0.0

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last = None
        self.velocity = 0.0

    def __call__(self, x: float, dt: float) -> float:
        if dt <= 0.0:
            # No time has passed, so there is nothing to update. Returning the
            # last output is the only sane answer; dividing by dt is not.
            return self._x.value if self._x.value is not None else x

        # The derivative is filtered too. An unfiltered derivative of a noisy
        # signal is almost pure noise, and feeding that into the cutoff would
        # make the filter flap open and shut on the jitter it is meant to
        # remove.
        raw_velocity = 0.0 if self._last is None else (x - self._last) / dt
        self._last = x
        self.velocity = self._dx(raw_velocity, smoothing_alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(self.velocity)
        return self._x(x, smoothing_alpha(cutoff, dt))


class OneEuroVec3:
    """Three independent One Euro filters, plus forward prediction.

    The axes are filtered separately, as the spec requires. They are not
    correlated in any way that would make a joint filter better, and keeping
    them independent means a fast sideways movement does not loosen the
    filtering on depth.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.filters = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(3)]

    def set_params(self, min_cutoff: float, beta: float) -> None:
        for f in self.filters:
            f.min_cutoff = min_cutoff
            f.beta = beta

    def reset(self) -> None:
        for f in self.filters:
            f.reset()

    def __call__(self, xyz: tuple[float, float, float], dt: float,
                 predict_s: float = 0.0) -> tuple[float, float, float]:
        out = [f(v, dt) for f, v in zip(self.filters, xyz)]
        if predict_s > 0.0:
            # Extrapolate along the filtered velocity to cover the pipeline
            # delay — the camera exposure, the inference, and the frame that is
            # already being drawn. Deliberately small: predicting far ahead
            # turns every direction change into an overshoot, which is a much
            # uglier artefact than the lag it removes.
            out = [v + f.velocity * predict_s for f, v in zip(self.filters, out)]
        return out[0], out[1], out[2]

    @property
    def speed(self) -> float:
        return math.sqrt(sum(f.velocity ** 2 for f in self.filters))


# ----------------------------------------------------------------------------
# Geometry, unchanged from step 2. Repeated so this file runs on its own.
# ----------------------------------------------------------------------------


def iris_separation_px(landmarks, frame_w: int, frame_h: int):
    """Pupil separation in pixels, and how far the head is turned.

    The 3D distance between the irises is the separation with the head-turn
    foreshortening already removed, so the ratio of 3D to flat 2D is exactly
    1/cos(yaw). Cosine being even means MediaPipe's z sign cannot bite us.
    """
    left, right = landmarks[IRIS_LEFT], landmarks[IRIS_RIGHT]
    dx = (left.x - right.x) * frame_w
    dy = (left.y - right.y) * frame_h
    dz = (left.z - right.z) * frame_w

    ipd_2d = math.hypot(dx, dy)
    ipd_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
    cos_yaw = min(1.0, max(-1.0, ipd_2d / ipd_3d if ipd_3d > 1e-6 else 1.0))

    mid_u = (left.x + right.x) * 0.5 * frame_w
    mid_v = (left.y + right.y) * 0.5 * frame_h
    return ipd_2d, ipd_3d, math.degrees(math.acos(cos_yaw)), mid_u, mid_v


def corrected_ipd_px(ipd_2d: float, ipd_3d: float, max_yaw_deg: float) -> float:
    max_gain = 1.0 / math.cos(math.radians(max_yaw_deg))
    gain = ipd_3d / ipd_2d if ipd_2d > 1e-6 else 1.0
    return ipd_2d * min(gain, max_gain)


def fallback_focal_px(frame_w: int, hfov_deg: float = 60.0) -> float:
    return (frame_w * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)


# ----------------------------------------------------------------------------


@dataclass
class EyeSample:
    """One reading. `x/y/z` are filtered and are what a consumer should use;
    `raw` is kept only so the two can be drawn against each other."""

    x: float = 0.0
    y: float = 0.0
    z: float = 600.0
    raw: tuple[float, float, float] = (0.0, 0.0, 600.0)
    valid: bool = False
    ipd_px: float = 0.0
    yaw_deg: float = 0.0
    speed: float = 0.0
    n_faces: int = 0
    timestamp: float = 0.0
    tracker_fps: float = 0.0
    inference_ms: float = 0.0
    preview: np.ndarray | None = field(default=None, repr=False)
    iris_px: tuple[tuple[float, float], tuple[float, float]] | None = None


class EyeTracker(threading.Thread):
    """Capture, inference, mm conversion and filtering, on its own thread."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(daemon=True, name="eye-tracker")
        self.cfg = cfg
        self._lock = threading.Lock()
        self._latest = EyeSample()
        self._history: deque = deque(maxlen=1200)
        self._running = threading.Event()
        self._running.set()

        d, t, f = cfg["display"], cfg["tracking"], cfg["filter"]
        self.cam_offset = [float(v) for v in d["cam_offset_mm"]]
        self.ipd_mm = float(t["ipd_mm"])
        self.max_yaw = float(t.get("max_yaw_deg", 35.0))
        self.z_min, self.z_max = (float(v) for v in t.get("z_clamp_mm", [200.0, 1500.0]))
        self.ease_seconds = float(t.get("lost_face_ease_s", 0.6))

        self.flip_x = bool(t.get("flip_x", False))
        self.flip_y = bool(t.get("flip_y", False))
        self.mirror_preview = True

        self.default_min_cutoff = float(f.get("min_cutoff", 1.0))
        self.default_beta = float(f.get("beta", 0.007))
        self.d_cutoff = float(f.get("d_cutoff", 1.0))
        self.min_cutoff = self.default_min_cutoff
        self.beta = self.default_beta

        self.default_predict_ms = float(t.get("predict_ms", 20.0))
        self.predict_ms = self.default_predict_ms
        self.predict_enabled = self.predict_ms > 0.0

        self.filter = OneEuroVec3(self.min_cutoff, self.beta, self.d_cutoff)

        self.focal_px: float | None = (
            float(t["focal_px"]) if t.get("focal_px") is not None else None
        )
        self.focal_is_estimated = self.focal_px is None

        self.frame_w = 0
        self.frame_h = 0
        self.backend = ""
        self.error: str | None = None
        self.default_position = (0.0, 0.0, 600.0)

    # -- the slot ------------------------------------------------------------

    @property
    def latest(self) -> EyeSample:
        with self._lock:
            return self._latest

    def history_snapshot(self) -> list:
        with self._lock:
            return list(self._history)

    def reset_filter(self) -> None:
        self.min_cutoff = self.default_min_cutoff
        self.beta = self.default_beta
        self.predict_ms = self.default_predict_ms
        self.predict_enabled = self.predict_ms > 0.0
        self.filter.reset()

    def stop(self) -> None:
        self._running.clear()

    # -- camera --------------------------------------------------------------

    def _open_camera(self):
        backends = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "auto": cv2.CAP_ANY}
        cam = self.cfg["camera"]
        preferred = str(cam.get("backend", "dshow")).lower()
        order = [preferred] + [b for b in ("dshow", "msmf", "auto") if b != preferred]
        index = int(cam.get("index", 0))

        for name in order:
            cap = cv2.VideoCapture(index, backends[name])
            if not cap.isOpened():
                cap.release()
                continue

            # Set the format BEFORE reading a single frame. Media Foundation
            # will not change resolution once the stream has started: it goes on
            # reporting the requested size while returning nothing but bad
            # frames. Measured — 0 good frames out of 40 with a probe read
            # first, 40 out of 40 without. DirectShow tolerates either order.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam["width"]))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam["height"]))
            # A small buffer matters: with a deep one the frame you read is
            # already old, which is latency you cannot get back.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            try:
                ok = cap.read()[0]
            except cv2.error:
                ok = False          # MSMF sometimes hands back a bad stride
            if ok:
                self.backend = name
                return cap
            cap.release()
        raise RuntimeError(f"could not open camera index {index}")

    @staticmethod
    def _face_size(landmarks, frame_w: int, frame_h: int) -> float:
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        return (max(xs) - min(xs)) * frame_w * (max(ys) - min(ys)) * frame_h

    def _choose_face(self, faces: list, frame_w: int, frame_h: int, previous):
        sizes = [self._face_size(f, frame_w, frame_h) for f in faces]
        largest = int(np.argmax(sizes))
        if previous is None or len(faces) == 1:
            return largest, sizes[largest]
        best, best_dist = 0, float("inf")
        for i, f in enumerate(faces):
            u = (f[IRIS_LEFT].x + f[IRIS_RIGHT].x) * 0.5 * frame_w
            v = (f[IRIS_LEFT].y + f[IRIS_RIGHT].y) * 0.5 * frame_h
            dist = math.hypot(u - previous[0], v - previous[1])
            if dist < best_dist:
                best, best_dist = i, dist
        if sizes[largest] > sizes[best] * 1.25:
            return largest, sizes[largest]
        return best, sizes[best]

    # -- the loop ------------------------------------------------------------

    def run(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        try:
            cap = self._open_camera()
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            return

        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.focal_px is None:
            self.focal_px = fallback_focal_px(self.frame_w)

        landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=2,
            )
        )

        cx, cy = self.frame_w * 0.5, self.frame_h * 0.5
        start = time.perf_counter()
        frame_times: list[float] = []
        previous_eye_px = None
        last_good = None
        lost_since = None
        last_time = None

        while self._running.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            now = time.perf_counter()
            dt = 0.0 if last_time is None else now - last_time
            last_time = now

            frame_times.append(now)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                   if len(frame_times) > 1 and frame_times[-1] > frame_times[0] else 0.0)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t0 = time.perf_counter()
            result = landmarker.detect_for_video(image, int((now - start) * 1000))
            inference_ms = (time.perf_counter() - t0) * 1000.0

            faces = result.face_landmarks
            if faces:
                idx, _ = self._choose_face(faces, self.frame_w, self.frame_h,
                                           previous_eye_px)
                lm = faces[idx]
                ipd_2d, ipd_3d, yaw, mid_u, mid_v = iris_separation_px(
                    lm, self.frame_w, self.frame_h)
                previous_eye_px = (mid_u, mid_v)
                ipd = corrected_ipd_px(ipd_2d, ipd_3d, self.max_yaw)

                z_cam = self.focal_px * self.ipd_mm / max(ipd, 1e-6)
                z_cam = min(self.z_max, max(self.z_min, z_cam))
                # u grows right in the raw (mirrored) image, v grows downward.
                x_cam = -(mid_u - cx) * z_cam / self.focal_px
                y_cam = -(mid_v - cy) * z_cam / self.focal_px
                if self.flip_x:
                    x_cam = -x_cam
                if self.flip_y:
                    y_cam = -y_cam

                raw = (x_cam + self.cam_offset[0],
                       y_cam + self.cam_offset[1],
                       z_cam + self.cam_offset[2])
                last_good = raw
                lost_since = None
                valid = True
                iris_px = ((lm[IRIS_LEFT].x * self.frame_w, lm[IRIS_LEFT].y * self.frame_h),
                           (lm[IRIS_RIGHT].x * self.frame_w, lm[IRIS_RIGHT].y * self.frame_h))
            else:
                if lost_since is None:
                    lost_since = now
                if last_good is None:
                    raw = self.default_position
                else:
                    t = min(1.0, (now - lost_since) / max(self.ease_seconds, 1e-3))
                    ease = t * t * (3.0 - 2.0 * t)
                    raw = tuple(a + (b - a) * ease
                                for a, b in zip(last_good, self.default_position))
                valid = False
                ipd = yaw = 0.0
                iris_px = None

            self.filter.set_params(self.min_cutoff, self.beta)
            predict_s = (self.predict_ms / 1000.0) if self.predict_enabled else 0.0
            filtered = self.filter(raw, dt, predict_s)

            sample = EyeSample(
                x=filtered[0], y=filtered[1], z=filtered[2], raw=raw, valid=valid,
                ipd_px=ipd, yaw_deg=yaw, speed=self.filter.speed,
                n_faces=len(faces), timestamp=now, tracker_fps=fps,
                inference_ms=inference_ms, preview=frame, iris_px=iris_px)

            with self._lock:
                self._latest = sample
                self._history.append((now, raw, filtered))
                while self._history and now - self._history[0][0] > HISTORY_SECONDS:
                    self._history.popleft()

        cap.release()


# ----------------------------------------------------------------------------
# Jitter measurement
# ----------------------------------------------------------------------------


def jitter_mm(history: list, axis: int, filtered: bool, window_s: float = 1.5) -> float:
    """How much the signal moves between consecutive frames, on average.

    Deliberately mean absolute *consecutive difference* rather than a standard
    deviation. Standard deviation cannot tell a genuine slow lean apart from
    frame-to-frame noise — leaning steadily across 10 cm gives a big standard
    deviation and no jitter at all. The difference between neighbouring samples
    only sees the fast component, which is what we are trying to kill.
    """
    if len(history) < 3:
        return 0.0
    cutoff = history[-1][0] - window_s
    values = [(h[2] if filtered else h[1])[axis] for h in history if h[0] >= cutoff]
    if len(values) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(values))))


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------

PANEL_W, PANEL_H = 1180, 700
FONT = cv2.FONT_HERSHEY_SIMPLEX
AXIS_NAMES = ("X", "Y", "Z")


def draw_graph(canvas, origin, size, history, axis: int) -> None:
    """Raw against filtered, over the last few seconds. The whole point of the
    program is visible here and nowhere else."""
    ox, oy = origin
    w, h = size
    cv2.rectangle(canvas, (ox, oy), (ox + w, oy + h), (55, 55, 62), 1)
    cv2.putText(canvas, f"{AXIS_NAMES[axis]} over the last {HISTORY_SECONDS:.0f} s",
                (ox, oy - 10), FONT, 0.5, (150, 150, 160), 1)
    cv2.putText(canvas, "raw", (ox + w - 106, oy - 10), FONT, 0.45, (95, 95, 130), 1)
    cv2.putText(canvas, "filtered", (ox + w - 66, oy - 10), FONT, 0.45,
                (110, 235, 130), 1)

    if len(history) < 3:
        cv2.putText(canvas, "collecting...", (ox + 14, oy + h // 2), FONT, 0.5,
                    (100, 100, 110), 1)
        return

    t_end = history[-1][0]
    t_start = t_end - HISTORY_SECONDS
    raw = np.array([h[1][axis] for h in history])
    filt = np.array([h[2][axis] for h in history])
    times = np.array([h[0] for h in history])

    lo = min(raw.min(), filt.min())
    hi = max(raw.max(), filt.max())
    mid = (lo + hi) * 0.5
    # A floor on the range, so that holding still does not auto-scale a
    # fraction of a millimetre of noise into a dramatic-looking waveform.
    span = max(hi - lo, 20.0) * 0.6
    lo, hi = mid - span, mid + span

    def to_px(t, v):
        px = ox + int((t - t_start) / HISTORY_SECONDS * w)
        py = oy + int((hi - v) / (hi - lo) * h)
        return px, int(np.clip(py, oy + 1, oy + h - 1))

    for frac in (0.25, 0.5, 0.75):
        y = oy + int(h * frac)
        cv2.line(canvas, (ox + 1, y), (ox + w - 1, y), (34, 34, 40), 1)
    cv2.putText(canvas, f"{hi - mid:+.0f} mm", (ox + 6, oy + 16), FONT, 0.36,
                (75, 75, 85), 1)
    cv2.putText(canvas, f"{lo - mid:+.0f} mm", (ox + 6, oy + h - 8), FONT, 0.36,
                (75, 75, 85), 1)

    for series, colour, thickness in ((raw, (105, 105, 140), 1),
                                      (filt, (110, 235, 130), 2)):
        pts = np.array([to_px(t, v) for t, v in zip(times, series)], np.int32)
        cv2.polylines(canvas, [pts], False, colour, thickness, cv2.LINE_AA)


def draw_filter_panel(canvas, origin, tracker, history, axis: int) -> None:
    ox, oy = origin
    cv2.putText(canvas, "filter", (ox, oy), FONT, 0.5, (150, 150, 160), 1)

    raw_j = jitter_mm(history, axis, filtered=False)
    filt_j = jitter_mm(history, axis, filtered=True)
    reduction = (1.0 - filt_j / raw_j) * 100.0 if raw_j > 1e-6 else 0.0

    y = oy + 30
    cv2.putText(canvas, f"jitter raw       {raw_j:5.2f} mm/frame", (ox, y),
                FONT, 0.46, (150, 150, 162), 1)
    y += 24
    cv2.putText(canvas, f"jitter filtered  {filt_j:5.2f} mm/frame", (ox, y),
                FONT, 0.46, (110, 235, 130), 1)
    y += 24
    colour = (110, 235, 130) if reduction > 50 else (150, 150, 162)
    cv2.putText(canvas, f"reduction        {reduction:5.1f} %", (ox, y),
                FONT, 0.46, colour, 1)
    y += 34

    for text in (
        f"min_cutoff  {tracker.min_cutoff:5.2f} Hz    [ ]",
        f"beta        {tracker.beta:6.4f}    - =",
        f"predict     {tracker.predict_ms:5.1f} ms" +
        ("      p" if tracker.predict_enabled else "  OFF  p"),
    ):
        cv2.putText(canvas, text, (ox, y), FONT, 0.46, (170, 170, 182), 1)
        y += 24


def draw_readout(canvas, origin, sample, tracker) -> None:
    ox, oy = origin
    colour = (110, 235, 130) if sample.valid else (90, 90, 110)
    y = oy
    for i, (name, hint) in enumerate((("X", "lean right -> +"),
                                      ("Y", "sit up -> +"),
                                      ("Z", "from the glass"))):
        value = (sample.x, sample.y, sample.z)[i]
        cv2.putText(canvas, name, (ox, y), FONT, 0.7, (170, 170, 180), 2)
        cv2.putText(canvas, f"{value / 10:+7.1f} cm", (ox + 30, y), FONT, 0.7,
                    colour, 2)
        cv2.putText(canvas, hint, (ox + 178, y), FONT, 0.4, (110, 110, 122), 1)
        y += 34

    y += 12
    status = f"tracking, {sample.n_faces} face(s)" if sample.valid \
        else "NO FACE - easing to centre"
    cv2.putText(canvas, status, (ox, y), FONT, 0.5,
                colour if sample.valid else (100, 150, 235), 1)
    y += 28
    for text in (f"head speed  {sample.speed:6.0f} mm/s",
                 f"tracker     {sample.tracker_fps:5.1f} fps",
                 f"inference   {sample.inference_ms:5.1f} ms"):
        cv2.putText(canvas, text, (ox, y), FONT, 0.46, (150, 150, 162), 1)
        y += 24


def draw_preview(canvas, origin, size, sample, tracker) -> None:
    ox, oy = origin
    w, h = size
    frame = sample.preview
    if frame is None:
        cv2.rectangle(canvas, (ox, oy), (ox + w, oy + h), (55, 55, 62), 1)
        cv2.putText(canvas, "waiting for camera", (ox + 12, oy + h // 2), FONT,
                    0.5, (110, 110, 120), 1)
        return
    small = cv2.resize(frame, (w, h))
    if tracker.mirror_preview:
        small = cv2.flip(small, 1)
    if sample.iris_px:
        for u, v in sample.iris_px:
            sx = u / max(tracker.frame_w, 1) * w
            if tracker.mirror_preview:
                sx = w - sx
            cv2.circle(small, (int(sx), int(v / max(tracker.frame_h, 1) * h)),
                       3, (0, 255, 255), -1)
    canvas[oy:oy + h, ox:ox + w] = small
    cv2.rectangle(canvas, (ox, oy), (ox + w, oy + h), (55, 55, 62), 1)


def draw_help(canvas, origin) -> None:
    ox, oy = origin
    cv2.putText(canvas, "controls", (ox, oy), FONT, 0.46, (150, 150, 160), 1)
    y = oy + 24
    for key, what in [("1 2 3", "graph X, Y or Z"),
                      ("[ ]", "min_cutoff (still)"),
                      ("- =", "beta (moving)"),
                      ("p", "prediction on/off"),
                      ("r", "reset to config"),
                      ("w", "save the filter"),
                      ("m", "mirror preview"),
                      ("q", "quit")]:
        cv2.putText(canvas, key, (ox, y), FONT, 0.44, (160, 160, 172), 1)
        cv2.putText(canvas, what, (ox + 58, y), FONT, 0.44, (95, 95, 108), 1)
        y += 21


def render(canvas, sample, tracker, history, axis) -> None:
    cv2.putText(canvas, "one euro filter - jitter without lag", (26, 34), FONT,
                0.62, (210, 210, 220), 1)
    cv2.line(canvas, (26, 48), (PANEL_W - 26, 48), (45, 45, 52), 1)

    draw_readout(canvas, (30, 92), sample, tracker)
    draw_filter_panel(canvas, (30, 330), tracker, history, axis)
    draw_preview(canvas, (30, 512), (330, 170), sample, tracker)
    draw_graph(canvas, (400, 92), (750, 260), history, axis)
    draw_help(canvas, (400, 400))
    draw_tuning_advice(canvas, (620, 400), tracker, history, axis)


def draw_tuning_advice(canvas, origin, tracker, history, axis) -> None:
    """Say what to do next, rather than leaving two abstract numbers on screen."""
    ox, oy = origin
    cv2.putText(canvas, "how to tune", (ox, oy), FONT, 0.46, (150, 150, 160), 1)
    lines = [
        "Hold still and watch the jitter figure.",
        "Lower min_cutoff with [ until it settles.",
        "",
        "Then move your head quickly and watch the",
        "two traces. If the bright one trails the",
        "dim one, raise beta with =.",
        "",
        "Too far either way is obvious: a floaty",
        "scene means beta is too low, a shivering",
        "one means min_cutoff is too high.",
    ]
    y = oy + 24
    for text in lines:
        cv2.putText(canvas, text, (ox, y), FONT, 0.42, (120, 120, 132), 1)
        y += 19


# ----------------------------------------------------------------------------
# Self test
# ----------------------------------------------------------------------------


def selftest() -> int:
    print("\n" + "=" * 72)
    print("One Euro filter self test")
    print("=" * 72)
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}" + (f"  — {detail}" if detail else ""))

    # --- the alpha relation -------------------------------------------------
    a = smoothing_alpha(1.0, 1.0 / 30.0)
    check("alpha in range", 0.0 < a < 1.0, f"{a:.4f} at 1 Hz, 30 fps")
    check("higher cutoff lets more through",
          smoothing_alpha(10.0, 1 / 30) > smoothing_alpha(1.0, 1 / 30))
    check("longer gap lets more through",
          smoothing_alpha(1.0, 1 / 15) > smoothing_alpha(1.0, 1 / 60))

    # --- a constant signal must survive intact ------------------------------
    f = OneEuroFilter(1.0, 0.007)
    for _ in range(200):
        out = f(100.0, 1 / 30)
    check("constant input is preserved", abs(out - 100.0) < 1e-6, f"{out:.6f}")

    # --- noise reduction ----------------------------------------------------
    rng = np.random.default_rng(7)
    noise = 100.0 + rng.normal(0, 3.0, 400)
    f = OneEuroFilter(1.0, 0.007)
    filtered = [f(v, 1 / 30) for v in noise]
    raw_j = float(np.mean(np.abs(np.diff(noise[100:]))))
    filt_j = float(np.mean(np.abs(np.diff(filtered[100:]))))
    check("jitter is reduced when still", filt_j < raw_j * 0.35,
          f"{raw_j:.2f} -> {filt_j:.2f} mm/frame, {100*(1-filt_j/raw_j):.0f}% less")

    # --- and it must get out of the way when moving -------------------------
    # A ramp at 400 mm/s, which is an ordinary head movement. The adaptive
    # filter should track it far more closely than a fixed low cutoff would.
    ramp = [400.0 * (i / 30.0) for i in range(90)]
    adaptive = OneEuroFilter(1.0, 0.007)
    fixed = OneEuroFilter(1.0, 0.0)  # beta=0 makes it an ordinary low pass
    lag_adaptive = abs(ramp[-1] - [adaptive(v, 1 / 30) for v in ramp][-1])
    lag_fixed = abs(ramp[-1] - [fixed(v, 1 / 30) for v in ramp][-1])
    check("adaptive cutoff cuts the lag", lag_adaptive < lag_fixed * 0.6,
          f"{lag_fixed:.1f} mm behind at beta=0, {lag_adaptive:.1f} mm with beta=0.007")

    # --- frame-rate independence -------------------------------------------
    #
    # This is the one that matters most. A fixed per-frame blend factor would
    # smooth over a fixed number of frames, so the same settings would behave
    # differently on a 30 fps machine and a 120 fps one. The project has been
    # caught by exactly that before, so it gets asserted rather than assumed.
    # Sampling at a lower rate is inherently later — a zero-order hold on a
    # ramp is half a frame behind whatever the filter does, and no filter can
    # remove that. So the comparison adds that unavoidable part back before
    # asking whether the *filter* behaves the same at every rate.
    speed = 200.0

    def run_at(fps: float) -> float:
        filt = OneEuroFilter(1.0, 0.007)
        out = 0.0
        for i in range(int(2.0 * fps)):
            out = filt(speed * (i / fps), 1.0 / fps)  # 200 mm/s ramp, 2 seconds
        return out

    rates = (15, 30, 60, 120, 240)
    raw_results = {fps: run_at(fps) for fps in rates}
    compensated = {fps: v + speed / fps / 2.0 for fps, v in raw_results.items()}
    spread = max(compensated.values()) - min(compensated.values())
    check("frame rate independent", spread < 1.5,
          "at 2 s into a 200 mm/s ramp, sampling delay accounted for: " +
          ", ".join(f"{fps}fps={v:.2f}" for fps, v in compensated.items()) +
          f"  (spread {spread:.3f} mm)")

    # --- degenerate time steps ---------------------------------------------
    f = OneEuroFilter(1.0, 0.007)
    f(50.0, 1 / 30)
    check("dt=0 does not divide by zero", abs(f(80.0, 0.0) - f._x.value) < 1e-9)
    check("dt<0 is ignored", math.isfinite(f(80.0, -1.0)))

    # --- prediction ---------------------------------------------------------
    v3 = OneEuroVec3(1.0, 0.007, 1.0)
    for i in range(120):  # settle onto a steady 300 mm/s in x
        out_plain = v3((300.0 * i / 60.0, 0.0, 600.0), 1 / 60, 0.0)
    v3p = OneEuroVec3(1.0, 0.007, 1.0)
    for i in range(120):
        out_pred = v3p((300.0 * i / 60.0, 0.0, 600.0), 1 / 60, 0.020)
    lead = out_pred[0] - out_plain[0]
    check("prediction leads by v*t", abs(lead - 300.0 * 0.020) < 0.6,
          f"{lead:.2f} mm, expected {300.0 * 0.020:.2f}")
    check("prediction leaves a still signal alone",
          abs(OneEuroVec3(1.0, 0.007, 1.0)((5.0, 5.0, 600.0), 1 / 60, 0.02)[0] - 5.0) < 1e-9)

    # --- axes stay independent ---------------------------------------------
    v3 = OneEuroVec3(1.0, 0.007, 1.0)
    for i in range(90):
        out = v3((500.0 * i / 30.0, 0.0, 600.0), 1 / 30, 0.0)
    check("a fast X does not disturb Y or Z",
          abs(out[1]) < 1e-9 and abs(out[2] - 600.0) < 1e-9,
          f"y={out[1]:.6f}, z={out[2]:.4f}")

    # --- reset --------------------------------------------------------------
    v3.reset()
    check("reset clears the state", v3((0.0, 0.0, 600.0), 1 / 30)[0] == 0.0)

    # --- the jitter metric --------------------------------------------------
    now = time.perf_counter()
    steady = [(now - 1.0 + i * 0.01, (0.0, 0.0, 600.0), (0.0, 0.0, 600.0))
              for i in range(100)]
    check("no jitter on a flat signal", jitter_mm(steady, 0, True) == 0.0)
    # A steady lean must not register as jitter — this is why the metric is a
    # consecutive difference and not a standard deviation.
    lean = [(now - 1.0 + i * 0.01, (i * 0.1, 0.0, 600.0), (i * 0.1, 0.0, 600.0))
            for i in range(100)]
    check("a steady lean is not jitter", jitter_mm(lean, 0, True) < 0.11,
          f"{jitter_mm(lean, 0, True):.3f} mm/frame across a 10 mm lean")
    check("short history is handled", jitter_mm([], 0, True) == 0.0)

    print()
    if failures:
        print(f"  {failures} check(s) failed.")
        return 1
    print("  All self tests passed.")
    return 0


# ----------------------------------------------------------------------------


def _set_filter_values(text: str, values: dict[str, dict[str, float]]) -> str:
    """Write the tuned numbers back, leaving the comments in place."""
    out = []
    section = None
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        m_sec = re.match(r"^(\w+)\s*:", stripped)
        if m_sec:
            section = m_sec.group(1)
        elif section in values:
            for key, value in values[section].items():
                m = re.match(rf"^(\s+){key}\s*:\s*\S+(\s*#.*)?$", stripped)
                if m:
                    nl = "\n" if line.endswith("\n") else ""
                    line = f"{m.group(1)}{key}: {value:g}{m.group(2) or ''}{nl}"
                    break
        out.append(line)
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="One Euro filter tuning.")
    parser.add_argument("--selftest", action="store_true",
                        help="check the filter maths without a camera")
    args = parser.parse_args()
    if args.selftest:
        return selftest()

    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg["tracking"].get("focal_px") is None:
        print("\n  focal_px is not calibrated — distances will be wrong.")
        print('  Run:  python "calibration.py" --focal\n')

    tracker = EyeTracker(cfg)
    tracker.start()

    deadline = time.perf_counter() + 12.0
    while tracker.frame_w == 0 and tracker.error is None:
        if time.perf_counter() > deadline:
            tracker.stop()
            print("Timed out waiting for the camera.")
            return 1
        time.sleep(0.05)
    if tracker.error:
        print(f"Tracker failed to start: {tracker.error}")
        return 1

    print(f"Camera: {tracker.backend} at {tracker.frame_w}x{tracker.frame_h}")
    print("Hold still to tune min_cutoff, then move to tune beta.\n")

    window = "one euro filter"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    axis = 0
    dirty = False

    try:
        while True:
            canvas = np.full((PANEL_H, PANEL_W, 3), 18, np.uint8)
            render(canvas, tracker.latest, tracker, tracker.history_snapshot(), axis)
            cv2.imshow(window, canvas)

            key = cv2.waitKey(8) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("1"), ord("2"), ord("3")):
                axis = key - ord("1")
            elif key == ord("["):
                tracker.min_cutoff = max(0.05, tracker.min_cutoff - 0.05)
                dirty = True
            elif key == ord("]"):
                tracker.min_cutoff = min(20.0, tracker.min_cutoff + 0.05)
                dirty = True
            elif key == ord("-"):
                tracker.beta = max(0.0, tracker.beta - 0.001)
                dirty = True
            elif key == ord("="):
                tracker.beta = min(1.0, tracker.beta + 0.001)
                dirty = True
            elif key == ord("p"):
                tracker.predict_enabled = not tracker.predict_enabled
            elif key == ord("m"):
                tracker.mirror_preview = not tracker.mirror_preview
            elif key == ord("r"):
                tracker.reset_filter()
                dirty = False
                print("  reset to the values in config.yaml")
            elif key == ord("w"):
                text = CONFIG_PATH.read_text(encoding="utf-8")
                text = _set_filter_values(text, {
                    "filter": {"min_cutoff": tracker.min_cutoff, "beta": tracker.beta},
                    "tracking": {"predict_ms": tracker.predict_ms
                                 if tracker.predict_enabled else 0.0},
                })
                CONFIG_PATH.write_text(text, encoding="utf-8", newline="")
                dirty = False
                print(f"  saved: min_cutoff={tracker.min_cutoff:g}  "
                      f"beta={tracker.beta:g}")

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        tracker.stop()
        tracker.join(timeout=2.0)
        cv2.destroyAllWindows()

    if dirty:
        print("\n  Filter values were changed but not saved. Press w to keep them.")
    print("\n  Next: 'grid room.py'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
