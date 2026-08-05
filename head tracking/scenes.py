"""
Step 6 of the parallax window: the webcam drives it. This is the real thing.

Steps 1 to 5 built two halves that were each proven on their own — a head
tracker verified on camera with the right signs, and an off-axis projection
verified to a millionth of a pixel against a perfect input. This program is the
join, and nothing else. The projection is unchanged from step 5, the tracker is
unchanged from step 3; the mouse has simply been unplugged and the webcam
plugged in.

That separation is why, if something is wrong here, it is the tracker or the
join — the maths has already been ruled out.

Sit in front of the screen and move your head. The screen should stop being a
picture.

  * `--eye-source mouse` puts the mouse back in charge. Useful for recording:
    the illusion is for *your* eyes, not a camera's, so a video shot from a
    fixed tripod while you move your head shows a scene wobbling for no reason
    and looks broken. Drive it by hand instead, or move the recording camera
    and your head together.
  * `D` opens the debug HUD — eye position, jitter, both frame rates, the live
    frustum extents, the measured pipeline latency, and a webcam thumbnail with
    the iris landmarks marked.

Usage:

    python "head tracked parallax.py"
    python "head tracked parallax.py" --eye-source mouse
    python "head tracked parallax.py" --selftest      check the maths
    python "head tracked parallax.py" --render-test   render frames to PNG

    move head    move the eye         TAB   off-axis / symmetric
    D            debug HUD            f     fullscreen
    r            recentre             g     cycle grid density
    a            auto-sweep           q     quit


Step 5, for reference — the illusion itself, driven by the mouse.

This is where the screen stops being a picture and becomes a window. Move the
mouse and the room does not slide — it *opens*. Move left and you see more of
the right-hand wall, exactly as you would leaning across a real window frame.
The near shapes sweep across the far ones. The grid on the back wall swings
while the mouth of the room stays nailed to the glass.

**Nothing rotates.** That is the entire trick, and it is worth being blunt
about because it is so counter-intuitive. The virtual camera never turns to
look at anything; its view direction stays perpendicular to the screen at all
times. What changes is the *shape* of the view volume: the frustum is sheared
so that its four side planes always pass exactly through the four physical
corners of the window. This is Robert Kooima's generalized perspective
projection.

If you were to implement this by orbiting a camera around the scene with a
`lookAt()`, you would get step 4's behaviour dressed up differently, and it
would be obvious immediately.

**Press TAB to switch between this and step 4's symmetric frustum.** That is
the demonstration: the same room, the same eye position, the same everything,
and one of them has depth and the other does not.

The eye is driven by the mouse here rather than by the webcam, on purpose. The
frustum maths and the head tracker are two separate things that can each be
wrong, and debugging them at the same time means a sign error and a maths error
look identical. Proving the projection against a perfect, noiseless, zero-lag
input first means that when the tracker is connected in step 6, anything that
goes wrong is the tracker.

Usage:

    python "mouse parallax.py"
    python "mouse parallax.py" --render-test   render frames to PNG
    python "mouse parallax.py" --selftest      check the maths, no window

    move mouse   move the eye         TAB   off-axis / symmetric
    scroll       eye nearer/further   a     auto-sweep the eye in a circle
    f            fullscreen           g     cycle grid density
    r            recentre the eye     q     quit
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
MODEL_PATH = HERE / "models" / "face_landmarker.task"

# MediaPipe pulls in TensorFlow, which is noisy on import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

ROOM_DEPTH_MM = 1200.0
DEFAULT_EYE_Z = 600.0
GRID_DENSITIES = (6, 12, 24)


# ----------------------------------------------------------------------------
# Display geometry
# ----------------------------------------------------------------------------


def set_dpi_aware() -> None:
    """Ask Windows to stop lying about the screen size.

    Must happen before any window exists. Without it, a process on a 150%-scaled
    desktop is told this 2560x1600 panel is 1707x1067, and every conversion
    between pixels and millimetres comes out 1.5x wrong.
    """
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


class ScreenGeometry:
    """Converts between desktop pixels and screen-space millimetres.

    Screen space has its origin at the centre of the *panel*, +X right, +Y up.
    But the frustum has to be built from the rectangle that is actually being
    drawn into, which is the **window**, not the panel — so a floating window
    needs to know where its own corners are in physical space. That is what this
    class is for, and it is why the pixels-per-millimetre scale exists.

    Fullscreen is then just the case where the window rectangle happens to equal
    the panel rectangle. There is no separate code path for it.
    """

    def __init__(self, cfg: dict) -> None:
        d = cfg["display"]
        self.panel_w_mm = float(d["screen_w_mm"])
        self.panel_h_mm = float(d["screen_h_mm"])
        self.native_px = [int(v) for v in d["native_px"]]
        self.px_per_mm = self.native_px[0] / self.panel_w_mm

        # Sanity: the two axes should agree, since pixels are square on any
        # normal panel. A big disagreement means a mistyped measurement.
        vertical = self.native_px[1] / self.panel_h_mm
        if abs(vertical - self.px_per_mm) / self.px_per_mm > 0.02:
            print(f"  warning: pixels are not square — {self.px_per_mm:.3f} px/mm "
                  f"across, {vertical:.3f} down. Check screen_w_mm/screen_h_mm.")

    def window_rect_mm(self, win_x: int, win_y: int, win_w: int, win_h: int):
        """The window's corners in screen-space millimetres.

        `win_x, win_y` are the window's top-left in desktop pixels, where the
        desktop's Y grows downward — hence the negation, since screen-space Y
        grows up.

        Returns Kooima's corner triple: bottom-left, bottom-right, top-left.
        """
        centre_x_px = win_x + win_w * 0.5
        centre_y_px = win_y + win_h * 0.5

        cx_mm = (centre_x_px - self.native_px[0] * 0.5) / self.px_per_mm
        cy_mm = -(centre_y_px - self.native_px[1] * 0.5) / self.px_per_mm

        half_w = win_w * 0.5 / self.px_per_mm
        half_h = win_h * 0.5 / self.px_per_mm

        pa = np.array([cx_mm - half_w, cy_mm - half_h, 0.0])  # bottom-left
        pb = np.array([cx_mm + half_w, cy_mm - half_h, 0.0])  # bottom-right
        pc = np.array([cx_mm - half_w, cy_mm + half_h, 0.0])  # top-left
        return pa, pb, pc


# ----------------------------------------------------------------------------
# Head tracking — carried over from steps 2 and 3, unchanged
# ----------------------------------------------------------------------------

IRIS_LEFT = 468
IRIS_RIGHT = 473


def smoothing_alpha(cutoff_hz: float, dt: float) -> float:
    """Blend factor for a one-pole low pass, from a cutoff and a time step.

    `1 - exp(-dt/tau)`, the exact discretisation, not the `1/(1 + tau/dt)` in
    the One Euro paper — see step 3 for the measurement that settled it. The
    two agree to first order; this one holds up better when the frame rate
    varies, which it does far more here than it did in step 3, because the
    render loop and the camera now run at completely different rates.
    """
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 - math.exp(-dt / tau)


class LowPass:
    def __init__(self) -> None:
        self.value: float | None = None

    def __call__(self, x: float, alpha: float) -> float:
        self.value = x if self.value is None else alpha * x + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class OneEuroFilter:
    """Adaptive low pass: cutoff rises with the speed of the signal, so it
    filters hard when you are still and gets out of the way when you move."""

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
            return self._x.value if self._x.value is not None else x
        raw_velocity = 0.0 if self._last is None else (x - self._last) / dt
        self._last = x
        self.velocity = self._dx(raw_velocity, smoothing_alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(self.velocity)
        return self._x(x, smoothing_alpha(cutoff, dt))


class OneEuroVec3:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.filters = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(3)]

    def reset(self) -> None:
        for f in self.filters:
            f.reset()

    def __call__(self, xyz, dt: float, predict_s: float = 0.0):
        out = [f(v, dt) for f, v in zip(self.filters, xyz)]
        if predict_s > 0.0:
            out = [v + f.velocity * predict_s for f, v in zip(self.filters, out)]
        return out[0], out[1], out[2]

    @property
    def speed(self) -> float:
        return math.sqrt(sum(f.velocity ** 2 for f in self.filters))


def iris_separation_px(landmarks, frame_w: int, frame_h: int):
    """Pupil separation, and how far the head is turned.

    The 3D distance between the irises is the separation with the head-turn
    foreshortening already removed, so 3D/2D is exactly 1/cos(yaw). Cosine
    being even means MediaPipe's z sign convention cannot cause a sign error.
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


class EyeSample:
    """One reading from the tracker. `xyz` is filtered and predicted."""

    __slots__ = ("x", "y", "z", "raw", "vel", "valid", "ipd_px", "yaw_deg",
                 "speed", "n_faces", "captured_at", "published_at",
                 "tracker_fps", "inference_ms", "preview", "iris_px")

    def __init__(self, x=0.0, y=0.0, z=DEFAULT_EYE_Z, raw=(0.0, 0.0, DEFAULT_EYE_Z),
                 vel=(0.0, 0.0, 0.0), valid=False, ipd_px=0.0, yaw_deg=0.0,
                 speed=0.0, n_faces=0, captured_at=0.0, published_at=0.0,
                 tracker_fps=0.0, inference_ms=0.0, preview=None, iris_px=None):
        self.x, self.y, self.z = x, y, z
        self.raw = raw
        # Filtered velocity in mm/s, kept so the renderer can carry the eye
        # forward between tracker updates instead of holding it still.
        self.vel = vel
        self.valid = valid
        self.ipd_px = ipd_px
        self.yaw_deg = yaw_deg
        self.speed = speed
        self.n_faces = n_faces
        # Two timestamps, not one. `captured_at` is when the camera frame came
        # back; `published_at` is when the result reached the shared slot. The
        # gap between captured_at and the moment a frame is drawn is the
        # pipeline latency the whole design is fighting.
        self.captured_at = captured_at
        self.published_at = published_at
        self.tracker_fps = tracker_fps
        self.inference_ms = inference_ms
        self.preview = preview
        self.iris_px = iris_px

    @property
    def position(self):
        return (self.x, self.y, self.z)


class EyeTracker(threading.Thread):
    """Capture, inference, millimetre conversion and filtering, off the render
    thread entirely.

    A single lock-protected slot, never a queue. The renderer runs at hundreds
    of frames a second and the camera at thirty; if the consumer ever falls
    behind, the correct thing is to discard everything but the newest reading,
    because an old head position is not stale data, it is wrong data.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(daemon=True, name="eye-tracker")
        self.cfg = cfg
        self._lock = threading.Lock()
        self._latest = EyeSample()
        self._history: deque = deque(maxlen=400)
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

        self.predict_ms = float(t.get("predict_ms", 20.0))
        self.predict_enabled = self.predict_ms > 0.0
        self.filter = OneEuroVec3(float(f.get("min_cutoff", 1.0)),
                                  float(f.get("beta", 0.007)),
                                  float(f.get("d_cutoff", 1.0)))

        self.focal_px: float | None = (
            float(t["focal_px"]) if t.get("focal_px") is not None else None)
        self.focal_is_estimated = self.focal_px is None

        self.frame_w = 0
        self.frame_h = 0
        self.capture_fps = 0.0
        self.backend = ""
        self.error: str | None = None
        self.default_position = (0.0, 0.0, DEFAULT_EYE_Z)

    @property
    def latest(self) -> EyeSample:
        with self._lock:
            return self._latest

    def history_snapshot(self) -> list:
        with self._lock:
            return list(self._history)

    def stop(self) -> None:
        self._running.clear()

    def _open_camera(self):
        import cv2

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
            # will not change resolution once the stream has started delivering:
            # it keeps reporting the requested size while returning nothing but
            # bad frames, for ever. Measured — 0 good out of 40 with a probe
            # read first, 40 out of 40 without. DirectShow tolerates either
            # order, which is exactly why this stayed hidden until the default
            # backend changed.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam["width"]))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam["height"]))
            # A deep buffer hands back a frame that is already old, which is
            # latency that can never be recovered further down the pipe.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self._safe_read(cap)[0]:
                self.backend = name
                return cap
            cap.release()
        raise RuntimeError(f"could not open camera index {index}")

    @staticmethod
    def _safe_read(cap):
        """`cap.read()`, but a bad frame is a skipped frame rather than a crash.

        Media Foundation occasionally hands OpenCV a frame with an inconsistent
        stride, and OpenCV raises `cv2.error: _step >= minstep` when it tries to
        wrap it. It is intermittent and harmless — the next frame is fine — but
        uncaught it kills the capture thread outright, and the renderer happily
        carries on drawing a head position that has stopped updating.
        """
        import cv2

        try:
            return cap.read()
        except cv2.error:
            return False, None

    def _measure_capture_rate(self, cap, frames: int = 20) -> float:
        """How fast the camera really delivers, measured rather than believed.

        `CAP_PROP_FPS` cannot be trusted — DirectShow reports 0 for this camera
        while delivering 10 fps, which is how a 3x tracking slowdown went
        unnoticed until it was measured directly. A slow camera is not a
        cosmetic problem: at 10 fps the head position is 100 ms old before the
        renderer even sees it, which is most of the entire latency budget.
        """
        for _ in range(5):          # let auto-exposure settle first
            self._safe_read(cap)
        start = time.perf_counter()
        got = 0
        for _ in range(frames):
            if self._safe_read(cap)[0]:
                got += 1
        elapsed = time.perf_counter() - start
        return got / elapsed if elapsed > 0 else 0.0

    @staticmethod
    def _face_size(landmarks, frame_w: int, frame_h: int) -> float:
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        return (max(xs) - min(xs)) * frame_w * (max(ys) - min(ys)) * frame_h

    def _choose_face(self, faces, frame_w: int, frame_h: int, previous):
        """Largest face wins, but an already-tracked one is only given up when
        another is clearly bigger. Without the margin, two people at similar
        distances make the tracker flip between them every few frames."""
        sizes = [self._face_size(f, frame_w, frame_h) for f in faces]
        largest = int(np.argmax(sizes))
        if previous is None or len(faces) == 1:
            return largest
        best, best_dist = 0, float("inf")
        for i, f in enumerate(faces):
            u = (f[IRIS_LEFT].x + f[IRIS_RIGHT].x) * 0.5 * frame_w
            v = (f[IRIS_LEFT].y + f[IRIS_RIGHT].y) * 0.5 * frame_h
            dist = math.hypot(u - previous[0], v - previous[1])
            if dist < best_dist:
                best, best_dist = i, dist
        return largest if sizes[largest] > sizes[best] * 1.25 else best

    def run(self) -> None:
        """Wrapper so that *any* failure in the capture loop is recorded rather
        than merely printed to stderr.

        This was learned the hard way: a malformed frame killed this thread, and
        because a dead daemon thread is silent from the main thread's point of
        view, the renderer carried on drawing a head position that had quietly
        stopped updating. A frozen illusion looks like a bad filter, not like a
        crash, so it could have been debugged for a long time in the wrong place.
        """
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 — deliberately broad
            import traceback

            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    def _run(self) -> None:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        try:
            cap = self._open_camera()
        except Exception as exc:  # noqa: BLE001 — reported to the main thread
            self.error = str(exc)
            return

        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.focal_px is None:
            self.focal_px = fallback_focal_px(self.frame_w)

        self.capture_fps = self._measure_capture_rate(cap)
        if self.capture_fps < 20.0:
            print(f"\n  WARNING: the camera is only delivering "
                  f"{self.capture_fps:.1f} fps at {self.frame_w}x{self.frame_h} "
                  f"over {self.backend}.")
            print(f"  That is {1000 / max(self.capture_fps, 1e-6):.0f} ms between "
                  f"head updates, and the illusion will lag badly.")
            print("  Try a different camera.backend in config.yaml (msmf is "
                  "usually the fast one),")
            print("  or drop camera.width/height to 640x480.\n")

        landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO, num_faces=2))

        cx, cy = self.frame_w * 0.5, self.frame_h * 0.5
        start = time.perf_counter()
        frame_times: list[float] = []
        previous_eye_px = None
        last_good = None
        lost_since = None
        last_time = None

        consecutive_failures = 0
        while self._running.is_set():
            ok, frame = self._safe_read(cap)
            if not ok:
                # The odd bad frame is normal. A long run of them means the
                # camera has gone away — unplugged, or grabbed by another app —
                # and that has to surface rather than spin silently forever.
                consecutive_failures += 1
                if consecutive_failures > 150:
                    self.error = "the camera stopped returning frames"
                    break
                time.sleep(0.005)
                continue
            consecutive_failures = 0
            captured_at = time.perf_counter()
            dt = 0.0 if last_time is None else captured_at - last_time
            last_time = captured_at

            frame_times.append(captured_at)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                   if len(frame_times) > 1 and frame_times[-1] > frame_times[0] else 0.0)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t0 = time.perf_counter()
            result = landmarker.detect_for_video(
                image, int((captured_at - start) * 1000))
            inference_ms = (time.perf_counter() - t0) * 1000.0

            faces = result.face_landmarks
            if faces:
                lm = faces[self._choose_face(faces, self.frame_w, self.frame_h,
                                             previous_eye_px)]
                ipd_2d, ipd_3d, yaw, mid_u, mid_v = iris_separation_px(
                    lm, self.frame_w, self.frame_h)
                previous_eye_px = (mid_u, mid_v)
                ipd = corrected_ipd_px(ipd_2d, ipd_3d, self.max_yaw)

                z_cam = self.focal_px * self.ipd_mm / max(ipd, 1e-6)
                z_cam = min(self.z_max, max(self.z_min, z_cam))
                # u grows right in the raw (mirrored) image and v grows down,
                # while screen space has +X right and +Y up — hence both
                # negations. Getting either wrong inverts the illusion.
                x_cam = -(mid_u - cx) * z_cam / self.focal_px
                y_cam = -(mid_v - cy) * z_cam / self.focal_px
                if self.flip_x:
                    x_cam = -x_cam
                if self.flip_y:
                    y_cam = -y_cam
                # Shift from the lens to the centre of the panel.
                raw = (x_cam + self.cam_offset[0], y_cam + self.cam_offset[1],
                       z_cam + self.cam_offset[2])
                last_good = raw
                lost_since = None
                valid = True
                iris_px = ((lm[IRIS_LEFT].x * self.frame_w,
                            lm[IRIS_LEFT].y * self.frame_h),
                           (lm[IRIS_RIGHT].x * self.frame_w,
                            lm[IRIS_RIGHT].y * self.frame_h))
            else:
                # Hold the last position, then ease back to centre. Never snap:
                # a jump happens exactly when you look away, which is when it is
                # most likely to be caught out of the corner of your eye.
                if lost_since is None:
                    lost_since = captured_at
                if last_good is None:
                    raw = self.default_position
                else:
                    t = min(1.0, (captured_at - lost_since) / max(self.ease_seconds, 1e-3))
                    ease = t * t * (3.0 - 2.0 * t)
                    raw = tuple(a + (b - a) * ease
                                for a, b in zip(last_good, self.default_position))
                valid = False
                ipd = yaw = 0.0
                iris_px = None

            predict_s = (self.predict_ms / 1000.0) if self.predict_enabled else 0.0
            filtered = self.filter(raw, dt, predict_s)

            sample = EyeSample(
                x=filtered[0], y=filtered[1], z=filtered[2], raw=raw,
                vel=tuple(f.velocity for f in self.filter.filters), valid=valid,
                ipd_px=ipd, yaw_deg=yaw, speed=self.filter.speed,
                n_faces=len(faces), captured_at=captured_at,
                published_at=time.perf_counter(), tracker_fps=fps,
                inference_ms=inference_ms, preview=frame, iris_px=iris_px)
            with self._lock:
                self._latest = sample
                self._history.append((captured_at, raw, filtered))

        cap.release()


def jitter_mm(history: list, axis: int, filtered: bool, window_s: float = 1.5) -> float:
    """Mean absolute difference between consecutive samples — deliberately not a
    standard deviation, which cannot tell a slow deliberate lean apart from
    frame-to-frame noise."""
    if len(history) < 3:
        return 0.0
    cutoff = history[-1][0] - window_s
    values = [(h[2] if filtered else h[1])[axis] for h in history if h[0] >= cutoff]
    if len(values) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(values))))


# ----------------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------------


def frustum(l: float, r: float, b: float, t: float, n: float, f: float) -> np.ndarray:
    """The standard OpenGL perspective matrix, written out rather than called
    from a helper, because the whole project turns on being able to see and
    change these six numbers."""
    return np.array([
        [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
        [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
        [0.0, 0.0, -(f + n) / (f - n), -2 * f * n / (f - n)],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype="f8")


def off_axis_projection(pa, pb, pc, pe, near: float, far: float) -> np.ndarray:
    """Kooima's generalized perspective projection — the heart of the illusion.

    Robert Kooima, "Generalized Perspective Projection" (2008). Given three
    corners of a physical screen and an eye position, all in the same
    coordinate system, it produces a projection matrix whose view volume has
    that screen rectangle as an exact cross-section.

        pa = bottom-left corner of the screen
        pb = bottom-right corner
        pc = top-left corner
        pe = eye position

    How it works, in one sentence: build an orthonormal basis on the screen,
    measure where the eye is relative to it, and set the frustum's four side
    planes so that each one passes through the corresponding screen edge.

    The consequence worth internalising: **the screen corners are nailed to the
    world.** Whatever the eye does, the four corners of the physical panel
    always land on the four corners of the rendered image, because the frustum
    is defined by them. Everything else in the scene moves relative to that
    fixed frame, which is exactly what looking through a real window is like.
    The self-test asserts this to sub-pixel accuracy.

    Contrast with `symmetric_projection` below, which throws away the eye's
    sideways position and so can only translate the picture.
    """
    pa, pb, pc, pe = (np.asarray(v, dtype="f8") for v in (pa, pb, pc, pe))

    # An orthonormal basis attached to the screen.
    vr = pb - pa            # along the bottom edge, pointing right
    vu = pc - pa            # up the left edge
    vr /= np.linalg.norm(vr)
    vu /= np.linalg.norm(vu)
    vn = np.cross(vr, vu)   # screen normal, pointing back out towards the viewer
    vn /= np.linalg.norm(vn)

    # From the eye to three of the corners.
    va = pa - pe
    vb = pb - pe
    vc = pc - pe

    # Perpendicular distance from the eye to the screen plane. Note this is the
    # distance to the *plane*, not to the centre of the screen — sliding
    # sideways does not change it, which is precisely why sideways movement
    # shears the frustum rather than scaling it.
    d = -float(np.dot(vn, va))
    if d < 1e-6:
        # The eye is on or behind the screen plane. There is no sensible
        # projection, so hold it just in front rather than emitting NaNs.
        d = 1e-6

    # The frustum extents, found by projecting the corner vectors onto the
    # screen basis and scaling from distance d to the near plane. These are
    # asymmetric in general: l != -r and b != -t.
    scale = near / d
    l = float(np.dot(vr, va)) * scale
    r = float(np.dot(vr, vb)) * scale
    b = float(np.dot(vu, va)) * scale
    t = float(np.dot(vu, vc)) * scale

    p = frustum(l, r, b, t, near, far)

    # Rotate the world into the screen's basis. For a screen lying in the z=0
    # plane this is the identity, but writing it out keeps the maths correct
    # for a tilted or rotated panel — which is what makes the multi-monitor
    # idea in the spec's "notes for later" a config change rather than a
    # rewrite.
    m = np.eye(4)
    m[0, :3] = vr
    m[1, :3] = vu
    m[2, :3] = vn

    # Move the eye to the origin.
    t_mat = np.eye(4)
    t_mat[:3, 3] = -pe

    return p @ m @ t_mat


def frustum_extents(pa, pb, pc, pe, near: float, far: float):
    """The l/r/b/t values `off_axis_projection` computes, exposed for the HUD
    and the self-test. Duplicated deliberately rather than returned from the
    function above, so the projection stays a single clean expression."""
    pa, pb, pc, pe = (np.asarray(v, dtype="f8") for v in (pa, pb, pc, pe))
    vr = pb - pa
    vu = pc - pa
    vr /= np.linalg.norm(vr)
    vu /= np.linalg.norm(vu)
    vn = np.cross(vr, vu)
    vn /= np.linalg.norm(vn)
    va, vb, vc = pa - pe, pb - pe, pc - pe
    d = max(-float(np.dot(vn, va)), 1e-6)
    scale = near / d
    return (float(np.dot(vr, va)) * scale, float(np.dot(vr, vb)) * scale,
            float(np.dot(vu, va)) * scale, float(np.dot(vu, vc)) * scale, d)


def symmetric_projection(pa, pb, pc, eye, near: float, far: float) -> np.ndarray:
    """An ordinary centred perspective camera, sitting at `eye` and looking
    straight down -Z.

    Note what this ignores: where the eye is *sideways*. The frustum is built
    from the screen rectangle's size alone, symmetric about the view axis, so
    moving the eye left only translates the camera — it does not change the
    shape of the view volume. That is what makes the room slide instead of
    opening up, and it is the thing step 5 fixes.
    """
    width = float(np.linalg.norm(pb - pa))
    height = float(np.linalg.norm(pc - pa))
    centre = (pa + pb + pc) / 3.0 + (pb - pa + pc - pa) / 6.0  # rectangle centre

    # Perpendicular distance from the eye to the screen plane.
    d = max(float(eye[2] - centre[2]), 1.0)

    half_w = width * 0.5 * near / d
    half_h = height * 0.5 * near / d
    proj = frustum(-half_w, half_w, -half_h, half_h, near, far)

    view = np.eye(4)
    view[0, 3] = -eye[0]
    view[1, 3] = -eye[1]
    view[2, 3] = -eye[2]
    return proj @ view


# ----------------------------------------------------------------------------
# The scene
# ----------------------------------------------------------------------------

LINE_VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
uniform float u_depth;
in vec3 in_pos;
in vec3 in_colour;
out vec3 v_colour;
out float v_depth;
void main() {
    v_colour = in_colour;
    // World Z is negative going away from the viewer, so this is 0 at the
    // glass and 1 at the back wall. Used to fade the room into darkness.
    v_depth = clamp(-in_pos.z / u_depth, 0.0, 1.0);
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

LINE_FRAGMENT_SHADER = """
#version 330
in vec3 v_colour;
in float v_depth;
out vec4 f_colour;
void main() {
    // Squared falloff. Linear fade looks flat; the eye reads a steeper curve
    // as distance, and the receding parallel lines are the strongest depth cue
    // this scene has.
    //
    // The 0.22 floor matters: falling all the way to black makes the back wall
    // disappear, and losing it costs the far end of every receding line — which
    // is precisely the geometry the illusion depends on being able to see.
    float fade = 1.0 - v_depth;
    fade = fade * fade * 0.78 + 0.22;
    f_colour = vec4(v_colour * fade, 1.0);
}
"""

SOLID_VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
uniform float u_depth;      // how far back the scene extends, in mm
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_colour;
out vec3 v_normal;
out vec3 v_colour;
out float v_depth;
void main() {
    v_normal = in_normal;
    v_colour = in_colour;
    v_depth = clamp(-in_pos.z / u_depth, 0.0, 1.0);
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

SOLID_FRAGMENT_SHADER = """
#version 330
uniform float u_fade;       // 0 = no falloff, 1 = fades hard into darkness
uniform float u_ambient;    // floor on the lighting, so faces are not black
in vec3 v_normal;
in vec3 v_colour;
in float v_depth;
out vec4 f_colour;
void main() {
    // One soft directional light, from up and to the left. The direction
    // matters more than it looks: lighting the stepped ledges from off-axis is
    // what makes each one read as a physical surface catching light, rather
    // than as a flat band of colour.
    vec3 light = normalize(vec3(-0.35, 0.65, 0.55));
    float lambert = max(dot(normalize(v_normal), light), 0.0);
    float shade = u_ambient + (1.0 - u_ambient) * lambert;

    // Squared falloff into darkness. The eye reads a steep brightness gradient
    // as distance, and for the shaft that gradient is doing as much work as the
    // perspective is — a tunnel lit evenly to the back looks like wallpaper.
    float d = 1.0 - v_depth;
    float fade = mix(1.0, d * d, u_fade);
    f_colour = vec4(v_colour * shade * fade, 1.0);
}
"""


def _grid_lines(divisions: int, screen_w: float, screen_h: float,
                depth: float) -> np.ndarray:
    """The five-sided room: floor, ceiling, both side walls, back wall.

    No front wall — that is where you are. The mouth of the box is exactly the
    window rectangle at Z = 0, so the room appears to start at the glass.
    """
    hw, hh = screen_w * 0.5, screen_h * 0.5
    verts: list[float] = []

    wall = (0.30, 0.86, 0.94)   # cyan
    edge = (0.55, 0.98, 1.00)   # brighter, for the corner edges
    back = (0.22, 0.62, 0.72)

    def line(p0, p1, colour):
        verts.extend([*p0, *colour, *p1, *colour])

    depth_steps = [depth * i / divisions for i in range(divisions + 1)]

    # Rings going back: each one is the room's cross-section at that depth.
    # These are the lines that make the perspective unmistakable.
    for z in depth_steps:
        z = -z
        line((-hw, -hh, z), (hw, -hh, z), wall)
        line((hw, -hh, z), (hw, hh, z), wall)
        line((hw, hh, z), (-hw, hh, z), wall)
        line((-hw, hh, z), (-hw, -hh, z), wall)

    # Lines running away from the viewer along each surface.
    across = max(divisions // 2, 3)
    for i in range(across + 1):
        x = -hw + 2 * hw * i / across
        line((x, -hh, 0.0), (x, -hh, -depth), wall)   # floor
        line((x, hh, 0.0), (x, hh, -depth), wall)     # ceiling
    for i in range(max(across // 2, 2) + 1):
        y = -hh + 2 * hh * i / max(across // 2, 2)
        line((-hw, y, 0.0), (-hw, y, -depth), wall)   # left wall
        line((hw, y, 0.0), (hw, y, -depth), wall)     # right wall

    # The four corner edges, brighter so the box reads as solid.
    for sx in (-1, 1):
        for sy in (-1, 1):
            line((sx * hw, sy * hh, 0.0), (sx * hw, sy * hh, -depth), edge)

    # A grid on the back wall.
    for i in range(across + 1):
        x = -hw + 2 * hw * i / across
        line((x, -hh, -depth), (x, hh, -depth), back)
    for i in range(max(across // 2, 2) + 1):
        y = -hh + 2 * hh * i / max(across // 2, 2)
        line((-hw, y, -depth), (hw, y, -depth), back)

    return np.array(verts, dtype="f4")


def _cube(size: float):
    h = size * 0.5
    faces = [
        ((0, 0, 1), [(-h, -h, h), (h, -h, h), (h, h, h), (-h, -h, h), (h, h, h), (-h, h, h)]),
        ((0, 0, -1), [(h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, -h, -h), (-h, h, -h), (h, h, -h)]),
        ((1, 0, 0), [(h, -h, h), (h, -h, -h), (h, h, -h), (h, -h, h), (h, h, -h), (h, h, h)]),
        ((-1, 0, 0), [(-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, -h, -h), (-h, h, h), (-h, h, -h)]),
        ((0, 1, 0), [(-h, h, h), (h, h, h), (h, h, -h), (-h, h, h), (h, h, -h), (-h, h, -h)]),
        ((0, -1, 0), [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, -h), (h, -h, h), (-h, -h, h)]),
    ]
    return [(p, n) for n, tri in faces for p in tri]


def _pyramid(size: float):
    h = size * 0.5
    apex = (0.0, h, 0.0)
    base = [(-h, -h, h), (h, -h, h), (h, -h, -h), (-h, -h, -h)]
    out = []
    for i in range(4):
        p0, p1 = base[i], base[(i + 1) % 4]
        n = np.cross(np.subtract(p1, p0), np.subtract(apex, p0))
        n = n / (np.linalg.norm(n) or 1.0)
        out += [(p0, n), (p1, n), (apex, n)]
    n = (0.0, -1.0, 0.0)
    out += [(base[0], n), (base[2], n), (base[1], n),
            (base[0], n), (base[3], n), (base[2], n)]
    return out


def _octahedron(size: float):
    h = size * 0.5
    tips = [(0, h, 0), (0, -h, 0)]
    ring = [(h, 0, 0), (0, 0, h), (-h, 0, 0), (0, 0, -h)]
    out = []
    for tip_i, tip in enumerate(tips):
        for i in range(4):
            p0, p1 = ring[i], ring[(i + 1) % 4]
            if tip_i == 1:
                p0, p1 = p1, p0
            n = np.cross(np.subtract(p1, p0), np.subtract(tip, p0))
            n = n / (np.linalg.norm(n) or 1.0)
            out += [(p0, n), (p1, n), (tip, n)]
    return out


def _solid_shapes(screen_w: float, screen_h: float, depth: float) -> np.ndarray:
    """Shapes at assorted depths, so there is something for the parallax to act
    on. Seeded, so the scene is the same every run — comparing two screenshots
    is impossible if the furniture moves between them."""
    rng = np.random.default_rng(20260803)
    builders = (_cube, _pyramid, _octahedron)
    palette = [(1.00, 0.42, 0.38), (0.98, 0.78, 0.32), (0.55, 0.85, 1.00),
               (0.72, 0.60, 1.00), (0.45, 0.95, 0.70)]

    # Sized as a fraction of the room, not in absolute millimetres. The room's
    # cross-section is the window, which can be any size the user drags it to —
    # fixed-size shapes would fill a small window completely and vanish in a
    # large one.
    short_side = min(screen_w, screen_h)

    verts: list[float] = []
    for i in range(9):
        size = float(rng.uniform(0.10, 0.20)) * short_side
        shape = builders[i % len(builders)](size)
        colour = palette[i % len(palette)]

        # Keep them clear of the walls so nothing pokes through a surface. The
        # max() guards the degenerate case where a shape is nearly as wide as
        # the room, which would otherwise give an empty range.
        margin = size * 0.9
        span_x = max(screen_w * 0.5 - margin, 0.0)
        span_y = max(screen_h * 0.5 - margin, 0.0)
        cx = float(rng.uniform(-span_x, span_x)) if span_x > 0 else 0.0
        cy = float(rng.uniform(-span_y, span_y)) if span_y > 0 else 0.0
        cz = -float(rng.uniform(depth * 0.10, depth * 0.90))

        for (px, py, pz), n in shape:
            verts.extend([px + cx, py + cy, pz + cz, *n, *colour])
    return np.array(verts, dtype="f4")


def _quad(verts, p0, p1, p2, p3, normal, colour):
    """Two triangles, flat-shaded. Winding is counter-clockwise seen from the
    side the normal points to."""
    for p in (p0, p1, p2, p0, p2, p3):
        verts.extend([*p, *normal, *colour])


def _shaft_geometry(screen_w: float, screen_h: float, depth: float,
                    steps: int = 14) -> np.ndarray:
    """"The Shaft" — a stepped rectangular well sinking into the screen.

    This is the scene the whole project was aimed at, and the reason it works
    where the grid room does not is **occlusion**. Wireframe lines pass through
    each other and tell the eye almost nothing; solid stepped ledges hide the
    ones behind them, and occlusion is the strongest depth cue human vision has
    — strong enough to argue with stereo, which the grid room never could.

    The construction is a tube whose cross-section shrinks in discrete steps.
    Each step contributes two surfaces:

      * a *wall*, parallel to the view direction, which falls away into shadow
        and carries the linear perspective;
      * a *ledge*, square-on to the viewer, which catches the light and reads
        as a bright rectangular ring.

    That alternation of lit ring and dark wall, repeating into the distance, is
    what makes the reference image look like a hole rather than a picture. The
    mouth is exactly the window rectangle at Z = 0, so the illusion starts
    precisely at the glass.
    """
    verts: list[float] = []
    half_w, half_h = screen_w * 0.5, screen_h * 0.5

    # Each step shrinks the opening a little and moves it back. Shrinking by a
    # constant *fraction* rather than a constant amount keeps the steps looking
    # even as they recede, because perspective already shrinks them.
    shrink = 0.90
    step_z = depth / steps

    wall = (0.62, 0.64, 0.70)     # matte grey, slightly cool
    ledge = (0.88, 0.89, 0.93)    # brighter, so the rings read as lit surfaces

    w, h = half_w, half_h
    z = 0.0
    for _ in range(steps):
        z_next = z - step_z
        nw, nh = w * shrink, h * shrink

        # --- the four walls of this segment, normals pointing inward ---
        _quad(verts, (-w, -h, z), (w, -h, z), (w, -h, z_next), (-w, -h, z_next),
              (0.0, 1.0, 0.0), wall)                                   # floor
        _quad(verts, (w, h, z), (-w, h, z), (-w, h, z_next), (w, h, z_next),
              (0.0, -1.0, 0.0), wall)                                  # ceiling
        _quad(verts, (-w, h, z), (-w, -h, z), (-w, -h, z_next), (-w, h, z_next),
              (1.0, 0.0, 0.0), wall)                                   # left
        _quad(verts, (w, -h, z), (w, h, z), (w, h, z_next), (w, -h, z_next),
              (-1.0, 0.0, 0.0), wall)                                  # right

        # --- the ledge: a flat rectangular ring facing the viewer ---
        # Four strips between the old opening and the smaller new one.
        n = (0.0, 0.0, 1.0)
        _quad(verts, (-w, -h, z_next), (w, -h, z_next),
              (nw, -nh, z_next), (-nw, -nh, z_next), n, ledge)         # bottom
        _quad(verts, (w, h, z_next), (-w, h, z_next),
              (-nw, nh, z_next), (nw, nh, z_next), n, ledge)           # top
        _quad(verts, (-w, h, z_next), (-w, -h, z_next),
              (-nw, -nh, z_next), (-nw, nh, z_next), n, ledge)         # left
        _quad(verts, (w, -h, z_next), (w, h, z_next),
              (nw, nh, z_next), (nw, -nh, z_next), n, ledge)           # right

        w, h, z = nw, nh, z_next

    # Cap the far end, so the shaft bottoms out in darkness rather than showing
    # a hole through to the clear colour.
    _quad(verts, (-w, -h, z), (w, -h, z), (w, h, z), (-w, h, z),
          (0.0, 0.0, 1.0), (0.10, 0.10, 0.13))
    return np.array(verts, dtype="f4")


def _depth_card_geometry(screen_w: float, screen_h: float,
                         depth: float) -> np.ndarray:
    """"Depth Cards" — flat quads parked at distinct depths, overlapping.

    The plainest possible demonstration of differential parallax. Because the
    cards are flat and face-on, there is nothing to see *except* how they slide
    across one another, and their occlusion order visibly changes as the eye
    moves. Deliberately austere: any shading or perspective on the cards
    themselves would give the game away as ordinary 3D.
    """
    rng = np.random.default_rng(11)
    palette = [(0.98, 0.62, 0.48), (0.62, 0.84, 0.58), (0.98, 0.86, 0.46),
               (0.98, 0.76, 0.82), (0.58, 0.82, 0.94), (0.76, 0.68, 0.94),
               (0.55, 0.90, 0.82)]
    verts: list[float] = []
    for i, colour in enumerate(palette):
        z = -(depth * 0.06 + i * depth * 0.13)
        cw = screen_w * float(rng.uniform(0.26, 0.40))
        ch = screen_h * float(rng.uniform(0.30, 0.46))
        cx = float(rng.uniform(-screen_w * 0.22, screen_w * 0.22))
        cy = float(rng.uniform(-screen_h * 0.20, screen_h * 0.20))
        _quad(verts,
              (cx - cw, cy - ch, z), (cx + cw, cy - ch, z),
              (cx + cw, cy + ch, z), (cx - cw, cy + ch, z),
              (0.0, 0.0, 1.0), colour)
    return np.array(verts, dtype="f4")


def _aquarium_geometry(screen_w: float, screen_h: float,
                       depth: float) -> np.ndarray:
    """"The Aquarium" — a closed box with drifting motes, and one element that
    pokes *out* through the glass.

    The protruding piece is in the spec for a reason, and it is a negative
    demonstration: content in front of the screen plane gets clipped by the
    physical bezel the moment it reaches the edge of the panel, and the illusion
    tears. Kept small and centred so the failure is visible without being
    constant. Everything believable lives behind the glass.
    """
    rng = np.random.default_rng(29)
    verts: list[float] = []
    half_w, half_h = screen_w * 0.5, screen_h * 0.5
    back = -depth * 0.85

    # The tank: five inward-facing walls in a deep blue-green.
    tint = (0.14, 0.30, 0.34)
    _quad(verts, (-half_w, -half_h, back), (half_w, -half_h, back),
          (half_w, half_h, back), (-half_w, half_h, back),
          (0.0, 0.0, 1.0), (0.07, 0.16, 0.20))
    _quad(verts, (-half_w, -half_h, 0.0), (half_w, -half_h, 0.0),
          (half_w, -half_h, back), (-half_w, -half_h, back),
          (0.0, 1.0, 0.0), tint)
    _quad(verts, (half_w, half_h, 0.0), (-half_w, half_h, 0.0),
          (-half_w, half_h, back), (half_w, half_h, back),
          (0.0, -1.0, 0.0), tint)
    _quad(verts, (-half_w, half_h, 0.0), (-half_w, -half_h, 0.0),
          (-half_w, -half_h, back), (-half_w, half_h, back),
          (1.0, 0.0, 0.0), tint)
    _quad(verts, (half_w, -half_h, 0.0), (half_w, half_h, 0.0),
          (half_w, half_h, back), (half_w, -half_h, back),
          (-1.0, 0.0, 0.0), tint)

    # Motes drifting inside.
    for _ in range(70):
        s = float(rng.uniform(0.008, 0.022)) * screen_w
        cx = float(rng.uniform(-half_w + s * 2, half_w - s * 2))
        cy = float(rng.uniform(-half_h + s * 2, half_h - s * 2))
        cz = float(rng.uniform(back + s * 2, -s * 2))
        glow = float(rng.uniform(0.6, 1.0))
        colour = (0.45 * glow, 0.92 * glow, 0.86 * glow)
        for (px, py, pz), n in _octahedron(s):
            verts.extend([px + cx, py + cy, pz + cz, *n, *colour])

    # The one thing in front of the glass. +30 mm, small, centred.
    for (px, py, pz), n in _cube(screen_w * 0.10):
        verts.extend([px, py, pz + 30.0, *n, 0.98, 0.72, 0.35])
    return np.array(verts, dtype="f4")


class SolidScene:
    """A scene made only of triangles, sharing the solid shader.

    The shaft, the cards and the aquarium are all just vertex buffers — the
    projection, the window geometry and the tracker know nothing about which
    one is loaded, which is what makes switching them a keypress.
    """

    def __init__(self, ctx, name: str, description: str, program,
                 vertices: np.ndarray) -> None:
        self.ctx = ctx
        self.name = name
        self.description = description
        self.prog = program
        self.vbo = ctx.buffer(vertices.tobytes())
        self.vao = ctx.vertex_array(
            program, [(self.vbo, "3f 3f 3f", "in_pos", "in_normal", "in_colour")])
        self.count = len(vertices) // 9

    def cycle_density(self) -> int:      # only the grid room has one
        return 0

    def draw(self, projection: np.ndarray) -> None:
        import moderngl

        packed = np.ascontiguousarray(projection.T, dtype="f4").tobytes()
        self.prog["mvp"].write(packed)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.vao.render(moderngl.TRIANGLES, vertices=self.count)

    def release(self) -> None:
        self.vao.release()
        self.vbo.release()


class GridRoom:
    """Geometry plus the two shader programs. Knows nothing about projection —
    it is handed a matrix and draws with it, which is what lets step 5 change
    the projection without touching any of this."""

    name = "grid room"
    description = "wireframe box with floating shapes"

    def __init__(self, ctx, screen_w: float, screen_h: float,
                 depth: float = ROOM_DEPTH_MM) -> None:
        import moderngl

        self.ctx = ctx
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.depth = depth
        self.density_index = 1

        self.line_prog = ctx.program(vertex_shader=LINE_VERTEX_SHADER,
                                     fragment_shader=LINE_FRAGMENT_SHADER)
        self.solid_prog = ctx.program(vertex_shader=SOLID_VERTEX_SHADER,
                                      fragment_shader=SOLID_FRAGMENT_SHADER)
        self.line_prog["u_depth"] = depth
        self.solid_prog["u_depth"] = depth
        # The grid room keeps its shapes legible to the back rather than fading
        # them out; there would otherwise be nothing left to watch move.
        self.solid_prog["u_fade"] = 0.45
        self.solid_prog["u_ambient"] = 0.38

        self.line_vaos = []
        for divisions in GRID_DENSITIES:
            data = _grid_lines(divisions, screen_w, screen_h, depth)
            vbo = ctx.buffer(data.tobytes())
            self.line_vaos.append((
                ctx.vertex_array(self.line_prog,
                                 [(vbo, "3f 3f", "in_pos", "in_colour")]),
                len(data) // 6))

        solids = _solid_shapes(screen_w, screen_h, depth)
        vbo = ctx.buffer(solids.tobytes())
        self.solid_vao = ctx.vertex_array(
            self.solid_prog, [(vbo, "3f 3f 3f", "in_pos", "in_normal", "in_colour")])
        self.n_solid_verts = len(solids) // 9

    def cycle_density(self) -> int:
        self.density_index = (self.density_index + 1) % len(GRID_DENSITIES)
        return GRID_DENSITIES[self.density_index]

    def draw(self, projection: np.ndarray) -> None:
        import moderngl

        # numpy is row-major, GLSL wants column-major, hence the transpose.
        packed = np.ascontiguousarray(projection.T, dtype="f4").tobytes()
        self.solid_prog["mvp"].write(packed)
        self.line_prog["mvp"].write(packed)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.solid_vao.render(moderngl.TRIANGLES, vertices=self.n_solid_verts)

        vao, count = self.line_vaos[self.density_index]
        vao.render(moderngl.LINES, vertices=count)


VIGNETTE_VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos;
    gl_Position = vec4(in_pos * 2.0 - 1.0, 0.0, 1.0);
}
"""

VIGNETTE_FRAGMENT_SHADER = """
#version 330
uniform float u_strength;
in vec2 v_uv;
out vec4 f_colour;
void main() {
    // Darkening that hugs the border rather than a soft radial gradient. The
    // point is not "cinematic vignette" — it is to imply a casing around the
    // opening, so the panel bezel reads as the frame of a window rather than
    // as the edge of a picture. Measurably strengthens the effect, and costs
    // one pass.
    vec2 d = min(v_uv, 1.0 - v_uv) * 2.0;      // 0 at the edge, 1 at the centre
    float edge = clamp(min(d.x, d.y) / 0.16, 0.0, 1.0);
    float shade = 1.0 - (1.0 - edge * edge) * u_strength;
    f_colour = vec4(0.0, 0.0, 0.0, 1.0 - shade);
}
"""


class Vignette:
    """A dark border drawn over the finished frame."""

    def __init__(self, ctx, strength: float) -> None:
        self.ctx = ctx
        self.strength = strength
        self.prog = ctx.program(vertex_shader=VIGNETTE_VERTEX_SHADER,
                                fragment_shader=VIGNETTE_FRAGMENT_SHADER)
        quad = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1], dtype="f4")
        self.vao = ctx.vertex_array(self.prog,
                                    [(ctx.buffer(quad.tobytes()), "2f", "in_pos")])

    def draw(self) -> None:
        import moderngl

        if self.strength <= 0.0:
            return
        self.prog["u_strength"] = self.strength
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.vao.render(moderngl.TRIANGLES)
        self.ctx.disable(moderngl.BLEND)


def build_scenes(ctx, screen_w: float, screen_h: float) -> list:
    """Every scene, in the order the number keys select them.

    The shaft is first because it is by a distance the most convincing, and the
    reason is worth stating: it is the only one built from solid surfaces that
    occlude each other. Occlusion is the strongest depth cue the visual system
    has, and on a flat panel viewed with two eyes it is the only one powerful
    enough to argue with stereo disparity. The grid room, which was the whole of
    step 4 through 6, gives the eye almost nothing but linear perspective.
    """
    solid_prog = ctx.program(vertex_shader=SOLID_VERTEX_SHADER,
                             fragment_shader=SOLID_FRAGMENT_SHADER)

    def make(name, description, vertices, depth, fade, ambient):
        # Each scene wants its own depth and falloff, so they are set when the
        # scene is drawn rather than baked into the shader.
        scene = SolidScene(ctx, name, description, solid_prog, vertices)
        scene.depth, scene.fade, scene.ambient = depth, fade, ambient
        return scene

    shaft_depth = 1500.0
    scenes = [
        make("the shaft", "stepped well sinking into the screen",
             _shaft_geometry(screen_w, screen_h, shaft_depth),
             shaft_depth, 1.0, 0.30),
        GridRoom(ctx, screen_w, screen_h),
        make("depth cards", "flat quads at distinct depths",
             _depth_card_geometry(screen_w, screen_h, 1200.0),
             1200.0, 0.15, 1.0),
        make("the aquarium", "closed tank, one element in front of the glass",
             _aquarium_geometry(screen_w, screen_h, 1200.0),
             1200.0, 0.75, 0.35),
    ]
    return scenes


def draw_scene(scene, projection: np.ndarray) -> None:
    """Apply the scene's own depth and falloff, then draw it."""
    if isinstance(scene, SolidScene):
        scene.prog["u_depth"] = scene.depth
        scene.prog["u_fade"] = scene.fade
        scene.prog["u_ambient"] = scene.ambient
    scene.draw(projection)


# ----------------------------------------------------------------------------
# Debug HUD
# ----------------------------------------------------------------------------

HUD_VERTEX_SHADER = """
#version 330
uniform vec2 origin;   // top-left of the panel, as a fraction of the window
uniform vec2 extent;   // its size, likewise
in vec2 in_pos;        // unit quad, (0,0) at its top-left
out vec2 v_uv;
void main() {
    // The texture is written straight from an OpenCV image, whose first row is
    // the top, so v runs the same way as the quad's y and needs no flip.
    v_uv = in_pos;
    vec2 frac = origin + in_pos * extent;
    gl_Position = vec4(frac.x * 2.0 - 1.0, 1.0 - frac.y * 2.0, 0.0, 1.0);
}
"""

HUD_FRAGMENT_SHADER = """
#version 330
uniform sampler2D panel;
in vec2 v_uv;
out vec4 f_colour;
void main() {
    vec4 c = texture(panel, v_uv);
    // Pure black is the HUD's background and reads as transparent, so the
    // scene shows through everywhere nothing was drawn.
    float a = clamp(max(max(c.r, c.g), c.b) * 3.0, 0.0, 1.0);
    f_colour = vec4(c.rgb, a * 0.92);
}
"""

HUD_W, HUD_H = 470, 580
HUD_REFRESH_HZ = 12.0


class DebugHud:
    """An overlay drawn with OpenCV and uploaded as a texture.

    Drawing text through pyglet in the middle of a ModernGL frame means fighting
    over GL state. Rendering the panel into a numpy image with the same cv2
    calls used in steps 2 and 3, then pasting it on as a textured quad, keeps
    the two libraries entirely out of each other's way — and reuses drawing
    code that has already been looked at and corrected.

    Rebuilt at 12 Hz rather than every frame. At ~900 fps the cv2 work would
    otherwise dominate the frame time, and a readout that updates faster than
    the eye can follow is worse than useless anyway.
    """

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.visible = False
        self.prog = ctx.program(vertex_shader=HUD_VERTEX_SHADER,
                                fragment_shader=HUD_FRAGMENT_SHADER)
        quad = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1], dtype="f4")
        self.vao = ctx.vertex_array(self.prog, [(ctx.buffer(quad.tobytes()),
                                                 "2f", "in_pos")])
        self.texture = ctx.texture((HUD_W, HUD_H), 3)
        self.texture.repeat_x = self.texture.repeat_y = False
        self._last_build = 0.0

    def _build(self, sample, tracker, history, extents, render_fps,
               worst_frame_ms, eye_source, off_axis) -> np.ndarray:
        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX
        img = np.zeros((HUD_H, HUD_W, 3), np.uint8)
        good = (130, 245, 150)
        dim = (150, 150, 162)
        warn = (110, 175, 255)

        cv2.putText(img, "debug", (14, 26), font, 0.55, (215, 215, 225), 1)
        cv2.putText(img, "OFF-AXIS" if off_axis else "SYMMETRIC (wrong on purpose)",
                    (90, 26), font, 0.45, good if off_axis else warn, 1)
        cv2.line(img, (14, 38), (HUD_W - 14, 38), (60, 60, 70), 1)

        colour = good if sample.valid else (95, 95, 118)
        y = 66
        for name, value, unit in (("X", sample.x / 10, "cm"),
                                  ("Y", sample.y / 10, "cm"),
                                  ("Z", sample.z / 10, "cm")):
            cv2.putText(img, name, (14, y), font, 0.6, (175, 175, 185), 1)
            cv2.putText(img, f"{value:+7.1f} {unit}", (44, y), font, 0.6, colour, 1)
            y += 28

        y += 6
        if tracker is not None and tracker.error:
            # A dead tracker must never look like a still head.
            cv2.putText(img, "TRACKER STOPPED", (14, y), font, 0.5, (90, 90, 255), 1)
            y += 20
            cv2.putText(img, tracker.error[:44], (14, y), font, 0.38,
                        (110, 110, 200), 1)
            y += 24
        else:
            source = "webcam" if eye_source == "head" else eye_source
            state = f"{source}, {sample.n_faces} face(s)" if sample.valid \
                else (f"{source}, NO FACE" if eye_source == "head" else source)
            cv2.putText(img, state, (14, y), font, 0.45,
                        colour if sample.valid else warn, 1)
            y += 26
            if tracker is not None:
                rate_colour = dim if tracker.capture_fps >= 20 else warn
                cv2.putText(img, f"camera {tracker.backend} "
                                 f"{tracker.capture_fps:.0f} fps delivered",
                            (14, y), font, 0.4, rate_colour, 1)
                y += 22

        # Latency. This is the number the spec cares most about, so it is stated
        # honestly: it covers the camera frame arriving to this frame being
        # drawn, and excludes the camera's own exposure and readout and the
        # display's response, neither of which the program can see.
        age_ms = (time.perf_counter() - sample.captured_at) * 1000.0 \
            if sample.captured_at else 0.0
        pipeline_ms = (sample.published_at - sample.captured_at) * 1000.0 \
            if sample.captured_at else 0.0
        latency_colour = good if age_ms < 50 else warn
        for text, c in (
            (f"eye sample age   {age_ms:6.1f} ms", latency_colour),
            (f"  capture->publish {pipeline_ms:5.1f} ms", dim),
            (f"  inference        {sample.inference_ms:5.1f} ms", dim),
            ("  (excludes camera exposure + display)", (105, 105, 118)),
        ):
            cv2.putText(img, text, (14, y), font, 0.42, c, 1)
            y += 20

        y += 8
        raw_j = jitter_mm(history, 0, filtered=False)
        filt_j = jitter_mm(history, 0, filtered=True)
        for text in (f"jitter X   {raw_j:5.2f} -> {filt_j:5.2f} mm/frame",
                     f"head speed {sample.speed:6.0f} mm/s",
                     f"tracker    {sample.tracker_fps:5.1f} fps",
                     f"render     {render_fps:6.0f} fps  (worst {worst_frame_ms:.1f} ms)"):
            cv2.putText(img, text, (14, y), font, 0.42, dim, 1)
            y += 20

        y += 8
        l, r, b, t, d = extents
        cv2.putText(img, "frustum", (14, y), font, 0.42, (150, 150, 160), 1)
        y += 20
        cv2.putText(img, f"l {l:+7.3f}   r {r:+7.3f}", (14, y), font, 0.42, dim, 1)
        y += 19
        cv2.putText(img, f"b {b:+7.3f}   t {t:+7.3f}", (14, y), font, 0.42, dim, 1)
        y += 19
        # Asymmetry is the single clearest sign the shear is actually happening.
        # 0% is a centred eye; the closer to 100%, the further off to one side.
        skew = abs(l + r) / max(r - l, 1e-9) * 100.0
        cv2.putText(img, f"shear {skew:5.1f}%", (14, y), font, 0.42, dim, 1)
        cv2.putText(img, f"eye-to-plane {d:6.1f} mm", (150, y), font, 0.42, dim, 1)
        y += 24

        if tracker is not None and tracker.focal_is_estimated:
            cv2.putText(img, "focal length NOT calibrated", (14, y), font,
                        0.42, warn, 1)
            y += 22

        # The webcam thumbnail, below everything rather than beside it — the
        # first attempt tucked it into the bottom-right corner and the frustum
        # readout ran straight through it.
        if sample.preview is not None and tracker is not None:
            tw, th = 180, 101
            ox, oy = 14, min(y + 6, HUD_H - th - 34)
            small = cv2.resize(sample.preview, (tw, th))
            small = cv2.flip(small, 1)   # mirrored, so you see yourself
            if sample.iris_px:
                for u, v in sample.iris_px:
                    sx = tw - u / max(tracker.frame_w, 1) * tw
                    sy = v / max(tracker.frame_h, 1) * th
                    cv2.circle(small, (int(sx), int(sy)), 2, (0, 255, 255), -1)
            img[oy:oy + th, ox:ox + tw] = small
            cv2.rectangle(img, (ox, oy), (ox + tw, oy + th), (70, 70, 80), 1)

        cv2.putText(img, "D hud   TAB projection   a sweep   r recentre   q quit",
                    (14, HUD_H - 14), font, 0.38, (105, 105, 118), 1)
        return img

    def draw(self, sample, tracker, history, extents, render_fps, worst_frame_ms,
             eye_source, off_axis, window_w: int, window_h: int) -> None:
        import cv2
        import moderngl

        if not self.visible:
            return
        now = time.perf_counter()
        if now - self._last_build > 1.0 / HUD_REFRESH_HZ:
            img = self._build(sample, tracker, history, extents, render_fps,
                              worst_frame_ms, eye_source, off_axis)
            self.texture.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes())
            self._last_build = now

        # Kept at its native pixel size wherever possible, so the text stays
        # crisp instead of being stretched by whatever shape the window is.
        margin = 16.0
        self.prog["origin"] = (margin / window_w, margin / window_h)
        self.prog["extent"] = (min(HUD_W / window_w, 0.9),
                               min(HUD_H / window_h, 0.9))

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.texture.use(0)
        self.prog["panel"] = 0
        self.vao.render(moderngl.TRIANGLES)
        self.ctx.disable(moderngl.BLEND)


# ----------------------------------------------------------------------------
# Headless render, for checking the result without a window
# ----------------------------------------------------------------------------


def project_to_pixels(matrix: np.ndarray, point, width: int, height: int):
    """Push a world point through a projection matrix to a pixel position."""
    clip = matrix @ np.array([*point, 1.0], dtype="f8")
    if abs(clip[3]) < 1e-12:
        return None
    ndc = clip[:3] / clip[3]
    return ((ndc[0] * 0.5 + 0.5) * width, (0.5 - ndc[1] * 0.5) * height)


def selftest() -> int:
    """The acceptance tests from the spec, plus the ones they imply."""
    print("\n" + "=" * 72)
    print("Off-axis projection self test")
    print("=" * 72)
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}" + (f"  — {detail}" if detail else ""))

    w, h = 346.0, 216.0
    near, far = 10.0, 10000.0
    pa = np.array([-w / 2, -h / 2, 0.0])
    pb = np.array([+w / 2, -h / 2, 0.0])
    pc = np.array([-w / 2, +h / 2, 0.0])
    pd = pb + pc - pa                      # the implied fourth corner
    corners = [pa, pb, pc, pd]

    # --- spec §5: a centred eye must give a symmetric frustum ---------------
    l, r, b, t, d = frustum_extents(pa, pb, pc, [0.0, 0.0, 600.0], near, far)
    check("centred eye gives a symmetric frustum",
          abs(l + r) < 1e-12 and abs(b + t) < 1e-12,
          f"l={l:.6f} r={r:.6f} b={b:.6f} t={t:.6f}")
    check("perpendicular distance is the eye's z", abs(d - 600.0) < 1e-12,
          f"d={d:.6f}")

    # --- spec §5: moving +X shears, and in the right direction --------------
    l2, r2, b2, t2, d2 = frustum_extents(pa, pb, pc, [80.0, 0.0, 600.0], near, far)
    check("moving the eye +X makes |l| > |r|", abs(l2) > abs(r2),
          f"l={l2:.4f} r={r2:.4f}")
    check("the frustum shears, it does not rotate",
          abs(d2 - 600.0) < 1e-12,
          "sideways movement leaves the eye-to-plane distance untouched")
    check("vertical extents unchanged by a sideways move",
          abs(b2 - b) < 1e-12 and abs(t2 - t) < 1e-12)

    # NOTE: the spec (§5) says "moving pe in +X must make |l| < |r|". That is
    # the wrong way round. Moving right puts the eye further from the left edge
    # of the screen and closer to the right, so the left extent grows and the
    # right one shrinks. The corner-lock test below is what actually proves the
    # projection is correct, and it passes — so the spec's inequality is a slip,
    # not a specification we are failing to meet.

    # --- spec §10 test 1: the corner-lock test, the definitive one ----------
    #
    # Whatever the eye does, a vertex sitting at a physical screen corner must
    # land on the same pixel. The corners are nailed to the world; if they
    # drift, the projection is wrong.
    px_w, px_h = 1920, 1200
    rng = np.random.default_rng(4)
    reference = None
    worst = 0.0
    for _ in range(200):
        eye = np.array([rng.uniform(-400, 400), rng.uniform(-300, 300),
                        rng.uniform(200, 1500)])
        matrix = off_axis_projection(pa, pb, pc, eye, near, far)
        pixels = [project_to_pixels(matrix, c, px_w, px_h) for c in corners]
        if reference is None:
            reference = pixels
            continue
        for got, want in zip(pixels, reference):
            worst = max(worst, math.hypot(got[0] - want[0], got[1] - want[1]))
    check("screen corners are pixel-locked over 200 eye positions", worst < 1e-6,
          f"worst drift {worst:.3e} px, on a {px_w}x{px_h} image")

    # And they must land exactly on the image corners, not merely agree.
    matrix = off_axis_projection(pa, pb, pc, [0.0, 0.0, 600.0], near, far)
    got = [project_to_pixels(matrix, c, px_w, px_h) for c in corners]
    want = [(0, px_h), (px_w, px_h), (0, 0), (px_w, 0)]
    ok = all(abs(g[0] - e[0]) < 1e-6 and abs(g[1] - e[1]) < 1e-6
             for g, e in zip(got, want))
    check("corners land on the image corners", ok,
          "bottom-left -> (0, H), top-right -> (W, 0)")

    # --- the symmetric frustum must FAIL that same test ---------------------
    #
    # Otherwise the test proves nothing — it has to be able to tell the two
    # projections apart, or it would pass on step 4's broken version too.
    matrix_a = symmetric_projection(pa, pb, pc, [0.0, 0.0, 600.0], near, far)
    matrix_b = symmetric_projection(pa, pb, pc, [80.0, 0.0, 600.0], near, far)
    drift = max(math.hypot(*(np.subtract(project_to_pixels(matrix_b, c, px_w, px_h),
                                         project_to_pixels(matrix_a, c, px_w, px_h))))
                for c in corners)
    check("the symmetric frustum fails corner-lock", drift > 100.0,
          f"corners drift {drift:.0f} px for an 80 mm eye move — this is step 4's flaw")

    # --- spec §10 test 2: differential parallax -----------------------------
    #
    # The analytic gain: a point at depth pz, seen from an eye at ez, projects
    # onto the glass at a position that moves by |pz|/(ez+|pz|) per millimetre
    # of sideways eye movement. A point ON the glass never moves; a point
    # infinitely far away tracks the eye one for one.
    #
    # NOTE: the spec says an object at Z=-100 must shift "substantially more"
    # than one at Z=-1000. That is backwards — Z=-100 is nearly touching the
    # glass and is therefore nearly nailed to it. Verified below against both
    # the closed form and the actual matrix.
    ez, move = 600.0, 100.0
    eye0 = np.array([0.0, 0.0, ez])
    eye1 = np.array([move, 0.0, ez])
    m0 = off_axis_projection(pa, pb, pc, eye0, near, far)
    m1 = off_axis_projection(pa, pb, pc, eye1, near, far)

    measured = {}
    for depth in (-100.0, -1000.0):
        p = np.array([0.0, 0.0, depth])
        x0 = project_to_pixels(m0, p, px_w, px_h)[0]
        x1 = project_to_pixels(m1, p, px_w, px_h)[0]
        measured[depth] = abs(x1 - x0) / (px_w / w)   # pixels back to mm

    def analytic(depth: float) -> float:
        return abs(depth) / (ez + abs(depth)) * move

    for depth in (-100.0, -1000.0):
        err = abs(measured[depth] - analytic(depth)) / analytic(depth) * 100
        check(f"parallax at z={depth:.0f} matches the closed form", err < 2.0,
              f"{measured[depth]:.3f} mm measured, {analytic(depth):.3f} predicted "
              f"({err:.3f}% off)")

    ratio = measured[-1000.0] / measured[-100.0]
    check("the deeper object shifts more", ratio > 4.0,
          f"{ratio:.3f}x, closed form says {analytic(-1000.)/analytic(-100.):.3f}x")

    # A point exactly on the glass must not move at all — the cleanest possible
    # statement of what "the screen is a window" means.
    on_glass = np.array([40.0, -20.0, 0.0])
    x0 = project_to_pixels(m0, on_glass, px_w, px_h)
    x1 = project_to_pixels(m1, on_glass, px_w, px_h)
    check("a point on the glass never moves",
          math.hypot(x1[0] - x0[0], x1[1] - x0[1]) < 1e-6,
          f"{math.hypot(x1[0]-x0[0], x1[1]-x0[1]):.3e} px")

    # --- an off-centre window (the floating-window case) --------------------
    #
    # Everything above used a window centred on the panel. A dragged window is
    # the case that actually matters, and it must behave identically.
    off_pa = pa + np.array([300.0, 150.0, 0.0])
    off_pb = pb + np.array([300.0, 150.0, 0.0])
    off_pc = pc + np.array([300.0, 150.0, 0.0])
    off_pd = off_pb + off_pc - off_pa
    worst = 0.0
    reference = None
    for _ in range(100):
        eye = np.array([rng.uniform(-400, 400), rng.uniform(-300, 300),
                        rng.uniform(250, 1200)])
        matrix = off_axis_projection(off_pa, off_pb, off_pc, eye, near, far)
        pixels = [project_to_pixels(matrix, c, px_w, px_h)
                  for c in (off_pa, off_pb, off_pc, off_pd)]
        if reference is None:
            reference = pixels
            continue
        for got, want in zip(pixels, reference):
            worst = max(worst, math.hypot(got[0] - want[0], got[1] - want[1]))
    check("an off-centre window is also corner-locked", worst < 1e-6,
          f"worst drift {worst:.3e} px for a window dragged 300 mm right, 150 mm up")

    # --- degenerate input ---------------------------------------------------
    m = off_axis_projection(pa, pb, pc, [0.0, 0.0, 0.0], near, far)
    check("an eye on the screen plane does not produce NaNs",
          bool(np.all(np.isfinite(m))))

    print()
    if failures:
        print(f"  {failures} check(s) failed.")
        return 1
    print("  All self tests passed.")
    return 0


def render_test(cfg: dict) -> int:
    import moderngl
    from PIL import Image

    geom = ScreenGeometry(cfg)
    near = float(cfg["render"]["near_mm"])
    far = float(cfg["render"]["far_mm"])
    samples = int(cfg["render"].get("msaa", 4))

    width, height = 960, 600
    ctx = moderngl.create_standalone_context(require=460)
    print(f"  {ctx.info['GL_RENDERER'].strip()}")

    colour = ctx.texture((width, height), 4, samples=samples)
    depth_buf = ctx.depth_texture((width, height), samples=samples)
    msaa = ctx.framebuffer(color_attachments=[colour], depth_attachment=depth_buf)
    resolved = ctx.simple_framebuffer((width, height))

    # A window of this pixel size, centred on the panel.
    win_x = (geom.native_px[0] - width) // 2
    win_y = (geom.native_px[1] - height) // 2
    pa, pb, pc = geom.window_rect_mm(win_x, win_y, width, height)
    print(f"  window rect: {np.linalg.norm(pb - pa):.1f} x "
          f"{np.linalg.norm(pc - pa):.1f} mm")

    scenes = build_scenes(ctx, float(np.linalg.norm(pb - pa)),
                          float(np.linalg.norm(pc - pa)))

    vignette = Vignette(ctx, float(cfg["render"].get("vignette", 0.35)))
    views = [
        ("centre", np.array([0.0, 0.0, DEFAULT_EYE_Z])),
        ("left", np.array([-110.0, 0.0, DEFAULT_EYE_Z])),
        ("above", np.array([0.0, 95.0, DEFAULT_EYE_Z])),
    ]
    for index, scene in enumerate(scenes):
        slug = scene.name.replace(" ", "-")
        for name, eye in views:
            msaa.use()
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            draw_scene(scene, off_axis_projection(pa, pb, pc, eye, near, far))
            vignette.draw()
            ctx.copy_framebuffer(resolved, msaa)
            img = Image.frombytes(
                "RGB", (width, height),
                resolved.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)
            img.save(HERE / f"render-test-{index + 1}-{slug}-{name}.png")
        print(f"  {index + 1}. {scene.name} — {scene.description}")

    print("\n  Compare the -centre and -left images for each scene.")
    return 0


# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid room, symmetric frustum.")
    parser.add_argument("--render-test", action="store_true",
                        help="render a few frames to PNG and exit, no window")
    parser.add_argument("--selftest", action="store_true",
                        help="check the projection maths, no window")
    parser.add_argument("--eye-source", choices=("head", "mouse"), default="head",
                        help="what drives the eye. 'mouse' is for recording — see "
                             "the note at the top of this file about why filming "
                             "this effect from a tripod does not work")
    parser.add_argument("--hud", action="store_true",
                        help="start with the debug HUD open")
    parser.add_argument("--grab", type=float, default=None, metavar="SECONDS",
                        help="save what is on screen to PNG after N seconds. The "
                             "output is gitignored on purpose: with the HUD open "
                             "it contains the webcam preview, which makes it "
                             "personal data. Do not commit it.")
    parser.add_argument("--seconds", type=float, default=None,
                        help="close automatically after N seconds (for benchmarking)")
    parser.add_argument("--no-vsync", action="store_true",
                        help="unlock the frame rate, to see how much headroom there is")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    import yaml

    set_dpi_aware()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    if args.render_test:
        return render_test(cfg)

    import moderngl
    import pyglet

    geom = ScreenGeometry(cfg)
    near = float(cfg["render"]["near_mm"])
    far = float(cfg["render"]["far_mm"])
    samples = int(cfg["render"].get("msaa", 4))
    vsync = bool(cfg["render"].get("vsync", True)) and not args.no_vsync

    # Default to a window with the same shape as the panel, so the room's mouth
    # is not distorted before anything else has had a chance to go wrong.
    win_w = int(geom.native_px[0] * 0.5)
    win_h = int(round(win_w * geom.panel_h_mm / geom.panel_w_mm))

    config = pyglet.gl.Config(major_version=4, minor_version=6, double_buffer=True,
                              depth_size=24, sample_buffers=1, samples=samples)
    try:
        window = pyglet.window.Window(width=win_w, height=win_h, config=config,
                                      resizable=True, vsync=vsync,
                                      caption="grid room - symmetric frustum")
    except pyglet.window.NoSuchConfigException:
        print(f"  {samples}x MSAA is not available; falling back to none.")
        config = pyglet.gl.Config(major_version=4, minor_version=6,
                                  double_buffer=True, depth_size=24)
        window = pyglet.window.Window(width=win_w, height=win_h, config=config,
                                      resizable=True, vsync=vsync,
                                      caption="grid room - symmetric frustum")

    ctx = moderngl.create_context()
    print(f"  {ctx.info['GL_RENDERER'].strip()}  |  GL {ctx.info['GL_VERSION'].strip()}")
    print(f"  {geom.px_per_mm:.3f} px/mm from a {geom.panel_w_mm:g} mm wide panel")

    state = {
        "eye": np.array([0.0, 0.0, DEFAULT_EYE_Z]),
        "scenes": None,
        "scene_index": 0,
        "room_size": None,
        "frames": 0,
        "last_report": time.perf_counter(),
        "fps": 0.0,
        # Separate totals, because the rolling counter above is reset twice a
        # second for the title bar and cannot give an honest overall average.
        "total_frames": 0,
        "started": time.perf_counter(),
        "worst_frame_ms": 0.0,
        "last_frame_at": None,
        # Every frame interval, so the tail can be looked at properly. An
        # average hides stutter completely — 200 fps average with one 80 ms
        # hitch a second reads as broken and averages as excellent.
        "frame_gaps": [],
        "off_axis": True,
        "auto": False,
        "auto_t": 0.0,
        "source": args.eye_source,
        # Stands in for a tracker reading when the mouse is driving, so the HUD
        # has one shape of object to deal with either way.
        "fallback_sample": EyeSample(valid=True),
    }

    # ---- the tracker -------------------------------------------------------
    tracker = None
    if args.eye_source == "head":
        if cfg["tracking"].get("focal_px") is None:
            print("\n  focal_px is not calibrated, so the depth scale is a guess.")
            print('  Run:  python "calibration.py" --focal\n')
        tracker = EyeTracker(cfg)
        tracker.start()
        deadline = time.perf_counter() + 15.0
        while tracker.frame_w == 0 and tracker.error is None:
            if time.perf_counter() > deadline:
                tracker.stop()
                print("  Timed out waiting for the camera.")
                return 1
            time.sleep(0.05)
        if tracker.error:
            print(f"  Tracker failed to start: {tracker.error}")
            print("  Run with --eye-source mouse to use the projection anyway.")
            return 1
        print(f"  camera: {tracker.backend} at {tracker.frame_w}x{tracker.frame_h}")

    # How far the eye travels for a given mouse movement. At 1.0 the eye tracks
    # the mouse exactly — physically honest, but the pointer can only cross the
    # panel, so the eye could never get further than half a screen width off
    # centre. Real head movement is larger than that, so the default exaggerates
    # a little to make the effect easy to explore with a mouse.
    eye_gain = 2.0

    def eye_from_mouse(win_mouse_x: int, win_mouse_y: int) -> None:
        """Turn a pointer position into an eye position in screen-space mm.

        The same conversion the tracker will use, so the mouse is a stand-in for
        a head rather than a different kind of input. pyglet gives mouse
        coordinates relative to the window with Y up; the desktop measures Y
        down from the top, hence the flip before converting.
        """
        win_x, win_y = window.get_location()
        desktop_x = win_x + win_mouse_x
        desktop_y = win_y + (window.height - win_mouse_y)

        x_mm = (desktop_x - geom.native_px[0] * 0.5) / geom.px_per_mm
        y_mm = -(desktop_y - geom.native_px[1] * 0.5) / geom.px_per_mm
        state["eye"][0] = x_mm * eye_gain
        state["eye"][1] = y_mm * eye_gain

    def rebuild_room():
        """The room's mouth is the window, so resizing the window rebuilds it."""
        pa, pb, pc = geom.window_rect_mm(*window.get_location(),
                                         window.width, window.height)
        size = (float(np.linalg.norm(pb - pa)), float(np.linalg.norm(pc - pa)))
        if state["room_size"] != size:
            # Every scene is built against the window rectangle, so resizing
            # rebuilds all of them rather than just the visible one.
            state["scenes"] = build_scenes(ctx, size[0], size[1])
            state["room_size"] = size
        return pa, pb, pc

    rebuild_room()
    vignette = Vignette(ctx, float(cfg["render"].get("vignette", 0.35)))
    hud = DebugHud(ctx)
    hud.visible = args.hud

    @window.event
    def on_draw():
        pa, pb, pc = rebuild_room()

        # The render loop reads whatever is in the slot right now and never
        # waits for the camera. At ~900 render frames a second against 30
        # camera frames, the same eye position is reused for about thirty
        # consecutive frames, which is exactly right — the alternative is
        # stalling the renderer on a webcam.
        if tracker is not None and not state["auto"]:
            sample = tracker.latest
            # The camera updates 30 times a second; this loop draws 240 times.
            # Holding the last reading for eight frames makes the motion move in
            # visible 30 Hz steps even though every frame is delivered on time —
            # smooth frames, stepped movement, which reads as stutter. Carrying
            # the eye forward along its own filtered velocity turns those steps
            # back into continuous motion.
            #
            # Clamped, because if the tracker stalls this would otherwise sail
            # off across the room on the last velocity it saw.
            age = min(time.perf_counter() - sample.published_at, 0.10)
            state["eye"][0] = sample.x + sample.vel[0] * age
            state["eye"][1] = sample.y + sample.vel[1] * age
            state["eye"][2] = sample.z + sample.vel[2] * age
        else:
            sample = state["fallback_sample"]
            sample.x, sample.y, sample.z = state["eye"]

        ctx.screen.use()
        ctx.viewport = (0, 0, window.width, window.height)
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        build = off_axis_projection if state["off_axis"] else symmetric_projection
        scene = state["scenes"][state["scene_index"]]
        draw_scene(scene, build(pa, pb, pc, state["eye"], near, far))
        vignette.draw()

        hud.draw(sample, tracker,
                 tracker.history_snapshot() if tracker is not None else [],
                 frustum_extents(pa, pb, pc, state["eye"], near, far),
                 state["fps"], state["worst_frame_ms"], state["source"],
                 state["off_axis"], window.width, window.height)

        if state.get("grab_pending"):
            state["grab_pending"] = False
            from PIL import Image

            # The framebuffer's own size, not the window's — they disagree on a
            # scaled desktop and mid-resize.
            fb_w, fb_h = ctx.screen.size
            img = Image.frombytes("RGB", (fb_w, fb_h),
                                  ctx.screen.read(components=3))
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img.save(HERE / "screen-grab.png")
            print(f"  saved screen-grab.png ({fb_w}x{fb_h})")

        state["frames"] += 1
        state["total_frames"] += 1
        now = time.perf_counter()
        if state["last_frame_at"] is not None:
            # Ignore the first few frames — shader upload and the initial buffer
            # allocation make them meaninglessly slow.
            gap_ms = (now - state["last_frame_at"]) * 1000.0
            if state["total_frames"] > 10:
                state["worst_frame_ms"] = max(state["worst_frame_ms"], gap_ms)
                state["frame_gaps"].append(gap_ms)
        state["last_frame_at"] = now

        if now - state["last_report"] >= 0.5:
            state["fps"] = state["frames"] / (now - state["last_report"])
            state["frames"] = 0
            state["last_report"] = now
            eye = state["eye"]
            l, r, b, t, d = frustum_extents(pa, pb, pc, eye, near, far)
            mode = "OFF-AXIS" if state["off_axis"] else "symmetric (step 4)"
            window.set_caption(
                f"mouse parallax [{mode}] - {state['fps']:.0f} fps - "
                f"eye {eye[0]:+.0f} {eye[1]:+.0f} {eye[2]:.0f} mm - "
                f"l/r {l:+.2f}/{r:+.2f}")

    def mouse_drives() -> bool:
        return state["source"] == "mouse" and not state["auto"]

    @window.event
    def on_mouse_motion(x, y, dx, dy):
        if mouse_drives():
            eye_from_mouse(x, y)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if mouse_drives():
            eye_from_mouse(x, y)

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        if not mouse_drives():
            return
        # Scrolling moves the eye towards or away from the glass. Leaning in
        # should widen the view the way putting your face to a real window does.
        state["eye"][2] = float(np.clip(state["eye"][2] - scroll_y * 25.0,
                                        120.0, 2000.0))

    def advance_auto(dt):
        """A slow circular sweep of the eye. Useful for recording, and for
        watching the effect without a hand on the mouse."""
        if not state["auto"]:
            return
        state["auto_t"] += dt
        angle = state["auto_t"] * 0.9
        state["eye"][0] = math.cos(angle) * 170.0
        state["eye"][1] = math.sin(angle * 0.7) * 110.0

    @window.event
    def on_key_press(symbol, modifiers):
        keys = pyglet.window.key
        step = 20.0
        if symbol == keys.TAB:
            state["off_axis"] = not state["off_axis"]
            print("  " + ("off-axis frustum — the room opens up"
                          if state["off_axis"] else
                          "symmetric frustum — the room only slides"))
        elif symbol == keys.D:
            hud.visible = not hud.visible
        elif symbol == keys.M and tracker is not None:
            # Hand control back and forth without restarting, which matters
            # when something looks wrong and the question is whether it is the
            # tracker or the projection.
            state["source"] = "mouse" if state["source"] == "head" else "head"
            print(f"  eye source: {state['source']}")
        elif symbol == keys.A:
            state["auto"] = not state["auto"]
            print(f"  auto-sweep {'on' if state['auto'] else 'off'}")
        elif symbol == keys.LEFT:
            state["eye"][0] -= step
        elif symbol == keys.RIGHT:
            state["eye"][0] += step
        elif symbol == keys.UP:
            state["eye"][1] += step
        elif symbol == keys.DOWN:
            state["eye"][1] -= step
        elif symbol == keys.W:
            state["eye"][2] = max(120.0, state["eye"][2] - step)
        elif symbol == keys.S:
            state["eye"][2] = min(2000.0, state["eye"][2] + step)
        elif symbol == keys.R:
            state["eye"] = np.array([0.0, 0.0, DEFAULT_EYE_Z])
            state["auto"] = False
        elif symbol in (keys._1, keys._2, keys._3, keys._4):
            index = symbol - keys._1
            if index < len(state["scenes"]):
                state["scene_index"] = index
                scene = state["scenes"][index]
                print(f"  scene {index + 1}: {scene.name} — {scene.description}")
        elif symbol == keys.G:
            print(f"  grid divisions: {state['scenes'][state['scene_index']].cycle_density()}")
        elif symbol == keys.V:
            vignette.strength = 0.0 if vignette.strength > 0 else                 float(cfg["render"].get("vignette", 0.35))
            print(f"  vignette {'off' if vignette.strength == 0 else 'on'}")
        elif symbol == keys.F:
            window.set_fullscreen(not window.fullscreen)
        elif symbol in (keys.Q, keys.ESCAPE):
            window.close()

    # pyglet only redraws on demand by default, and this needs a continuous
    # loop. Scheduling a callback keeps on_draw firing every frame.
    pyglet.clock.schedule(advance_auto)

    if args.grab is not None:
        # Only ask for the grab here; the read itself happens at the end of
        # on_draw. A clock callback fires at an arbitrary point in the frame, so
        # reading the framebuffer from one captures whatever happens to be in
        # the back buffer — which is not the frame that was just drawn.
        pyglet.clock.schedule_once(
            lambda dt: state.__setitem__("grab_pending", True), args.grab)

    if args.seconds:
        pyglet.clock.schedule_once(lambda dt: window.close(), args.seconds)
        print(f"  benchmarking for {args.seconds:g} s...")
    elif state["source"] == "head":
        print("\n  Move your head. Lean left and more of the right-hand wall")
        print("  should come into view, as through a real window.")
        print("  D opens the debug HUD. TAB switches the projection.")
        print("  M hands control to the mouse and back.\n")
    else:
        print("\n  Move the mouse. The room should open rather than slide —")
        print("  go left and more of the right-hand wall comes into view.")
        print("  Press TAB to switch to step 4's symmetric frustum and back.")
        print("  Press 'a' for a hands-off sweep.\n")

    # Not pyglet's default of 1/60, which would throw away most of a 240 Hz
    # panel's latency advantage, and not uncapped either — step 4 found vsync
    # does not block in windowed mode, so uncapped means ~900 fps of frames the
    # compositor mostly discards. The rate comes from config so the choice is
    # explicit.
    max_fps = float(cfg["render"].get("max_fps", 240) or 0)
    interval = (1.0 / max_fps) if max_fps > 0 else 0.0

    # Only start counting now. Bringing up MediaPipe and the camera takes
    # several seconds, and including that in the average makes the frame rate
    # read far lower than it is.
    state["started"] = time.perf_counter()
    state["total_frames"] = 0

    try:
        pyglet.app.run(interval=interval)
    finally:
        if tracker is not None:
            tracker.stop()
            tracker.join(timeout=2.0)

    elapsed = time.perf_counter() - state["started"]
    average = state["total_frames"] / elapsed if elapsed > 0 else 0.0
    print(f"\n  {state['total_frames']} frames in {elapsed:.1f} s = {average:.1f} fps average")
    print(f"  worst single frame: {state['worst_frame_ms']:.1f} ms")
    gaps = np.array(state["frame_gaps"]) if state["frame_gaps"] else np.array([0.0])
    budget = 1000.0 / 60.0
    print(f"  frame interval: median {np.median(gaps):5.2f} ms   "
          f"p95 {np.percentile(gaps, 95):5.2f}   p99 {np.percentile(gaps, 99):5.2f}   "
          f"max {gaps.max():5.2f}")
    over = int((gaps > budget).sum())
    print(f"  frames slower than 60 fps ({budget:.1f} ms): {over} "
          f"({100.0 * over / len(gaps):.2f}%)")
    if tracker is not None:
        last = tracker.latest
        print(f"  tracker: {last.tracker_fps:.1f} fps, "
              f"{last.inference_ms:.1f} ms inference, "
              f"camera delivered {tracker.capture_fps:.1f} fps over {tracker.backend}")
        if tracker.error:
            print(f"  TRACKER STOPPED: {tracker.error}")
    if average > 200:
        print("  (far above the 60 fps target — plenty of headroom for richer scenes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
