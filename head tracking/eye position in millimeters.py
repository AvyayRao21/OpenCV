"""
Step 2 of the parallax window: webcam pixels to an eye position in millimetres.

Still no 3D. This program does one job — work out where your eye is in physical
space relative to the centre of the screen — and then shows you that number in
every way it can, so the sign conventions can be checked by hand before
anything is built on top of them.

Check the signs before going any further. Lean to your right and X must go
positive. Sit up straight and Y must go positive. Lean back and Z must grow.
If any of those is backwards, flip the matching switch in config.yaml. A sign
error here is invisible in the code and completely obvious in the finished
illusion, where the scene slides the wrong way and the whole thing feels wrong
without it being clear why.

Capture and inference run on their own thread. The camera gives about 30 frames
a second, and the renderer that comes later needs 60 or more, so the two can
never share a loop — the render loop must never sit and wait for a camera
frame. The thread publishes its most recent result into a single slot and the
main loop reads whatever is there. Frames are never queued: a stale head
position is worse than no update at all.

Usage:

    python "eye position in millimeters.py"

    q / ESC   quit
    x         toggle flip_x        m         mirror the preview
    y         toggle flip_y        w         save the flip switches to config.yaml
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
MODEL_PATH = HERE / "models" / "face_landmarker.task"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

IRIS_LEFT = 468
IRIS_RIGHT = 473


# ----------------------------------------------------------------------------
# The geometry, same as step 1. Repeated here rather than imported, because each
# numbered program in this folder is meant to run on its own.
# ----------------------------------------------------------------------------


def iris_separation_px(landmarks, frame_w: int, frame_h: int):
    """Pupil separation in pixels, plus how far the head is turned.

    Turning your head squashes the pupil gap the camera sees, by exactly
    cos(yaw). Uncorrected, every glance sideways reads as a lurch backwards.

    MediaPipe gives a z for each landmark on roughly the same scale as x, so the
    3D distance between the irises is the separation with the foreshortening
    already removed. The ratio between the 3D and the flat 2D distance is
    therefore 1/cos(yaw) — and because cosine is even, whichever way MediaPipe's
    z axis points cannot matter.
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
    """Undo the foreshortening, but only up to a point. Past about 35 degrees an
    iris starts hiding behind the nose and its z is not worth trusting."""
    max_gain = 1.0 / math.cos(math.radians(max_yaw_deg))
    gain = ipd_3d / ipd_2d if ipd_2d > 1e-6 else 1.0
    return ipd_2d * min(gain, max_gain)


def fallback_focal_px(frame_w: int, hfov_deg: float = 60.0) -> float:
    return (frame_w * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)


# ----------------------------------------------------------------------------


@dataclass
class EyeSample:
    """One reading from the tracker. Position is in screen-space millimetres."""

    x: float = 0.0
    y: float = 0.0
    z: float = 600.0
    valid: bool = False
    ipd_px: float = 0.0
    yaw_deg: float = 0.0
    n_faces: int = 0
    timestamp: float = 0.0
    tracker_fps: float = 0.0
    inference_ms: float = 0.0
    preview: np.ndarray | None = field(default=None, repr=False)
    iris_px: tuple[tuple[float, float], tuple[float, float]] | None = None


class EyeTracker(threading.Thread):
    """Capture and inference on their own thread, publishing to a single slot.

    Deliberately not a queue. If the consumer is slower than the camera, the
    right thing to do is throw away everything but the newest reading — an old
    head position is not useful, it is actively wrong.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(daemon=True, name="eye-tracker")
        self.cfg = cfg
        self._lock = threading.Lock()
        self._latest = EyeSample()
        self._running = threading.Event()
        self._running.set()

        d = cfg["display"]
        t = cfg["tracking"]
        self.cam_offset = [float(v) for v in d["cam_offset_mm"]]
        self.ipd_mm = float(t["ipd_mm"])
        self.max_yaw = float(t.get("max_yaw_deg", 35.0))
        self.z_min, self.z_max = (float(v) for v in t.get("z_clamp_mm", [200.0, 1500.0]))
        self.ease_seconds = float(t.get("lost_face_ease_s", 0.6))

        # These are live-tunable from the keyboard, so they are plain attributes
        # rather than being read out of the config every frame.
        self.flip_x = bool(t.get("flip_x", False))
        self.flip_y = bool(t.get("flip_y", False))
        # Display only — the preview is mirrored so you see yourself the right
        # way round. Every measurement uses the raw frame, because the sign
        # conventions below assume it.
        self.mirror_preview = True

        self.focal_px: float | None = (
            float(t["focal_px"]) if t.get("focal_px") is not None else None
        )
        self.focal_is_estimated = self.focal_px is None

        self.frame_w = 0
        self.frame_h = 0
        self.backend = ""
        self.error: str | None = None

        # Where the eye goes when there is no face to look at: straight in front
        # of the middle of the screen, at a comfortable working distance.
        self.default_position = (0.0, 0.0, 600.0)

    # -- the slot ------------------------------------------------------------

    @property
    def latest(self) -> EyeSample:
        with self._lock:
            return self._latest

    def _publish(self, sample: EyeSample) -> None:
        with self._lock:
            self._latest = sample

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

    # -- face selection ------------------------------------------------------

    @staticmethod
    def _face_size(landmarks, frame_w: int, frame_h: int) -> float:
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        return (max(xs) - min(xs)) * frame_w * (max(ys) - min(ys)) * frame_h

    def _choose_face(self, faces: list, frame_w: int, frame_h: int, previous):
        """Pick which face to follow, and stay with it.

        Largest wins, but only by a margin. Without the margin, two people at
        similar distances make the tracker flip between them every few frames,
        which looks like the scene having a seizure.
        """
        sizes = [self._face_size(f, frame_w, frame_h) for f in faces]
        largest = int(np.argmax(sizes))
        if previous is None or len(faces) == 1:
            return largest, sizes[largest]

        # Whichever face is nearest to the one we were already following.
        best, best_dist = 0, float("inf")
        for i, f in enumerate(faces):
            u = (f[IRIS_LEFT].x + f[IRIS_RIGHT].x) * 0.5 * frame_w
            v = (f[IRIS_LEFT].y + f[IRIS_RIGHT].y) * 0.5 * frame_h
            dist = math.hypot(u - previous[0], v - previous[1])
            if dist < best_dist:
                best, best_dist = i, dist

        # Only switch if the other face is clearly bigger, not marginally.
        if sizes[largest] > sizes[best] * 1.25:
            return largest, sizes[largest]
        return best, sizes[best]

    # -- the loop ------------------------------------------------------------

    def run(self) -> None:
        try:
            cap = self._open_camera()
        except Exception as exc:  # noqa: BLE001 — surfaced to the main thread
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
        last_good: tuple[float, float, float] | None = None
        lost_since: float | None = None

        while self._running.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            now = time.perf_counter()
            frame_times.append(now)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = (
                (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                if len(frame_times) > 1 and frame_times[-1] > frame_times[0]
                else 0.0
            )

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
                    lm, self.frame_w, self.frame_h
                )
                previous_eye_px = (mid_u, mid_v)
                ipd = corrected_ipd_px(ipd_2d, ipd_3d, self.max_yaw)

                # Distance first. A known real size imaged at a known focal
                # length gives the distance directly.
                z_cam = self.focal_px * self.ipd_mm / max(ipd, 1e-6)
                z_cam = min(self.z_max, max(self.z_min, z_cam))

                # Then sideways and vertical, which both scale with distance:
                # the same offset in pixels means more millimetres the further
                # away you are.
                #
                # Negations, because both axes disagree with screen space:
                #   u grows right in the RAW camera image, which is a mirror of
                #   you, so moving to your own right makes u smaller.
                #   v grows downward, and screen Y grows up.
                x_cam = -(mid_u - cx) * z_cam / self.focal_px
                y_cam = -(mid_v - cy) * z_cam / self.focal_px
                if self.flip_x:
                    x_cam = -x_cam
                if self.flip_y:
                    y_cam = -y_cam

                # Finally shift from the lens to the centre of the panel. The
                # camera sits above the screen, so this is mostly a downward
                # correction. Leaving it out puts a permanent vertical bias in
                # the illusion.
                position = (
                    x_cam + self.cam_offset[0],
                    y_cam + self.cam_offset[1],
                    z_cam + self.cam_offset[2],
                )
                last_good = position
                lost_since = None

                iris_px = (
                    (lm[IRIS_LEFT].x * self.frame_w, lm[IRIS_LEFT].y * self.frame_h),
                    (lm[IRIS_RIGHT].x * self.frame_w, lm[IRIS_RIGHT].y * self.frame_h),
                )
                self._publish(EyeSample(*position, valid=True, ipd_px=ipd, yaw_deg=yaw,
                                        n_faces=len(faces), timestamp=now,
                                        tracker_fps=fps, inference_ms=inference_ms,
                                        preview=frame, iris_px=iris_px))
            else:
                # Face lost. Hold the last position, then ease back to centre —
                # never snap. A snap is far more noticeable than a slow drift,
                # and it happens exactly when you look away, which is when a
                # sudden jump is most likely to be seen out of the corner of
                # your eye.
                if lost_since is None:
                    lost_since = now
                if last_good is None:
                    position = self.default_position
                else:
                    t = min(1.0, (now - lost_since) / max(self.ease_seconds, 1e-3))
                    ease = t * t * (3.0 - 2.0 * t)  # smoothstep: no sudden start or stop
                    position = tuple(
                        a + (b - a) * ease
                        for a, b in zip(last_good, self.default_position)
                    )
                self._publish(EyeSample(*position, valid=False, n_faces=0,
                                        timestamp=now, tracker_fps=fps,
                                        inference_ms=inference_ms, preview=frame))

        cap.release()


# ----------------------------------------------------------------------------
# The overlay
# ----------------------------------------------------------------------------

PANEL_W, PANEL_H = 1180, 680
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_front_view(canvas, origin, size, sample, screen_w, screen_h) -> None:
    """The screen seen from where you are sitting, with a crosshair on your eye.

    This is the picture that makes a sign error obvious: lean right, and the
    dot has to move right.
    """
    ox, oy = origin
    w, h = size
    cv2.rectangle(canvas, (ox, oy), (ox + w, oy + h), (55, 55, 62), 1)
    cv2.putText(canvas, "FRONT  (the screen, as you see it)", (ox, oy - 10),
                FONT, 0.5, (150, 150, 160), 1)

    # A generous margin, so an eye well outside the screen still has somewhere
    # to be drawn.
    span_x = screen_w * 1.9
    span_y = screen_h * 1.9

    def to_px(mm_x, mm_y):
        return (int(ox + w * 0.5 + mm_x / span_x * w),
                int(oy + h * 0.5 - mm_y / span_y * h))

    # The panel itself.
    tl = to_px(-screen_w / 2, screen_h / 2)
    br = to_px(screen_w / 2, -screen_h / 2)
    cv2.rectangle(canvas, tl, br, (90, 90, 100), 1)
    cv2.line(canvas, to_px(-span_x / 2, 0), to_px(span_x / 2, 0), (40, 40, 46), 1)
    cv2.line(canvas, to_px(0, -span_y / 2), to_px(0, span_y / 2), (40, 40, 46), 1)

    colour = (110, 235, 130) if sample.valid else (90, 90, 110)
    px, py = to_px(sample.x, sample.y)
    px = int(np.clip(px, ox + 2, ox + w - 2))
    py = int(np.clip(py, oy + 2, oy + h - 2))
    cv2.line(canvas, (px - 14, py), (px + 14, py), colour, 1)
    cv2.line(canvas, (px, py - 14), (px, py + 14), colour, 1)
    cv2.circle(canvas, (px, py), 6, colour, 2)

    cv2.putText(canvas, "+X ->", (ox + w - 60, oy + h // 2 - 8), FONT, 0.4,
                (80, 80, 90), 1)
    cv2.putText(canvas, "+Y ^", (ox + w // 2 + 6, oy + 16), FONT, 0.4,
                (80, 80, 90), 1)


def draw_top_view(canvas, origin, size, sample, screen_w, z_max) -> None:
    """The same thing from above, which is the only way to see Z change."""
    ox, oy = origin
    w, h = size
    cv2.rectangle(canvas, (ox, oy), (ox + w, oy + h), (55, 55, 62), 1)
    cv2.putText(canvas, "TOP  (looking down)", (ox, oy - 10), FONT, 0.5,
                (150, 150, 160), 1)

    span_x = screen_w * 1.9

    def to_px(mm_x, mm_z):
        return (int(ox + w * 0.5 + mm_x / span_x * w),
                int(oy + 12 + mm_z / z_max * (h - 24)))

    # The screen, edge on, at Z = 0 — drawn at the top because the viewer is
    # further down the panel as Z grows.
    cv2.line(canvas, to_px(-screen_w / 2, 0), to_px(screen_w / 2, 0),
             (140, 140, 155), 2)
    cv2.putText(canvas, "screen", (to_px(screen_w / 2, 0)[0] + 6,
                to_px(screen_w / 2, 0)[1] + 4), FONT, 0.38, (110, 110, 125), 1)

    for z in range(200, int(z_max) + 1, 200):
        y = to_px(0, z)[1]
        cv2.line(canvas, (ox + 2, y), (ox + w - 2, y), (36, 36, 42), 1)
        cv2.putText(canvas, f"{z // 10} cm", (ox + 6, y - 4), FONT, 0.35,
                    (70, 70, 80), 1)

    colour = (110, 235, 130) if sample.valid else (90, 90, 110)
    px, py = to_px(sample.x, min(sample.z, z_max))
    px = int(np.clip(px, ox + 2, ox + w - 2))
    py = int(np.clip(py, oy + 2, oy + h - 2))
    cv2.line(canvas, to_px(0, 0), (px, py), (55, 70, 60), 1)
    cv2.circle(canvas, (px, py), 6, colour, 2)


def draw_preview(canvas, origin, size, sample, tracker) -> None:
    """The webcam feed, small, with the two iris points marked."""
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
            sy = v / max(tracker.frame_h, 1) * h
            cv2.circle(small, (int(sx), int(sy)), 3, (0, 255, 255), -1)

    canvas[oy:oy + h, ox:ox + w] = small
    cv2.rectangle(canvas, (ox, oy), (ox + w, oy + h), (55, 55, 62), 1)
    label = "mirrored (as you see yourself)" if tracker.mirror_preview else "raw camera image"
    cv2.putText(canvas, label, (ox, oy - 10), FONT, 0.45, (150, 150, 160), 1)


def draw_readout(canvas, origin, sample, tracker) -> None:
    ox, oy = origin
    colour = (110, 235, 130) if sample.valid else (90, 90, 110)
    lines = [
        ("X", f"{sample.x / 10:+7.1f} cm", "lean right -> +"),
        ("Y", f"{sample.y / 10:+7.1f} cm", "sit up -> +"),
        ("Z", f"{sample.z / 10:7.1f} cm", "from the glass"),
    ]
    y = oy
    for name, value, hint in lines:
        cv2.putText(canvas, name, (ox, y), FONT, 0.7, (170, 170, 180), 2)
        cv2.putText(canvas, value, (ox + 30, y), FONT, 0.7, colour, 2)
        cv2.putText(canvas, hint, (ox + 178, y), FONT, 0.4, (110, 110, 122), 1)
        y += 34

    y += 14
    status = f"tracking, {sample.n_faces} face(s)" if sample.valid \
        else "NO FACE - easing to centre"
    cv2.putText(canvas, status, (ox, y), FONT, 0.5,
                colour if sample.valid else (100, 150, 235), 1)
    y += 28

    for text in (
        f"pupil gap  {sample.ipd_px:6.1f} px",
        f"head yaw   {sample.yaw_deg:6.1f} deg",
        f"tracker    {sample.tracker_fps:5.1f} fps",
        f"inference  {sample.inference_ms:5.1f} ms",
        f"flip_x {'ON ' if tracker.flip_x else 'off'}   flip_y {'ON ' if tracker.flip_y else 'off'}",
    ):
        cv2.putText(canvas, text, (ox, y), FONT, 0.46, (150, 150, 162), 1)
        y += 24


def draw_focal_state(canvas, origin, tracker) -> None:
    """The focal length deserves its own box. If it has not been calibrated then
    every distance on screen is a guess, and that should be impossible to miss
    rather than one dim line in a list."""
    ox, oy = origin
    if tracker.focal_is_estimated:
        cv2.rectangle(canvas, (ox, oy), (ox + 320, oy + 74), (40, 70, 120), 1)
        cv2.putText(canvas, "FOCAL LENGTH NOT CALIBRATED", (ox + 12, oy + 24),
                    FONT, 0.46, (110, 175, 255), 1)
        cv2.putText(canvas, f"guessing {tracker.focal_px:.0f} px from a 60 deg fov",
                    (ox + 12, oy + 46), FONT, 0.42, (140, 140, 155), 1)
        cv2.putText(canvas, "directions are right, the scale is not",
                    (ox + 12, oy + 64), FONT, 0.42, (140, 140, 155), 1)
    else:
        cv2.rectangle(canvas, (ox, oy), (ox + 320, oy + 44), (45, 70, 50), 1)
        cv2.putText(canvas, f"focal length {tracker.focal_px:.0f} px, calibrated",
                    (ox + 12, oy + 28), FONT, 0.46, (110, 200, 130), 1)


def draw_help(canvas, origin) -> None:
    ox, oy = origin
    cv2.putText(canvas, "controls", (ox, oy), FONT, 0.46, (150, 150, 160), 1)
    y = oy + 26
    for key, what in [
        ("x / y", "flip an axis"),
        ("w", "save the flips to config.yaml"),
        ("m", "mirror the preview"),
        ("q", "quit"),
    ]:
        cv2.putText(canvas, key, (ox, y), FONT, 0.45, (160, 160, 172), 1)
        cv2.putText(canvas, what, (ox + 60, y), FONT, 0.45, (95, 95, 108), 1)
        y += 22


# ----------------------------------------------------------------------------


def main() -> int:
    import yaml

    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    screen_w = float(cfg["display"]["screen_w_mm"])
    screen_h = float(cfg["display"]["screen_h_mm"])
    z_max = float(cfg["tracking"].get("z_clamp_mm", [200, 1500])[1])

    if cfg["tracking"].get("focal_px") is None:
        print()
        print("  " + "!" * 66)
        print("  focal_px is not calibrated, so distance is being guessed from an")
        print("  assumed 60 degree field of view. Webcam lenses vary a lot, so Z")
        print("  will be wrong by tens of percent and X and Y will be wrong with")
        print("  it. Directions and signs are still meaningful; the scale is not.")
        print("  Fix it with:  python \"calibration.py\" --focal")
        print("  " + "!" * 66)
        print()

    tracker = EyeTracker(cfg)
    tracker.start()

    # Give the thread a moment to open the camera and report any failure.
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
    print("Check the signs now — lean right, sit up, lean back.\n")

    window = "eye position (mm)"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    last_print = 0.0
    dirty = False

    try:
        while True:
            sample = tracker.latest
            canvas = np.full((PANEL_H, PANEL_W, 3), 18, np.uint8)

            cv2.putText(canvas, "eye position in screen-space millimetres",
                        (26, 34), FONT, 0.62, (210, 210, 220), 1)
            cv2.line(canvas, (26, 48), (PANEL_W - 26, 48), (45, 45, 52), 1)

            draw_readout(canvas, (30, 92), sample, tracker)
            draw_preview(canvas, (30, 380), (330, 186), sample, tracker)
            draw_front_view(canvas, (400, 92), (380, 240), sample, screen_w, screen_h)
            draw_top_view(canvas, (400, 380), (380, 240), sample, screen_w, z_max)

            # A written statement of what the numbers currently claim, so the
            # check does not depend on reading a diagram correctly.
            side = "RIGHT" if sample.x > 15 else "LEFT" if sample.x < -15 else "centred"
            vert = "ABOVE" if sample.y > 15 else "BELOW" if sample.y < -15 else "level with"
            cv2.putText(canvas, "the tracker thinks you are:", (830, 100), FONT,
                        0.5, (150, 150, 160), 1)
            for i, text in enumerate([
                f"{side} of centre",
                f"{vert} the middle",
                f"{sample.z / 10:.0f} cm back",
            ]):
                cv2.putText(canvas, text, (830, 136 + i * 30), FONT, 0.62,
                            (110, 235, 130) if sample.valid else (90, 90, 110), 1)

            draw_focal_state(canvas, (830, 250), tracker)
            draw_help(canvas, (830, 380))

            cv2.imshow(window, canvas)

            now = time.perf_counter()
            if now - last_print > 0.5:
                last_print = now
                state = "" if sample.valid else "   (no face)"
                print(f"\r  X {sample.x/10:+7.1f}   Y {sample.y/10:+7.1f}   "
                      f"Z {sample.z/10:7.1f} cm   {sample.tracker_fps:4.1f} fps{state}   ",
                      end="", flush=True)

            key = cv2.waitKey(8) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("x"):
                tracker.flip_x = not tracker.flip_x
                dirty = True
            if key == ord("y"):
                tracker.flip_y = not tracker.flip_y
                dirty = True
            if key == ord("m"):
                tracker.mirror_preview = not tracker.mirror_preview
            if key == ord("w"):
                text = CONFIG_PATH.read_text(encoding="utf-8")
                for k, v in (("flip_x", tracker.flip_x), ("flip_y", tracker.flip_y)):
                    text = _set_flag(text, k, v)
                CONFIG_PATH.write_text(text, encoding="utf-8", newline="")
                dirty = False
                print(f"\n  saved: flip_x={tracker.flip_x}  flip_y={tracker.flip_y}")

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        tracker.stop()
        tracker.join(timeout=2.0)
        cv2.destroyAllWindows()

    print()
    if dirty:
        print("  Flip switches were changed but not saved. Press w next time to keep them.")
    print("  Next: 'one euro filter.py'")
    return 0


def _set_flag(text: str, key: str, value: bool) -> str:
    """Set one boolean in the tracking section, leaving comments alone."""
    import re

    out = []
    in_section = False
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if re.match(r"^tracking\s*:", stripped):
            in_section = True
        elif in_section and stripped and not stripped[0].isspace() \
                and not stripped.lstrip().startswith("#"):
            in_section = False
        if in_section:
            m = re.match(rf"^(\s+){key}\s*:\s*\S+(\s*#.*)?$", stripped)
            if m:
                nl = "\n" if line.endswith("\n") else ""
                line = f"{m.group(1)}{key}: {'true' if value else 'false'}{m.group(2) or ''}{nl}"
        out.append(line)
    return "".join(out)


if __name__ == "__main__":
    sys.exit(main())
