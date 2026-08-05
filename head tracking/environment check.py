"""
Environment check for the head-tracking parallax window.

Run this first, and any time something stops working. It verifies every piece
the project depends on and prints a single PASS/FAIL line per item, so a broken
setup is obvious instead of showing up later as a confusing crash inside the
renderer.

    python "environment check.py"

It opens the webcam for about two seconds to confirm the face landmarks come
through. No image is ever saved to disk.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# MediaPipe pulls in TensorFlow, which prints oneDNN notices on import. They are
# harmless and they bury the output of this script, so quieten them first.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

results: list[tuple[bool, str, str]] = []


def check(name: str) -> "Check":
    return Check(name)


class Check:
    """Small helper so each check reads as a `with` block and can never crash
    the whole script — a failure is recorded and the rest still run."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.detail = ""

    def __enter__(self) -> "Check":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            results.append((True, self.name, self.detail))
        else:
            results.append((False, self.name, f"{type(exc).__name__}: {exc}"))
        return True  # swallow the exception, keep going


print("=" * 70)
print("Head-tracking parallax window — environment check")
print("=" * 70)

# ---------------------------------------------------------------- interpreter
with check("Python 3.11") as c:
    v = sys.version_info
    c.detail = f"{v.major}.{v.minor}.{v.micro} at {sys.executable}"
    if (v.major, v.minor) != (3, 11):
        raise RuntimeError(
            f"this project needs Python 3.11 (mediapipe has no wheel for "
            f"{v.major}.{v.minor} in this install); run with `python`, not `py`"
        )

# ------------------------------------------------------------------- packages
for mod, label in [
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("mediapipe", "MediaPipe"),
    ("moderngl", "ModernGL"),
    ("moderngl_window", "moderngl-window"),
    ("pyglet", "pyglet"),
    ("yaml", "PyYAML"),
]:
    with check(label) as c:
        m = __import__(mod)
        c.detail = getattr(m, "__version__", "version unknown")

# ---------------------------------------------------------------- model asset
with check("Face Landmarker model") as c:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"missing {MODEL_PATH}\n      download it from:\n      {MODEL_URL}"
        )
    size_mb = MODEL_PATH.stat().st_size / 1e6
    c.detail = f"{MODEL_PATH.name}, {size_mb:.2f} MB"
    if size_mb < 3.0:
        raise RuntimeError("file is too small — the download was truncated")

# --------------------------------------------------------------------- OpenGL
with check("OpenGL 4.6 context") as c:
    import moderngl

    ctx = moderngl.create_standalone_context(require=460)
    c.detail = (
        f"{ctx.info['GL_RENDERER'].strip()} | {ctx.info['GL_VERSION'].strip()} | "
        f"max MSAA {ctx.info.get('GL_MAX_SAMPLES', '?')}x"
    )
    ctx.release()

# ------------------------------------------------------------------ the webcam
with check("Webcam + iris landmarks") as c:
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    # CAP_DSHOW first: on Windows it opens faster than the MSMF default and is
    # the more reliable of the two with external USB webcams.
    cap = None
    for api, api_name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]:
        probe = cv2.VideoCapture(0, api)
        if probe.isOpened() and probe.read()[0]:
            cap, backend = probe, api_name
            break
        probe.release()
    if cap is None:
        raise RuntimeError("no camera on index 0")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
    )

    timings: list[float] = []
    hits = 0
    n_landmarks = 0
    start = time.perf_counter()
    for _ in range(45):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t0 = time.perf_counter()
        result = landmarker.detect_for_video(
            image, int((time.perf_counter() - start) * 1000)
        )
        timings.append((time.perf_counter() - t0) * 1000)
        if result.face_landmarks:
            hits += 1
            n_landmarks = len(result.face_landmarks[0])

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Skip the first few: the first inference includes one-off warm-up cost.
    median_ms = float(np.median(timings[3:])) if len(timings) > 3 else float("nan")
    c.detail = (
        f"{backend} {width}x{height} | face in {hits}/{len(timings)} frames | "
        f"{n_landmarks} landmarks | {median_ms:.1f} ms median inference"
    )
    if n_landmarks == 0:
        raise RuntimeError(
            "no face detected — sit in front of the camera and check the lighting"
        )
    if n_landmarks != 478:
        raise RuntimeError(
            f"got {n_landmarks} landmarks, expected 478. Fewer than 478 means the "
            "iris refinement is off, and the iris points are what give us depth."
        )

# --------------------------------------------------------------------- summary
print()
width = max(len(name) for _, name, _ in results)
for ok, name, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<{width}}  {detail}")

failed = [r for r in results if not r[0]]
print()
if failed:
    print(f"{len(failed)} check(s) failed. Fix those before going further.")
    sys.exit(1)
print("All checks passed. The environment is ready.")
