"""
Step 1 of the parallax window: configuration and calibration. No rendering.

Everything the illusion does is driven by physical measurements. Get them wrong
and the scene still moves, it just moves by the wrong amount, which reads as
"nearly right" and is maddening to debug later. So this program comes first and
writes them all down properly.

It walks through three things:

  1. The panel   — how big the glass is, in millimetres.
  2. The webcam  — where the lens sits relative to the centre of the panel.
  3. The lens    — the camera's focal length in pixels, which is what converts a
                   pupil separation on screen into a distance in millimetres.

Usage:

    python "calibration.py"              full guided walkthrough
    python "calibration.py" --focal      only redo the focal length
    python "calibration.py" --show       print the current config and exit
    python "calibration.py" --selftest   check the maths, no camera needed

Results are written back into config.yaml in place, so the comments and any
hand edits in that file survive.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
MODEL_PATH = HERE / "models" / "face_landmarker.task"

# MediaPipe drags in TensorFlow, which is chatty on import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

# Iris centres in the 478-point face mesh. These only exist when iris refinement
# is on, which is what makes the landmark count 478 rather than 468.
IRIS_LEFT = 468
IRIS_RIGHT = 473

# Outer eye corners. Fallback if the iris points are ever missing.
EYE_OUTER_LEFT = 33
EYE_OUTER_RIGHT = 263

FOCAL_SAMPLE_FRAMES = 90


# ----------------------------------------------------------------------------
# config.yaml read / write
# ----------------------------------------------------------------------------
#
# PyYAML can read this file happily but cannot write it back without destroying
# every comment in it, and the comments are half the value of that file. So
# reading uses PyYAML and writing is a surgical line edit that leaves the rest
# of the document exactly as it was.


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _format_value(value: Any) -> str:
    """Render a Python value the way it should appear in the YAML file."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    if isinstance(value, float):
        # Keep it readable: 345.0 rather than 345.00000000000006.
        return f"{value:g}" if value != int(value) else f"{value:.1f}"
    return str(value)


def set_config_value(text: str, section: str, key: str, value: Any) -> str:
    """Replace one `section: key: value` in the YAML text, in place.

    Preserves the line's inline comment, except that a "MEASURE ME" marker is
    dropped once the value has actually been measured.
    """
    lines = text.splitlines(keepends=True)
    in_section = False
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if re.match(rf"^{re.escape(section)}\s*:", stripped):
            in_section = True
            continue
        # A non-indented, non-blank, non-comment line ends the section.
        if in_section and stripped and not stripped[0].isspace() and not stripped.lstrip().startswith("#"):
            break
        if not in_section:
            continue

        m = re.match(rf"^(\s+){re.escape(key)}\s*:\s*(.*?)(\s*#.*)?$", stripped)
        if not m:
            continue
        indent, _old, comment = m.group(1), m.group(2), m.group(3) or ""
        if "MEASURE ME" in comment:
            comment = ""
        newline = "\n" if line.endswith("\n") else ""
        lines[i] = f"{indent}{key}: {_format_value(value)}{comment}{newline}"
        return "".join(lines)

    raise KeyError(f"could not find {section}.{key} in the config file")


def write_config_values(updates: list[tuple[str, str, Any]], path: Path = CONFIG_PATH) -> None:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    for section, key, value in updates:
        text = set_config_value(text, section, key, value)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


# ----------------------------------------------------------------------------
# The geometry
# ----------------------------------------------------------------------------


def iris_separation_px(
    landmarks: list, frame_w: int, frame_h: int
) -> tuple[float, float, float, float, float]:
    """Measure the pupil separation, and undo the foreshortening from head turn.

    Returns (ipd_2d_px, ipd_3d_px, yaw_deg, mid_u_px, mid_v_px).

    Turning your head sideways squashes the pupil separation you can see in the
    image: what the camera measures is the true separation times cos(yaw). Left
    uncorrected, every glance to the side reads as "the head lurched backwards",
    because a smaller pupil gap means further away.

    The correction here does not need a head-pose solve. MediaPipe already
    returns a z for every landmark, on roughly the same scale as x once you
    multiply by the image width. So the 3D distance between the two irises is
    the separation *without* the foreshortening, and the ratio of the 3D
    distance to the flat 2D one is exactly 1/cos(yaw).

    A pleasant side effect: cosine is an even function, so the sign convention
    of MediaPipe's z axis cannot bite us here.
    """
    left, right = landmarks[IRIS_LEFT], landmarks[IRIS_RIGHT]

    dx = (left.x - right.x) * frame_w
    dy = (left.y - right.y) * frame_h
    # MediaPipe normalises z on the same scale as x, so it scales by the width.
    dz = (left.z - right.z) * frame_w

    ipd_2d = math.hypot(dx, dy)
    ipd_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

    # cos(yaw) = (what we see) / (what is really there)
    cos_yaw = ipd_2d / ipd_3d if ipd_3d > 1e-6 else 1.0
    cos_yaw = min(1.0, max(-1.0, cos_yaw))
    yaw_deg = math.degrees(math.acos(cos_yaw))

    mid_u = (left.x + right.x) * 0.5 * frame_w
    mid_v = (left.y + right.y) * 0.5 * frame_h
    return ipd_2d, ipd_3d, yaw_deg, mid_u, mid_v


def corrected_ipd_px(ipd_2d: float, ipd_3d: float, max_yaw_deg: float) -> float:
    """Apply the yaw correction, clamped.

    Past roughly 35 degrees one iris starts to disappear behind the nose and the
    landmark z gets unreliable, so the correction is capped rather than trusted.
    """
    max_gain = 1.0 / math.cos(math.radians(max_yaw_deg))
    gain = ipd_3d / ipd_2d if ipd_2d > 1e-6 else 1.0
    return ipd_2d * min(gain, max_gain)


def solve_focal_px(ipd_px: float, ipd_mm: float, distance_mm: float) -> float:
    """Focal length in pixels, from one known viewing distance.

    A pinhole camera images something of real size S, at distance Z, as
    S * focal / Z pixels. Rearranged for focal, with S = the real pupil
    separation and Z = the measured distance from the lens to the eyes.
    """
    return ipd_px * distance_mm / ipd_mm


def depth_from_ipd(ipd_px: float, ipd_mm: float, focal_px: float) -> float:
    """The same relation, run the other way: pixels of pupil gap -> millimetres."""
    return focal_px * ipd_mm / ipd_px


def eye_to_screen_space(
    mid_u: float,
    mid_v: float,
    z_cam: float,
    frame_w: int,
    frame_h: int,
    focal_px: float,
    cam_offset_mm: list[float],
    flip_x: bool = False,
    flip_y: bool = False,
) -> tuple[float, float, float]:
    """Convert an eye position in image pixels into screen-space millimetres.

    Signs, spelled out, because getting one backwards inverts the whole illusion
    and it is not obvious from the symptom which one is wrong:

      * The image's u grows to the right *in the raw camera image*, which is a
        mirror of the viewer. Moving to your own right makes u smaller. Hence
        the negation on X.
      * The image's v grows downward. Screen Y grows up. Hence the negation on Y.
      * Z_cam is already a positive distance out towards the face, which is the
        direction screen-space +Z points, so it needs no sign change.

    The result is then shifted by the camera's offset, because the camera sits
    above the top bezel and the coordinate origin is the centre of the panel.
    Skipping that shift leaves a permanent vertical bias — subtle, and it ruins
    the effect.
    """
    cx, cy = frame_w * 0.5, frame_h * 0.5

    x_cam = -(mid_u - cx) * z_cam / focal_px
    y_cam = -(mid_v - cy) * z_cam / focal_px

    if flip_x:
        x_cam = -x_cam
    if flip_y:
        y_cam = -y_cam

    return (
        x_cam + cam_offset_mm[0],
        y_cam + cam_offset_mm[1],
        z_cam + cam_offset_mm[2],
    )


def fallback_focal_px(frame_w: int, hfov_deg: float = 60.0) -> float:
    """Rough focal length from an assumed field of view.

    Only a starting point. Webcam field of view varies enormously between
    models, so this is wrong by tens of percent and the depth will be wrong by
    the same factor.
    """
    return (frame_w * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------


def ask_float(prompt: str, default: float, unit: str = "mm") -> float:
    while True:
        raw = input(f"{prompt} [{default:g} {unit}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Not a number. Try again, or press Enter to keep the default.")


def ask_yes(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def rule(title: str = "") -> None:
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


# ----------------------------------------------------------------------------
# Stage 1 and 2 — the panel and the camera position
# ----------------------------------------------------------------------------


def detect_native_resolution() -> tuple[int, int] | None:
    """The panel's real pixel count, not the DPI-scaled desktop size.

    Windows lies to processes that have not declared themselves DPI aware: this
    laptop runs at 150% scaling, so an unaware process is told the desktop is
    1707x1067 when the panel is really 2560x1600. Using the scaled number would
    put the pixels-per-millimetre scale out by exactly that factor, and windowed
    mode would place the frustum's corners in the wrong physical place.

    Has to be called before any window exists, which is why it lives in the
    calibration step rather than the renderer.
    """
    if sys.platform != "win32":
        return None
    import ctypes

    try:
        # 2 = per-monitor DPI aware. Falls back to the older system-wide call on
        # anything before Windows 8.1.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            return None
    try:
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except (AttributeError, OSError):
        return None


def measure_panel(cfg: dict) -> list[tuple[str, str, Any]]:
    rule("1. The panel")
    print(
        "Measure the glass only — the lit area, not the black bezel around it.\n"
        "A ruler across the middle of the screen is fine; a millimetre or two of\n"
        "error is not going to be visible."
    )
    w = ask_float("  Panel width", float(cfg["display"]["screen_w_mm"]))
    h = ask_float("  Panel height", float(cfg["display"]["screen_h_mm"]))

    updates: list[tuple[str, str, Any]] = [
        ("display", "screen_w_mm", w),
        ("display", "screen_h_mm", h),
    ]

    configured = [int(v) for v in cfg["display"]["native_px"]]
    detected = detect_native_resolution()
    native = configured
    if detected and list(detected) != configured:
        print(f"\n  Your display reports {detected[0]}x{detected[1]} px, but the "
              f"config says {configured[0]}x{configured[1]}.")
        if ask_yes("  Use the detected size?"):
            native = list(detected)
            updates.append(("display", "native_px", native))
    elif detected:
        print(f"\n  Display confirmed at {detected[0]}x{detected[1]} px.")
    else:
        print(f"\n  Could not detect the panel resolution; keeping "
              f"{configured[0]}x{configured[1]} from the config.")
        print("  If that is not your panel's real pixel count, edit native_px by hand.")

    print(f"\n  {w:g} x {h:g} mm over {native[0]}x{native[1]} px"
          f"  ->  {native[0] / w:.2f} px/mm")
    print("  (That scale is what lets a floating window know where its own")
    print("   corners are in physical space.)")
    return updates


def measure_camera_offset(cfg: dict, screen_h_mm: float) -> list[tuple[str, str, Any]]:
    rule("2. Where the webcam is")
    print(
        "The origin of everything is the centre of the panel, so the camera needs\n"
        "to be described relative to that. Measure straight up from the top edge\n"
        "of the glass to the centre of the lens."
    )
    current_y = float(cfg["display"]["cam_offset_mm"][1])
    default_gap = max(0.0, current_y - screen_h_mm * 0.5)
    gap = ask_float("  Glass top to lens centre", default_gap)

    x_off = ask_float("  Lens offset left/right of centre (+ is right)", 0.0)
    y_off = screen_h_mm * 0.5 + gap
    z_off = ask_float("  Lens in front of / behind the glass (+ is towards you)", 0.0)

    print(f"\n  cam_offset_mm = [{x_off:g}, {y_off:g}, {z_off:g}]")
    return [("display", "cam_offset_mm", [x_off, y_off, z_off])]


# ----------------------------------------------------------------------------
# Stage 3 — focal length, from the camera
# ----------------------------------------------------------------------------


def open_camera(cfg: dict):
    import cv2

    backends = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "auto": cv2.CAP_ANY,
    }
    name = str(cfg["camera"].get("backend", "dshow")).lower()
    order = [name] + [b for b in ("dshow", "msmf", "auto") if b != name]

    index = int(cfg["camera"].get("index", 0))
    for b in order:
        cap = cv2.VideoCapture(index, backends[b])
        if not cap.isOpened():
            cap.release()
            continue

        # Set the format BEFORE reading a frame. Media Foundation will not
        # change resolution once the stream has started — it keeps reporting the
        # requested size while returning nothing but bad frames.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg["camera"]["width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg["camera"]["height"]))
        try:
            ok = cap.read()[0]
        except cv2.error:
            ok = False
        if ok:
            return cap, b
        cap.release()
    raise RuntimeError(f"could not open camera index {index}")


def make_landmarker():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"missing {MODEL_PATH} — run 'environment check.py' for the URL"
        )
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
    )


def calibrate_focal(cfg: dict) -> list[tuple[str, str, Any]]:
    import cv2
    import numpy as np
    import mediapipe as mp

    rule("3. Focal length")
    print(
        "This is the one number that cannot be measured with a ruler, so it gets\n"
        "solved instead. Sit at a distance you can measure, hold still, and the\n"
        "known pupil separation gives up the focal length.\n"
    )
    ipd_mm = float(cfg["tracking"]["ipd_mm"])
    print(f"  Assuming your interpupillary distance is {ipd_mm:g} mm.")
    if not ask_yes("  Is that right (63 mm is the adult average)?"):
        ipd_mm = ask_float("  Your interpupillary distance", ipd_mm)

    print(
        "\n  Now measure from the webcam lens to the bridge of your nose with a\n"
        "  tape measure. Any distance works; 500-600 mm is comfortable."
    )
    distance_mm = ask_float("  Lens to eyes", 500.0)

    max_yaw = float(cfg["tracking"].get("max_yaw_deg", 35.0))
    cap, backend = open_camera(cfg)
    landmarker = make_landmarker()

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n  Camera open: {backend} at {frame_w}x{frame_h}.")
    print("  Face the camera straight on. SPACE to capture, ESC to cancel.\n")

    samples: list[float] = []
    yaws: list[float] = []
    capturing = False
    start = time.perf_counter()
    result_focal: float | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(
            image, int((time.perf_counter() - start) * 1000)
        )

        # Mirror only what is shown. Every measurement above uses the raw frame,
        # because the sign conventions in eye_to_screen_space assume raw.
        view = cv2.flip(frame, 1)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            ipd_2d, ipd_3d, yaw, mid_u, mid_v = iris_separation_px(lm, frame_w, frame_h)
            ipd = corrected_ipd_px(ipd_2d, ipd_3d, max_yaw)

            for idx in (IRIS_LEFT, IRIS_RIGHT):
                u = int((1.0 - lm[idx].x) * frame_w)  # mirrored for display
                v = int(lm[idx].y * frame_h)
                cv2.circle(view, (u, v), 4, (0, 255, 255), -1)
            cv2.line(
                view,
                (int((1.0 - lm[IRIS_LEFT].x) * frame_w), int(lm[IRIS_LEFT].y * frame_h)),
                (int((1.0 - lm[IRIS_RIGHT].x) * frame_w), int(lm[IRIS_RIGHT].y * frame_h)),
                (0, 255, 255),
                1,
            )

            steady = yaw <= 12.0
            colour = (120, 255, 120) if steady else (100, 160, 255)
            cv2.putText(view, f"ipd  {ipd:6.1f} px", (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            cv2.putText(view, f"yaw  {yaw:6.1f} deg", (14, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            if not steady:
                cv2.putText(view, "face the camera straight on", (14, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 160, 255), 2)

            if capturing and steady:
                samples.append(ipd)
                yaws.append(yaw)
        else:
            cv2.putText(view, "no face", (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)

        if capturing:
            frac = len(samples) / FOCAL_SAMPLE_FRAMES
            cv2.rectangle(view, (14, frame_h - 40), (frame_w - 14, frame_h - 20),
                          (60, 60, 60), 1)
            cv2.rectangle(view, (14, frame_h - 40),
                          (14 + int((frame_w - 28) * min(frac, 1.0)), frame_h - 20),
                          (120, 255, 120), -1)
            cv2.putText(view, f"hold still  {len(samples)}/{FOCAL_SAMPLE_FRAMES}",
                        (14, frame_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (120, 255, 120), 2)
            if len(samples) >= FOCAL_SAMPLE_FRAMES:
                break
        else:
            cv2.putText(view, "SPACE to capture", (14, frame_h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

        cv2.imshow("focal length calibration", view)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            capturing = True
            samples.clear()
            yaws.clear()

    cap.release()
    cv2.destroyAllWindows()

    if len(samples) < FOCAL_SAMPLE_FRAMES:
        print("\n  Cancelled — nothing was written.")
        return []

    arr = np.array(samples)
    # The median, not the mean: one blink or one bad landmark frame can throw a
    # mean off by more than the precision we are trying to achieve.
    ipd_px = float(np.median(arr))
    spread = float(np.percentile(arr, 95) - np.percentile(arr, 5))
    focal = solve_focal_px(ipd_px, ipd_mm, distance_mm)

    print(f"\n  pupil separation : {ipd_px:.2f} px  (90% of frames within {spread:.2f} px)")
    print(f"  mean yaw during capture : {float(np.mean(yaws)):.1f} deg")
    print(f"  focal length     : {focal:.1f} px")

    naive = fallback_focal_px(frame_w)
    print(f"  for comparison, a 60 deg field of view would give {naive:.0f} px")
    hfov = 2 * math.degrees(math.atan((frame_w * 0.5) / focal))
    print(f"  so this camera's actual horizontal field of view is about {hfov:.0f} deg")

    if spread > ipd_px * 0.05:
        print(
            "\n  WARNING: the measurement moved by more than 5% while capturing.\n"
            "  Try again holding stiller, or with better lighting."
        )

    # Immediate sanity check: run the number backwards and see if it returns the
    # distance that was just typed in.
    back = depth_from_ipd(ipd_px, ipd_mm, focal)
    print(f"\n  Check: that focal length puts you at {back:.0f} mm, and you said "
          f"{distance_mm:g} mm.")

    updates: list[tuple[str, str, Any]] = [("tracking", "focal_px", round(focal, 1))]
    if ipd_mm != float(cfg["tracking"]["ipd_mm"]):
        updates.append(("tracking", "ipd_mm", ipd_mm))
    return updates


# ----------------------------------------------------------------------------
# --show and --selftest
# ----------------------------------------------------------------------------


def show_config(cfg: dict) -> None:
    d, t = cfg["display"], cfg["tracking"]
    w, h = float(d["screen_w_mm"]), float(d["screen_h_mm"])
    native = d["native_px"]

    rule("Current configuration")
    print(f"  panel           {w:g} x {h:g} mm")
    print(f"  native pixels   {native[0]} x {native[1]}  ->  {native[0] / w:.3f} px/mm")
    print(f"  camera offset   {d['cam_offset_mm']} mm from panel centre")
    print(f"  screen corners  bottom-left  ({-w/2:+.1f}, {-h/2:+.1f}, 0.0)")
    print(f"                  bottom-right ({+w/2:+.1f}, {-h/2:+.1f}, 0.0)")
    print(f"                  top-left     ({-w/2:+.1f}, {+h/2:+.1f}, 0.0)")
    print(f"  ipd             {t['ipd_mm']:g} mm")

    focal = t.get("focal_px")
    if focal is None:
        print("  focal length    NOT CALIBRATED — run this program without --show")
    else:
        focal = float(focal)
        hfov = 2 * math.degrees(math.atan((int(cfg['camera']['width']) * 0.5) / focal))
        print(f"  focal length    {focal:g} px  (~{hfov:.0f} deg horizontal fov)")
        print("\n  What that focal length means in practice:")
        for dist in (300, 500, 700, 1000):
            px = focal * float(t["ipd_mm"]) / dist
            print(f"    at {dist:4d} mm, your pupils are {px:5.1f} px apart")


def _raises(exc_type, fn) -> bool:
    try:
        fn()
    except exc_type:
        return True
    except Exception:  # noqa: BLE001 — the wrong exception is still a failure
        return False
    return False


def selftest() -> int:
    """Check the maths without needing a camera or a face."""
    import numpy as np

    rule("Self test")
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}" + (f"  — {detail}" if detail else ""))

    # --- focal solve and its inverse must agree ---
    focal = solve_focal_px(ipd_px=100.0, ipd_mm=63.0, distance_mm=500.0)
    check("focal solve", abs(focal - 793.65) < 0.01, f"{focal:.2f} px")
    back = depth_from_ipd(100.0, 63.0, focal)
    check("depth round trip", abs(back - 500.0) < 1e-6, f"{back:.4f} mm")

    # --- a smaller pupil gap must mean further away ---
    near = depth_from_ipd(120.0, 63.0, focal)
    far = depth_from_ipd(80.0, 63.0, focal)
    check("further away reads as fewer pixels", far > near, f"{near:.0f} < {far:.0f} mm")

    # --- yaw correction ---
    class FakeLm:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    # Facing straight on: both irises at the same depth, so no correction.
    straight = [None] * 478
    straight[IRIS_LEFT] = FakeLm(0.45, 0.5, 0.0)
    straight[IRIS_RIGHT] = FakeLm(0.55, 0.5, 0.0)
    ipd_2d, ipd_3d, yaw, mu, mv = iris_separation_px(straight, 1000, 1000)
    check("no yaw when facing straight on", abs(yaw) < 1e-6, f"{yaw:.4f} deg")
    check("straight-on ipd", abs(ipd_2d - 100.0) < 1e-9, f"{ipd_2d:.3f} px")
    check("eye midpoint", abs(mu - 500.0) < 1e-9 and abs(mv - 500.0) < 1e-9)

    # Turned 30 degrees: the true 100 px separation appears as 100*cos(30) = 86.6,
    # with the depth difference making up the rest.
    turned = [None] * 478
    true_sep, ang = 0.10, math.radians(30.0)
    turned[IRIS_LEFT] = FakeLm(0.5 - true_sep * math.cos(ang) / 2, 0.5,
                               -true_sep * math.sin(ang) / 2)
    turned[IRIS_RIGHT] = FakeLm(0.5 + true_sep * math.cos(ang) / 2, 0.5,
                                +true_sep * math.sin(ang) / 2)
    ipd_2d, ipd_3d, yaw, _, _ = iris_separation_px(turned, 1000, 1000)
    check("yaw recovered", abs(yaw - 30.0) < 1e-4, f"{yaw:.4f} deg, expected 30")
    check("foreshortened measurement", abs(ipd_2d - 86.603) < 0.01, f"{ipd_2d:.3f} px")
    fixed = corrected_ipd_px(ipd_2d, ipd_3d, 35.0)
    check("correction restores it", abs(fixed - 100.0) < 0.01, f"{fixed:.3f} px")

    # ...and the clamp must stop it running away at extreme angles.
    ang = math.radians(70.0)
    extreme = [None] * 478
    extreme[IRIS_LEFT] = FakeLm(0.5 - true_sep * math.cos(ang) / 2, 0.5,
                                -true_sep * math.sin(ang) / 2)
    extreme[IRIS_RIGHT] = FakeLm(0.5 + true_sep * math.cos(ang) / 2, 0.5,
                                 +true_sep * math.sin(ang) / 2)
    ipd_2d, ipd_3d, yaw, _, _ = iris_separation_px(extreme, 1000, 1000)
    clamped = corrected_ipd_px(ipd_2d, ipd_3d, 35.0)
    limit = ipd_2d / math.cos(math.radians(35.0))
    check("clamp holds at 70 deg", abs(clamped - limit) < 1e-9,
          f"{clamped:.2f} px, uncorrected would be {ipd_3d:.2f}")

    # --- signs: the number one bug in this project ---
    frame_w, frame_h, focal = 1280, 720, 1100.0
    offset = [0.0, 118.5, 0.0]
    centre = eye_to_screen_space(640, 360, 600, frame_w, frame_h, focal, offset)
    check("dead centre maps to x=0", abs(centre[0]) < 1e-9, f"x={centre[0]:.3f}")
    check("camera offset is applied", abs(centre[1] - 118.5) < 1e-9, f"y={centre[1]:.1f}")

    # Leaning to my own right moves me towards smaller u in the raw image.
    right = eye_to_screen_space(540, 360, 600, frame_w, frame_h, focal, offset)
    check("leaning right gives +X", right[0] > 0, f"x={right[0]:+.1f} mm")
    # Standing up moves me towards smaller v.
    up = eye_to_screen_space(640, 260, 600, frame_w, frame_h, focal, offset)
    check("standing up gives +Y", up[1] > centre[1], f"y={up[1]:+.1f} mm")
    # And the flips must actually reverse those.
    flipped = eye_to_screen_space(540, 360, 600, frame_w, frame_h, focal, offset,
                                  flip_x=True)
    check("flip_x reverses X", flipped[0] < 0, f"x={flipped[0]:+.1f} mm")

    # A 100 px sideways move at 600 mm should be 100*600/1100 = 54.5 mm.
    check("lateral scale", abs(right[0] - 100 * 600 / focal) < 1e-9,
          f"{right[0]:.2f} mm for 100 px at 600 mm")

    # --- config read/write, against a fixture ---
    #
    # Deliberately a fixture and not the real config.yaml. Whether a MEASURE ME
    # marker is still present depends on whether calibration has been run, so a
    # test against the live file would pass or fail based on the user's data
    # rather than on the code.
    import yaml

    fixture = (
        "# leading comment\n"
        "display:\n"
        "  screen_w_mm: 344.6         # MEASURE ME — panel width\n"
        "  native_px: [1920, 1080]\n"
        "  cam_offset_mm: [0.0, 100.0, 0.0]   # MEASURE ME — a guess\n"
        "\n"
        "tracking:\n"
        "  ipd_mm: 63.0               # keep this comment\n"
        "  focal_px: null             # filled in by calibration\n"
        "  flip_x: false\n"
        "\n"
        "filter:\n"
        "  min_cutoff: 1.0\n"
    )
    edited = set_config_value(fixture, "tracking", "focal_px", 1234.5)
    edited = set_config_value(edited, "display", "cam_offset_mm", [0.0, 118.5, 0.0])
    edited = set_config_value(edited, "display", "native_px", [2560, 1600])
    parsed = yaml.safe_load(edited)

    check("focal_px written", parsed["tracking"]["focal_px"] == 1234.5)
    check("list value written", parsed["display"]["cam_offset_mm"] == [0.0, 118.5, 0.0])
    check("int list written", parsed["display"]["native_px"] == [2560, 1600])
    check("MEASURE ME cleared once measured", "MEASURE ME — a guess" not in edited)
    check("other MEASURE ME untouched", "MEASURE ME — panel width" in edited,
          "we did not write screen_w_mm, so its marker must stay")
    check("ordinary comments survived", "# keep this comment" in edited
          and "# leading comment" in edited)
    check("line count unchanged",
          len(edited.splitlines()) == len(fixture.splitlines()))
    check("unrelated keys untouched", parsed["filter"]["min_cutoff"] == 1.0
          and parsed["tracking"]["flip_x"] is False)
    check("missing key is an error",
          _raises(KeyError, lambda: set_config_value(fixture, "tracking", "nope", 1)))

    # The real config must still be readable and complete, which is a different
    # question from whether the writer works.
    live = load_config()
    check("live config parses",
          all(k in live for k in ("display", "tracking", "filter", "camera", "render")))

    print()
    if failures:
        print(f"  {failures} check(s) failed.")
        return 1
    print("  All self tests passed.")
    return 0


# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focal", action="store_true",
                        help="only redo the focal length measurement")
    parser.add_argument("--show", action="store_true",
                        help="print the current configuration and exit")
    parser.add_argument("--selftest", action="store_true",
                        help="check the maths without a camera")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    cfg = load_config()

    if args.show:
        show_config(cfg)
        return 0

    updates: list[tuple[str, str, Any]] = []
    if args.focal:
        updates += calibrate_focal(cfg)
    else:
        rule("Parallax window calibration")
        print(
            "Three measurements, then one camera routine. Press Enter at any\n"
            "prompt to keep what is already in config.yaml."
        )
        updates += measure_panel(cfg)
        screen_h = next((v for s, k, v in updates if k == "screen_h_mm"),
                        float(cfg["display"]["screen_h_mm"]))
        updates += measure_camera_offset(cfg, float(screen_h))
        if ask_yes("\nRun the focal length measurement now?"):
            updates += calibrate_focal(cfg)

    if not updates:
        print("\nNothing changed.")
        return 0

    write_config_values(updates)
    rule("Written to config.yaml")
    for section, key, value in updates:
        print(f"  {section}.{key} = {_format_value(value)}")

    show_config(load_config())
    print("\nNext: 'eye position in millimeters.py'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
