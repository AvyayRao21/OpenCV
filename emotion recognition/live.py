"""Live facial expression recognition from the webcam.

    python "emotion recognition/live.py"

Keys:
    q / Esc   quit
    s         toggle smoothing
    b         toggle the confidence bars
    p         toggle the preview of what the model actually sees
    [ / ]     tighten / widen the face crop
    r         reset the crop to the calibrated default
    w         save the current crop setting for next time

How it works: MediaPipe's BlazeFace detector finds faces, each face is cropped
to 48x48 grayscale and run through the compact CNN trained by train.py, and the
resulting probabilities are smoothed over time before being drawn.

The smoothing matters more than it sounds. Raw per-frame predictions flicker
between neighbouring emotions several times a second, which reads as broken
even when average accuracy is fine. An exponential moving average per tracked
face costs nothing and is the single biggest visible quality win.

Why the crop is adjustable
--------------------------
This model is far more sensitive to *how tightly the face is cropped* than to
lighting. Measured on the FER-2013 test set, feeding crops 25% wider than
training drops overall accuracy from 66.7% to 59.6%, and drops `disgust` from
71% to 35%. Crop framing, not brightness, is the thing worth tuning.

BOX_MARGIN below is calibrated, not guessed: running MediaPipe's detector on
FER-2013 images shows its box covers 0.888 of a FER-2013 crop, so expanding it
by 1/0.888 reproduces the framing the model was trained on.

There is no single best value, though, which is why you can change it live.
Overall accuracy peaks at margin 0.063, but `sad` and `disgust` both do better
at roughly 0.03 (tighter). Tune for the expressions you care about and watch
the preview: the goal is for your cropped face to be framed like the FER-2013
reference faces beside it.

Honest note on accuracy: this model scores 66.7% on the FER-2013 test set.
That is a good score for FER-2013 — the dataset's own labels are noisy and
human agreement sits around 65% — but roughly one call in three is wrong.
`fear` is the weakest class by a distance (37%) and is usually read as `sad`,
`angry` or `surprise`. Treat the output as a decent guess, not a verdict.
"""

import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
SETTINGS_PATH = MODEL_DIR / "live_settings.json"
# Optional: real FER-2013 faces shown beside the preview for comparison.
# Absent on a fresh clone, and the program works fine without them.
REFERENCE_DIR = HERE.parent / "FER-2013 - data set - 1" / "test"

MIN_DETECTION_CONFIDENCE = 0.6

# Calibrated so a live crop matches FER-2013's framing. See the module
# docstring; this is a measured value, not a guess.
CALIBRATED_MARGIN = 0.063
MARGIN_STEP = 0.01
MARGIN_LIMITS = (-0.10, 0.40)
# MediaPipe's box covers this fraction of a FER-2013 crop. Used only to show
# the effective scale on screen, so the number means something while tuning.
MEDIAPIPE_COVERAGE = 0.888

# How long the smoothing takes to catch up, in SECONDS rather than frames.
#
# This used to be a per-frame blend factor, which quietly meant different
# things on different machines: at 30 fps it smoothed over about a quarter of
# a second, but at the 80 fps this laptop actually runs, the same number
# smoothed over only 0.1 s — three times less. Expressing it as a time constant
# and deriving the per-frame factor from the real frame interval makes the feel
# identical whether the camera manages 15 fps or 120.
#
# 0.15 s is chosen to sit close to what the old fixed factor happened to give
# at 80 fps (about 0.1 s), which was tested on camera and felt right, while
# being slightly steadier. Raise it if labels flicker, lower it if they feel
# sluggish; the behaviour no longer depends on your frame rate either way.
SMOOTHING_SECONDS = 0.15
TRACK_PATIENCE = 8
MATCH_DISTANCE = 0.15

EMOTION_COLOURS = {
    "angry":    (60, 60, 220),
    "disgust":  (90, 170, 90),
    "fear":     (170, 90, 170),
    "happy":    (60, 200, 250),
    "neutral":  (185, 185, 185),
    "sad":      (200, 145, 70),
    "surprise": (220, 220, 70),
}


def load_interpreter(model_path):
    """Prefer the lightweight standalone runtime, fall back to TensorFlow.

    ai_edge_litert is about 3 MB; TensorFlow is closer to 600 MB. Anyone
    cloning this repo just to try the demo should not need the latter. Note
    that TF 2.20 removes tf.lite.Interpreter entirely, so the fallback has a
    shelf life.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            # Must go through the tf namespace: `from tensorflow.lite import
            # Interpreter` fails on TF 2.19, because tensorflow.lite is a lazily
            # loaded module rather than a real package attribute.
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            sys.exit("No TFLite runtime found. Install one of:\n"
                     "    pip install ai-edge-litert     (small, inference only)\n"
                     "    pip install tensorflow         (large, also trains)")
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def smoothing_alpha(dt):
    """Per-frame blend factor for a frame that took `dt` seconds.

    Derived from the time constant so the smoothing behaves the same at any
    frame rate. Returns 1.0 (no smoothing) if the timing looks nonsensical.
    """
    if dt <= 0 or dt > 1.0:
        return 1.0
    return 1.0 - math.exp(-dt / SMOOTHING_SECONDS)


def load_margin():
    # utf-8-sig tolerates a byte-order mark, which Windows editors like to add.
    try:
        text = SETTINGS_PATH.read_text(encoding="utf-8-sig")
        return float(json.loads(text)["box_margin"])
    except Exception:
        return CALIBRATED_MARGIN


def save_margin(margin):
    SETTINGS_PATH.write_text(json.dumps({"box_margin": round(margin, 3)},
                                        indent=2), encoding="utf-8")


def load_reference_faces(classes, per_class=1):
    """A few real FER-2013 faces to compare your own crop against.

    Returns None if the dataset is not present, which is the normal case for
    anyone who just cloned the repository.
    """
    if not REFERENCE_DIR.is_dir():
        return None
    faces = []
    for name in classes:
        folder = REFERENCE_DIR / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.jpg"))[:per_class]:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append((name, cv2.resize(img, (48, 48))))
    return faces or None


class FaceTrack:
    """One face, with probabilities smoothed across frames."""

    def __init__(self, centre, probs):
        self.centre = centre
        self.probs = probs
        self.missing = 0

    def update(self, centre, probs, alpha):
        self.centre = centre
        self.probs = alpha * probs + (1.0 - alpha) * self.probs
        self.missing = 0


def match_tracks(tracks, detections, frame_width):
    """Greedy nearest-centre matching. Faces do not move far between frames at
    30 fps, so anything cleverer would be wasted effort here."""
    threshold = MATCH_DISTANCE * frame_width
    unmatched = list(tracks)
    pairs = []

    for centre, box, probs in detections:
        best, best_distance = None, threshold
        for track in unmatched:
            distance = np.hypot(centre[0] - track.centre[0],
                                centre[1] - track.centre[1])
            if distance < best_distance:
                best, best_distance = track, distance
        if best is not None:
            unmatched.remove(best)
        pairs.append((best, centre, box, probs))
    return pairs, unmatched


def crop_face(frame_gray, box, margin):
    """Cut the face out with margin, clamped to the frame."""
    height, width = frame_gray.shape
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(width, x + w + mx), min(height, y + h + my)
    if x1 <= x0 or y1 <= y0:
        return None
    face = frame_gray[y0:y1, x0:x1]
    if face.size == 0:
        return None
    return cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)


def draw_face(frame, box, probs, classes, margin):
    """Box, label, runner-up and the crop rectangle actually being fed in."""
    x, y, w, h = box
    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1])
    name, confidence = classes[top], float(probs[top])
    colour = EMOTION_COLOURS.get(name, (200, 200, 200))

    # The detector box, and the wider rectangle the model actually receives.
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    mx, my = int(w * margin), int(h * margin)
    cv2.rectangle(frame, (x - mx, y - my), (x + w + mx, y + h + my),
                  (110, 110, 110), 1)

    label = f"{name}  {confidence * 100:.0f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ly = max(th + 8, y - 8)
    cv2.rectangle(frame, (x, ly - th - 8), (x + tw + 12, ly + 4), colour, -1)
    cv2.putText(frame, label, (x + 6, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (20, 20, 20), 2, cv2.LINE_AA)

    # Runner-up, so a near-tie is visible rather than hidden behind one word.
    runner = f"then {classes[second]} {float(probs[second]) * 100:.0f}%"
    cv2.putText(frame, runner, (x + 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (170, 170, 170), 1, cv2.LINE_AA)


def draw_panel(frame, probs, classes):
    """Bar chart of all seven probabilities for the most prominent face.

    Showing the whole distribution rather than just the winner makes the
    model's uncertainty visible — when 'fear' and 'sad' are nearly tied, that
    is worth seeing rather than hiding behind a single confident-looking word.
    """
    x, y = 16, 90
    row, bar_width, bar_height = 30, 190, 16

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - 26),
                  (x + bar_width + 78, y + row * len(classes) - 6),
                  (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "confidence", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)

    top = int(probs.argmax())
    for i, name in enumerate(classes):
        by = y + i * row
        colour = EMOTION_COLOURS.get(name, (200, 200, 200))
        dim = tuple(int(c * 0.45) for c in colour)

        cv2.putText(frame, name, (x, by + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (235, 235, 235) if i == top else (150, 150, 150), 1,
                    cv2.LINE_AA)
        bx = x + 74
        cv2.rectangle(frame, (bx, by), (bx + bar_width, by + bar_height),
                      (55, 55, 55), -1)
        filled = int(bar_width * float(probs[i]))
        if filled > 0:
            cv2.rectangle(frame, (bx, by), (bx + filled, by + bar_height),
                          colour if i == top else dim, -1)
        cv2.putText(frame, f"{float(probs[i]) * 100:3.0f}%",
                    (bx + bar_width + 6, by + 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (200, 200, 200), 1, cv2.LINE_AA)


def draw_preview(frame, crop, references, margin):
    """Show the 48x48 the model is actually being given, with real FER-2013
    faces beside it for comparison.

    This is the whole point of the tuner: you are trying to make your own crop
    sit in the frame the same way the reference faces do. Reading the number is
    much less useful than seeing the two side by side.
    """
    height, width = frame.shape[:2]
    size = 132
    pad = 12
    x0 = width - size - pad
    y0 = 60

    scale = (1 + 2 * margin) * MEDIAPIPE_COVERAGE
    panel_h = size + 78 + (56 if references else 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - pad, y0 - 30),
                  (x0 + size + pad, y0 + panel_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "model input", (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)

    if crop is not None:
        big = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
        frame[y0:y0 + size, x0:x0 + size] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    else:
        cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (60, 60, 60), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (120, 120, 120), 1)

    ty = y0 + size + 20
    tint = (110, 220, 110) if abs(scale - 1.0) < 0.04 else (110, 170, 235)
    cv2.putText(frame, f"margin {margin:+.3f}", (x0, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 215, 215), 1, cv2.LINE_AA)
    cv2.putText(frame, f"scale  {scale:.3f}", (x0, ty + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tint, 1, cv2.LINE_AA)
    cv2.putText(frame, "1.000 = as trained", (x0, ty + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    if references:
        ry = ty + 54
        cv2.putText(frame, "FER-2013 reference", (x0, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1,
                    cv2.LINE_AA)
        thumb = 40
        for i, (_, face) in enumerate(references[:3]):
            tx = x0 + i * (thumb + 6)
            small = cv2.resize(face, (thumb, thumb))
            frame[ry + 8:ry + 8 + thumb, tx:tx + thumb] = cv2.cvtColor(
                small, cv2.COLOR_GRAY2BGR)


def main():
    model_path = MODEL_DIR / "emotion.tflite"
    meta_path = MODEL_DIR / "emotion.json"
    if not model_path.exists() or not meta_path.exists():
        sys.exit(f"Model not found in {MODEL_DIR}.\n"
                 f'Train it first:  python "emotion recognition/train.py"')

    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    # Class order comes from the model's own metadata. Never retype it here —
    # the original scripts in this project hardcoded a different order and
    # would have mislabelled three emotions.
    classes = meta["classes"]

    interpreter = load_interpreter(model_path)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    import mediapipe as mp
    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=MIN_DETECTION_CONFIDENCE)

    references = load_reference_faces(classes)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit("Cannot open the webcam. Is another program using it?")

    margin = load_margin()
    print(f"Model: {model_path.name}  "
          f"({meta.get('test_accuracy', 0) * 100:.1f}% on the FER-2013 test set)")
    print(f"Crop margin: {margin:+.3f}  (calibrated default "
          f"{CALIBRATED_MARGIN:+.3f})")
    print("q/Esc quit  s smoothing  b bars  p preview  [ ] crop  r reset  "
          "w save crop")

    tracks = []
    frame_times = deque(maxlen=30)
    smoothing = True
    show_panel = True
    show_preview = True
    last_crop = None
    notice, notice_until = "", 0.0
    previous_frame_at = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost the camera feed.")
            break
        started = time.perf_counter()
        # Real elapsed time since the last frame, including the time spent
        # waiting on the camera — that is what the smoothing must track.
        frame_dt = 0.0 if previous_frame_at is None else started - previous_frame_at
        previous_frame_at = started

        frame = cv2.flip(frame, 1)  # mirror, so moving left looks like left
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        detections = []
        biggest_area, last_crop = -1, None
        for detection in results.detections or []:
            rel = detection.location_data.relative_bounding_box
            x = int(rel.xmin * width)
            y = int(rel.ymin * height)
            w = int(rel.width * width)
            h = int(rel.height * height)
            if w <= 0 or h <= 0:
                continue

            face = crop_face(gray, (x, y, w, h), margin)
            if face is None:
                continue
            if w * h > biggest_area:
                biggest_area, last_crop = w * h, face

            # The model rescales internally, so feed raw 0-255 values.
            batch = face.astype(np.float32)[np.newaxis, ..., np.newaxis]
            interpreter.set_tensor(input_index, batch)
            interpreter.invoke()
            probs = interpreter.get_tensor(output_index)[0].copy()

            detections.append(((x + w / 2, y + h / 2), (x, y, w, h), probs))

        pairs, unmatched = match_tracks(tracks, detections, width)
        alpha = smoothing_alpha(frame_dt) if smoothing else 1.0
        live = []
        for track, centre, box, probs in pairs:
            if track is None:
                track = FaceTrack(centre, probs)
                tracks.append(track)
            else:
                track.update(centre, probs, alpha)
            live.append((track, box))

        # Age out tracks whose face was not seen this frame.
        for track in unmatched:
            track.missing += 1
        tracks = [t for t in tracks if t.missing <= TRACK_PATIENCE]

        for track, box in live:
            draw_face(frame, box, track.probs, classes, margin)

        if show_panel and live:
            largest = max(live, key=lambda item: item[1][2] * item[1][3])
            draw_panel(frame, largest[0].probs, classes)

        if show_preview:
            draw_preview(frame, last_crop, references, margin)

        frame_times.append(time.perf_counter() - started)
        fps = 1.0 / max(np.mean(frame_times), 1e-6)

        cv2.putText(frame, f"{fps:4.1f} fps", (16, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2,
                    cv2.LINE_AA)
        status = f"faces {len(live)}   smoothing {'on' if smoothing else 'off'}"
        cv2.putText(frame, status, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (235, 235, 235), 2, cv2.LINE_AA)
        if not live:
            cv2.putText(frame, "no face detected", (16, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 235), 2,
                        cv2.LINE_AA)
        if notice and time.time() < notice_until:
            cv2.putText(frame, notice, (16, height - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 220, 110), 2,
                        cv2.LINE_AA)

        cv2.imshow("Live Expression Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            smoothing = not smoothing
        elif key == ord("b"):
            show_panel = not show_panel
        elif key == ord("p"):
            show_preview = not show_preview
        elif key == ord("["):
            margin = max(MARGIN_LIMITS[0], margin - MARGIN_STEP)
        elif key == ord("]"):
            margin = min(MARGIN_LIMITS[1], margin + MARGIN_STEP)
        elif key == ord("r"):
            margin = CALIBRATED_MARGIN
            notice, notice_until = "crop reset to calibrated default", time.time() + 2
        elif key == ord("w"):
            save_margin(margin)
            notice = f"saved margin {margin:+.3f} to live_settings.json"
            notice_until = time.time() + 2

    cap.release()
    face_detection.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
