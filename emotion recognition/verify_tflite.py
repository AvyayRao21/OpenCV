"""Check the exported TFLite model against the Keras model it came from.

The export applies dynamic-range quantization, which shrinks the file by
roughly 4x but can cost accuracy. Shipping a quantized model without
re-measuring it is how you quietly lose a few points. Run:

    python "emotion recognition/verify_tflite.py"
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
CACHE_DIR = HERE / "cache"


def main():
    import tensorflow as tf

    meta = json.loads((MODEL_DIR / "emotion.json").read_text())
    classes = meta["classes"]

    images = np.load(CACHE_DIR / "test_all_images.npy")
    labels = np.load(CACHE_DIR / "test_all_labels.npy")
    print(f"Test set: {images.shape[0]} images\n")

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_DIR / "emotion.tflite"))
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    # Batch the interpreter so the sweep does not take all day.
    batch = 256
    interpreter.resize_tensor_input(inp["index"], [batch, 48, 48, 1])
    interpreter.allocate_tensors()

    preds = np.zeros(len(images), dtype=np.int32)
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch].astype("float32")
        if len(chunk) < batch:  # pad the final short batch
            chunk = np.pad(chunk, ((0, batch - len(chunk)), (0, 0), (0, 0), (0, 0)))
        interpreter.set_tensor(inp["index"], chunk)
        interpreter.invoke()
        probs = interpreter.get_tensor(out["index"])
        preds[start:start + batch] = probs.argmax(axis=1)[:len(images) - start]

    overall = float((preds == labels).mean())
    print(f"TFLite overall accuracy: {overall * 100:.2f}%")
    print(f"Keras accuracy recorded at export: {meta['test_accuracy'] * 100:.2f}%")
    print(f"Difference from quantization: "
          f"{(overall - meta['test_accuracy']) * 100:+.2f} points\n")

    print("Per-class recall from the shipped TFLite file:")
    for i, name in enumerate(classes):
        mask = labels == i
        print(f"  {name:<9} {(preds[mask] == i).mean() * 100:6.2f}%")

    # Single-face latency, which is what the live loop actually pays.
    interpreter.resize_tensor_input(inp["index"], [1, 48, 48, 1])
    interpreter.allocate_tensors()
    one = images[:1].astype("float32")
    for _ in range(20):  # warm up
        interpreter.set_tensor(inp["index"], one)
        interpreter.invoke()
    started = time.perf_counter()
    for _ in range(300):
        interpreter.set_tensor(inp["index"], one)
        interpreter.invoke()
        interpreter.get_tensor(out["index"])
    per_face = (time.perf_counter() - started) / 300 * 1000
    print(f"\nSingle-face latency: {per_face:.2f} ms "
          f"({1000 / per_face:.0f} faces/sec, 33 ms budget at 30 fps)")


if __name__ == "__main__":
    main()
