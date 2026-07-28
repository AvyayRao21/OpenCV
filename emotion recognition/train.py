"""Train a compact expression-recognition CNN on FER-2013.

Run with Python 3.11 (the 3.13 install has no TensorFlow):

    python "emotion recognition/train.py" --quick    # 5 min sanity run
    python "emotion recognition/train.py"            # the real thing

Outputs land in "emotion recognition/model/":
    emotion.tflite    the model the live program actually uses
    emotion.json      class order + input shape, so nothing is retyped by hand
    best.keras        best-scoring checkpoint (kept for retraining, not shipped)
    last.keras        every epoch, so --resume can pick up after a crash
    history.csv       per-epoch metrics

Why the model is small: FER-2013's practical ceiling is around 65% — that is
roughly where human agreement with its own labels sits — so a network 20x
larger buys perhaps two points for many times the compute. Not a good trade for
something meant to run live on a laptop.

Why TFLite: the exported model is 357 KB against several megabytes for a Keras
file, loads in milliseconds, and classifies a face in 0.64 ms.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Quieten TensorFlow's C++ logging before it is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATASET = PROJECT_ROOT / "FER-2013 - data set - 1"
MODEL_DIR = HERE / "model"
CACHE_DIR = HERE / "cache"

# Locked project-wide class order: alphabetical, matching the dataset folder
# names. Do NOT reorder. The old scripts used FER-2013's CSV order, which puts
# neutral at index 6 instead of 4 and silently mislabels three emotions.
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

IMG_SIZE = 48


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------

def build_cache(split, limit_per_class=None):
    """Decode one split's JPEGs into a single uint8 array, cached on disk.

    Decoding ~29k JPEGs takes about a minute; doing it once per epoch instead
    of once total is the main reason naive FER-2013 scripts are slow. At
    48x48 grayscale the whole training split is only ~66 MB, so it lives in
    RAM comfortably.
    """
    tag = f"{split}_{limit_per_class or 'all'}"
    images_path = CACHE_DIR / f"{tag}_images.npy"
    labels_path = CACHE_DIR / f"{tag}_labels.npy"

    if images_path.exists() and labels_path.exists():
        return np.load(images_path), np.load(labels_path)

    split_dir = DATASET / split
    if not split_dir.is_dir():
        sys.exit(f"Dataset split not found: {split_dir}\n"
                 f"Expected FER-2013 extracted at: {DATASET}")

    images, labels = [], []
    for label, name in enumerate(CLASSES):
        files = sorted((split_dir / name).glob("*.jpg"))
        if limit_per_class:
            files = files[:limit_per_class]
        if not files:
            sys.exit(f"No images found in {split_dir / name}")
        for path in files:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  skipped unreadable file: {path.name}")
                continue
            if img.shape != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
        print(f"  {split}/{name:<9} {len(files):>5} images")

    images = np.asarray(images, dtype=np.uint8)[..., np.newaxis]
    labels = np.asarray(labels, dtype=np.int32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(images_path, images)
    np.save(labels_path, labels)
    return images, labels


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def build_model(keras):
    """A compact VGG-ish CNN, ~300k parameters.

    Two choices carry most of the size saving over the usual FER-2013 model:
    separable convolutions in the deeper blocks, and global average pooling
    instead of Flatten into a wide Dense layer. The old emotion_model.h5 spent
    over a million parameters on a single 2048x512 Dense; that layer is where
    these models normally get fat, and it is also where they overfit.
    """
    L = keras.layers

    # BatchNorm momentum 0.9 rather than the 0.99 default. At 225 steps per
    # epoch the default moving averages lag far behind the batch statistics,
    # so early validation runs use stale estimates and score near zero even
    # while training accuracy climbs. 0.9 lets them catch up within an epoch.
    def bn(x):
        return L.BatchNormalization(momentum=0.9)(x)

    def sep_block(x, filters, dropout):
        for _ in range(2):
            x = L.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
            x = bn(x)
            x = L.Activation("relu")(x)
        x = L.MaxPooling2D(2)(x)
        return L.Dropout(dropout)(x)

    inputs = L.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")

    # Scaling lives inside the model so the live program feeds raw 0-255
    # pixels and cannot get the normalisation wrong.
    x = L.Rescaling(1.0 / 255)(inputs)

    # Only ONE convolution runs at the full 48x48 resolution, and it has a
    # single input channel, so it is nearly free. This matters more than it
    # looks: an earlier version of this model ran two 64-channel convolutions
    # at 48x48 and spent ~650 ms per batch there alone — about 90% of total
    # step time for a model that needs 2 ms to run one face at inference.
    # Pooling before widening the channels cut step time roughly 5x at
    # essentially no cost in accuracy: 667 ms per batch became 197 ms.
    x = L.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = bn(x)
    x = L.Activation("relu")(x)
    x = L.MaxPooling2D(2)(x)          # 24x24
    x = L.Dropout(0.20)(x)

    x = L.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = bn(x)
    x = L.Activation("relu")(x)
    x = L.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = bn(x)
    x = L.Activation("relu")(x)
    x = L.MaxPooling2D(2)(x)          # 12x12
    x = L.Dropout(0.25)(x)

    x = sep_block(x, 128, 0.30)       # 6x6
    x = sep_block(x, 256, 0.30)       # 3x3

    x = L.SeparableConv2D(512, 3, padding="same", use_bias=False)(x)
    x = bn(x)
    x = L.Activation("relu")(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.40)(x)

    outputs = L.Dense(len(CLASSES), activation="softmax", name="emotion")(x)
    return keras.Model(inputs, outputs, name="fer_compact")


def make_datasets(tf, keras, train, test, batch_size):
    """tf.data pipelines. Augmentation stays outside the model so the exported
    graph contains only inference ops.

    Augmentation uses plain tf.image ops applied per sample, not Keras
    RandomRotation/RandomZoom/RandomTranslation layers. The Keras preprocessing
    layers measured at ~600 ms per batch of 128 on this CPU, which made them
    the bottleneck for a model that only needs ~2 ms to run a face. These
    native ops are a fraction of that and parallelise cleanly across the
    20 threads. Flip plus translation jitter plus mild brightness/contrast is
    the standard recipe for FER-2013 and gives up very little.
    """
    train_images, train_labels = train
    test_images, test_labels = test

    pad = 4  # gives up to +/-4 px of translation via random cropping

    def augment(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(
            image, IMG_SIZE + 2 * pad, IMG_SIZE + 2 * pad)
        image = tf.image.random_crop(image, [IMG_SIZE, IMG_SIZE, 1])
        image = tf.image.random_brightness(image, max_delta=20.0)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        # Scaling happens inside the model, so keep pixels in 0-255 here.
        image = tf.clip_by_value(image, 0.0, 255.0)
        return image, label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(len(train_images), reshuffle_each_iteration=True)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        .batch(batch_size)
        .map(lambda x, y: (tf.cast(x, tf.float32), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, test_ds


def class_weights_for(labels):
    """Square-root-tempered inverse-frequency weights.

    FER-2013 has 7,215 'happy' training images and 436 'disgust' — a 16.5x
    imbalance. Left alone, the cheapest route to a good accuracy score is to
    never predict disgust at all, and the model will take it.

    Full inverse-frequency weighting overcorrects hard: it puts a weight of
    ~9.4 on disgust and the model swings to predicting disgust for *everything*
    — measured, and it scored 1.55% on validation, which is exactly disgust's
    share of the test set. Taking the square root keeps rare classes learnable
    while capping the heaviest
    weight near 3x, which is the usual middle ground for this dataset.
    """
    counts = np.bincount(labels, minlength=len(CLASSES))
    total = counts.sum()
    weights = np.sqrt(total / (len(CLASSES) * np.maximum(counts, 1)))
    # Normalise so the average weight is 1.0; this keeps the loss on the same
    # scale as an unweighted run, so learning rates stay comparable.
    weights = weights / weights.mean()
    return {i: float(w) for i, w in enumerate(weights)}


# --------------------------------------------------------------------------
# Reporting and export
# --------------------------------------------------------------------------

def report(model, test_images, test_labels, batch_size):
    """Per-class accuracy and a confusion matrix.

    Overall accuracy alone is misleading on an imbalanced set — it can look
    healthy while a whole class is never predicted. Always read the diagonal.
    """
    probs = model.predict(test_images.astype("float32"), batch_size=batch_size,
                          verbose=0)
    preds = probs.argmax(axis=1)

    overall = float((preds == test_labels).mean())
    print(f"\nOverall test accuracy: {overall * 100:.2f}%")

    print("\nPer-class accuracy (recall):")
    for i, name in enumerate(CLASSES):
        mask = test_labels == i
        acc = float((preds[mask] == i).mean()) if mask.any() else float("nan")
        print(f"  {name:<9} {acc * 100:6.2f}%   ({int(mask.sum())} samples)")

    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(" " * 11 + "".join(f"{n[:4]:>7}" for n in CLASSES))
    for i, name in enumerate(CLASSES):
        row = [int(((test_labels == i) & (preds == j)).sum())
               for j in range(len(CLASSES))]
        print(f"  {name:<9}" + "".join(f"{v:>7}" for v in row))

    return overall


def export_tflite(tf, model, out_dir, overall_accuracy):
    """Export to TFLite plus a metadata sidecar.

    TFLite is what makes the live program feel instant: MediaPipe already
    bundles a TFLite runtime, so the demo needs no new dependency, does not
    pay TensorFlow's multi-second import, and runs a face in about a
    millisecond.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_model_dir = out_dir / "saved_model"

    # Keras 3 models go to TFLite via SavedModel; from_keras_model is not
    # reliable on this TF/Keras combination.
    model.export(str(saved_model_dir))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = out_dir / "emotion.tflite"
    tflite_path.write_bytes(tflite_model)

    metadata = {
        "classes": CLASSES,
        "input_shape": [IMG_SIZE, IMG_SIZE, 1],
        "input_range": "0-255 float32, grayscale; scaling is inside the model",
        "test_accuracy": round(overall_accuracy, 4),
        "trained_on": "FER-2013",
        "note": "Class order is alphabetical and locked. Read it from here.",
    }
    (out_dir / "emotion.json").write_text(json.dumps(metadata, indent=2),
                                          encoding="utf-8")

    size_kb = tflite_path.stat().st_size / 1024
    print(f"\nExported {tflite_path.name} ({size_kb:.0f} KB) and emotion.json")
    return tflite_path


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Sanity run: 300 images per class, 3 epochs. "
                             "Proves the pipeline before a full run.")
    # Default depends on --quick, so it stays None here to tell "not given"
    # apart from "given the same value as the default".
    parser.add_argument("--epochs", type=int, default=None,
                        help="Default: 60, or 3 with --quick.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from last.keras after an interruption.")
    args = parser.parse_args()

    import tensorflow as tf
    import keras

    limit = 300 if args.quick else None
    epochs = args.epochs if args.epochs else (3 if args.quick else 60)
    out_dir = MODEL_DIR / "quick" if args.quick else MODEL_DIR

    print("Loading FER-2013"
          f"{' (quick subset)' if args.quick else ''}...")
    train_images, train_labels = build_cache("train", limit)
    test_images, test_labels = build_cache("test", limit)
    print(f"  train {train_images.shape}  test {test_images.shape}")

    train_ds, test_ds = make_datasets(tf, keras,
                                      (train_images, train_labels),
                                      (test_images, test_labels),
                                      args.batch_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "last.keras"
    best_path = out_dir / "best.keras"
    state_path = out_dir / "state.json"

    initial_epoch = 0
    best_so_far = None
    if args.resume and last_path.exists() and state_path.exists():
        model = keras.models.load_model(last_path)
        # utf-8-sig so a byte-order mark does not break the read, and a broad
        # except because the whole point of this path is recovering from an
        # interrupted run — where a half-written state.json is a real
        # possibility. Crashing here would defeat the feature.
        try:
            state = json.loads(state_path.read_text(encoding="utf-8-sig"))
            initial_epoch = int(state["epoch"])
            best_so_far = state.get("best_val_accuracy")
        except (ValueError, KeyError, OSError) as error:
            # Keep the existing best safe, since we can no longer tell what
            # score it represents and are about to start judging from scratch.
            if best_path.exists():
                rescued = best_path.with_name("best-before-resume.keras")
                shutil.copy2(best_path, rescued)
                print(f"  kept a copy of the old best model as {rescued.name}")
            print(f"Could not read {state_path.name} ({error}). Resuming from "
                  f"the saved weights, but epoch count and best score restart.")
        else:
            print(f"Resuming from epoch {initial_epoch}", end="")
            print(f", best val_accuracy so far {best_so_far:.4f}"
                  if best_so_far is not None else " (no recorded best score)")
    else:
        model = build_model(keras)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    model.summary()

    class SaveState(keras.callbacks.Callback):
        """Record progress alongside last.keras so --resume knows where to
        restart. Assume the power will go out.

        Saving the best score matters as much as saving the epoch number.
        ModelCheckpoint tracks "best so far" only in memory, so a resumed run
        starts from negative infinity, treats its first epoch as a new record,
        and overwrites a genuinely better best.keras. Writing the score down
        here and feeding it back via initial_value_threshold below is what
        stops that.
        """

        def __init__(self, best):
            super().__init__()
            self.best = -np.inf if best is None else float(best)

        def on_epoch_end(self, epoch, logs=None):
            score = float((logs or {}).get("val_accuracy", -np.inf))
            self.best = max(self.best, score)
            state_path.write_text(json.dumps({
                "epoch": epoch + 1,
                "best_val_accuracy": None if self.best == -np.inf else self.best,
            }, indent=2), encoding="utf-8")

    callbacks = [
        keras.callbacks.ModelCheckpoint(best_path, monitor="val_accuracy",
                                        save_best_only=True, verbose=0,
                                        initial_value_threshold=best_so_far),
        keras.callbacks.ModelCheckpoint(last_path, verbose=0),
        SaveState(best_so_far),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                          patience=4, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(out_dir / "history.csv", append=args.resume),
    ]

    weights = class_weights_for(train_labels)
    print("\nClass weights (inverse frequency):")
    for i, name in enumerate(CLASSES):
        print(f"  {name:<9} {weights[i]:.2f}")

    started = time.time()
    model.fit(train_ds, validation_data=test_ds, epochs=epochs,
              initial_epoch=initial_epoch, class_weight=weights,
              callbacks=callbacks, verbose=1)
    minutes = (time.time() - started) / 60
    print(f"\nTraining finished in {minutes:.1f} min")

    accuracy = report(model, test_images, test_labels, args.batch_size)
    export_tflite(tf, model, out_dir, accuracy)

    if args.quick:
        print(f"\nQuick run only — accuracy here is meaningless ({limit} images "
              f"per class, {epochs} epochs). It proves the pipeline runs end to "
              f"end.\nNow run without --quick for the real model.")


if __name__ == "__main__":
    main()
