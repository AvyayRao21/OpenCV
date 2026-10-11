# Facial expression recognition

A convolutional neural network trained from scratch to sort a face into one of
seven expressions, with a live webcam demo built on MediaPipe face detection.

No pretrained weights are used anywhere. The network is trained entirely on a
laptop CPU, in about 42 minutes. Importing someone else's weights would have
scored higher, but training it locally from public data was the requirement.

Built in 2026 through AI-assisted development: the code and the measurements
below were produced with an AI coding assistant, against requirements I set,
and tested on my own webcam. See
[How this was built](../README.md#how-this-was-built).

> **Read [DISCLAIMER.md](../DISCLAIMER.md) before using this on anybody.** It
> recognises expressions, not feelings, and it is wrong about one time in
> three.

## Running the demo

The trained model is included, so this works straight away:

```bash
pip install opencv-python mediapipe numpy tensorflow
python "emotion recognition/live.py"
```

| Key | Does |
| --- | --- |
| `q` / `Esc` | Quit |
| `s` | Toggle smoothing |
| `b` | Toggle the confidence bars |
| `p` | Toggle the preview of what the model actually sees |
| `[` `]` | Tighten / widen the face crop |
| `r` | Reset the crop to the calibrated default |
| `w` | Save your crop setting for next time |

## What it does per frame

1. MediaPipe's BlazeFace detector finds every face.
2. Each face is cropped, converted to greyscale and resized to 48×48 — the size
   the model was trained on.
3. The network returns seven probabilities.
4. Those probabilities are blended with the previous frame's before anything is
   drawn.

That last step matters more than it sounds. Raw frame-by-frame predictions
flicker between similar expressions several times a second, which looks broken
even when the average accuracy is fine. Smoothing costs nothing and is the
single biggest improvement to how it feels. Press `s` to see the difference.

The smoothing is expressed as a time constant rather than a per-frame amount,
so it behaves identically whether your webcam manages 15 fps or 120.

## Accuracy

Measured on the FER-2013 test set, which the model never sees during training.

**66.7% overall.** For context, FER-2013's own labels are noisy enough that
humans agree with them only about 65% of the time, so this is a reasonable
result — and it still means one prediction in three is wrong.

| Expression | Correctly identified |
| --- | ---: |
| happy | 86.6% |
| surprise | 84.2% |
| disgust | 70.3% |
| neutral | 69.6% |
| angry | 59.9% |
| sad | 53.7% |
| **fear** | **37.0%** |

`fear` is the clear weak point. It shares raised eyebrows with `surprise` and a
downturned mouth with `sad`, and at 48×48 pixels there is very little detail
left to separate them. It usually gets called one of those two, or `angry`.
That number is published here rather than hidden inside the average.

## The thing nobody expected

On the first real camera test, `sad`, `disgust` and `fear` almost never
appeared. Uneven lighting was the obvious suspect, so that was tested first:
the test set was distorted to imitate bad webcam conditions — brighter, darker,
flatter, harsher — and the accuracy drop measured.

Barely at all. The model held between 64% and 67% throughout. Every contrast
normalisation tried after that made things **worse**, not better: adaptive
histogram equalisation cost about a point, plain histogram equalisation over
five.

What actually matters is how tightly the face is cropped:

| Crop | Overall | `disgust` |
| --- | ---: | ---: |
| 10% tighter | 63.2% | 74.8% |
| **As trained** | **66.7%** | 71.2% |
| 25% wider | 59.6% | 35.1% |
| 50% wider | 47.2% | **8.1%** |

`disgust` falls from 71% to 8% on framing alone. Position barely matters — an
off-centre face is fine. It is the *scale* that has to be right.

So rather than guess the right crop, it was measured. Running MediaPipe's face
detector on the FER-2013 images themselves shows its box covers 88.8% of one of
those crops, which means expanding it by 1.126× reproduces the framing the model
was trained on. That is where the default comes from.

There is no single correct value, though. Overall accuracy peaks at that
setting, but `sad` and `disgust` both do better with a tighter crop. That is why
you can change it live with `[` and `]`, with a preview of the exact image being
fed to the model, and real FER-2013 faces beside it to match against.

## Training it yourself

Download [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) and
extract it alongside the project so the folders sit at
`FER-2013 - data set - 1/train/<expression>/` and `.../test/<expression>/`. The
dataset is about 29,000 training images and is not included here.

```bash
python "emotion recognition/train.py" --quick    # ~15 seconds
python "emotion recognition/train.py"            # the real run, ~42 minutes
```

| Flag | Does |
| --- | --- |
| `--quick` | 300 images per class, 3 epochs. The accuracy is meaningless — the point is finding out in seconds whether the pipeline is broken, instead of an hour in. |
| `--epochs N` | Number of epochs. Default 60, or 3 with `--quick`. |
| `--batch-size N` | Default 128. |
| `--resume` | Continue an interrupted run from the last completed epoch. |

A checkpoint is written every epoch, so a crash or a power cut costs one epoch
rather than the whole run.

Afterwards, `verify_tflite.py` re-scores the exported model against the full
test set. The export is quantised to shrink it, and quantisation can quietly
cost accuracy, so this checks how much. In the included model it cost 0.09 of a
percentage point.

## How the model is built

A compact network, about 326,000 parameters, roughly a twentieth the size of
the usual FER-2013 architectures. Two decisions carry most of that saving:

- **Only one convolution runs at full 48×48 resolution**, before any pooling.
  An earlier version ran two wide convolutions there and spent about 90% of its
  training time in that one block. Pooling before widening the channels cut
  training time by 3.4× at no cost in accuracy. On a CPU, *where* you pool
  matters more than how many parameters you have.
- **Global average pooling instead of a wide fully-connected layer.** That layer
  is where these models normally get fat, and where they overfit.

The rarest expression, `disgust`, has 436 training images against 7,215 for
`happy` — a 16.5× imbalance. Left alone, the cheapest way to a good score is to
never predict `disgust` at all. Weighting the classes by the square root of
their inverse frequency fixes that without overcorrecting; full inverse
weighting swings the model into predicting `disgust` for almost everything.

The result is exported to TensorFlow Lite: 357 KB, and 0.64 ms to classify one
face, which is about 2% of the frame budget at 30 fps.

## Files

| File | What it is |
| --- | --- |
| `live.py` | The webcam demo |
| `train.py` | Training, evaluation and export |
| `verify_tflite.py` | Re-scores the exported model on the full test set |
| `model/emotion.tflite` | The trained model, 357 KB |
| `model/emotion.json` | Class order and input shape, read by the demo so the labels can never drift out of sync with the model |
