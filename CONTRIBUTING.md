# Contributing

Thanks for looking. This is a personal learning project, so it is worth being
upfront about what that means for contributions.

## What this project is

These are built to learn computer vision, and the code is meant to be read as
much as run. That shapes what belongs here: **a clear version of something is
better than a clever version of it.** If a change makes a program faster but
harder to follow, it probably does not fit.

## How this repository is built and maintained

Development here is AI-assisted. I set the requirements, plan each project in
stages, test on real hardware and review every change before it is published.
An AI coding assistant (Claude Code) implements much of the code and also
maintains the repository scaffolding: this guide, the changelog, the READMEs,
the disclaimer and the licence file. That is why the paperwork is more
thorough than you might expect from a personal project. The full breakdown is
in [How this was built](README.md#how-this-was-built).

AI-assisted contributions are welcome on the same terms as any other: say so
in the pull request, and show that you actually ran the thing.

## What is genuinely welcome

- **Bug reports.** Especially anything that crashes, or a program that behaves
  differently on Linux or macOS. I develop on Windows, so those paths get the
  least testing.
- **Corrections.** If something in the README or the disclaimer is wrong,
  particularly the accuracy figures or the claims about what the model can
  infer, I want to know.
- **Ideas for new experiments.** Open an issue and describe it. This collection
  grows by adding small self-contained programs, so suggestions along those
  lines are the most useful kind.
- **Portability fixes** that do not add dependencies.

## What I will probably decline

- Rewrites of existing programs into a shared framework. They are deliberately
  standalone so each can be read on its own.
- New dependencies, unless something genuinely cannot be done without one. The
  current set is small on purpose.
- Swapping the trained model for a pretrained one. Training it locally from
  public data is the point of that project, so a better score obtained by
  importing someone else's weights would defeat it.
- Anything that sends data off the machine. Everything here runs locally and
  that is not going to change.
- Features aimed at the uses ruled out in [DISCLAIMER.md](DISCLAIMER.md).

## Practical notes

- Python 3.11. Not 3.13 — MediaPipe and TensorFlow have no wheels for it.
- Every program should run with no arguments and no setup beyond `pip install`.
- Quote your file paths. Several folders here contain spaces.
- If you change anything about the expression model, re-run
  `verify_tflite.py` and put the real numbers in the pull request. Claims about
  accuracy need measurements behind them.
- Keep [CHANGELOG.md](CHANGELOG.md) updated for anything a user would notice.

## Reporting a bug

Include your OS, your Python version, the program you ran, and the full error.
For anything involving the camera, your webcam resolution and the frame rate
shown on screen help a lot.
