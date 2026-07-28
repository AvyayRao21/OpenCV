# Disclaimer

These are personal learning projects. They are not production software, not a
product, and not a service. Read this before pointing any of them at a person.

## The expression recogniser does not read feelings

This is the part that matters most, so it goes first.

The model looks at a face and picks whichever of seven labels the pixels most
resemble. That is all it does. **A face the model calls "sad" is a face that
looks like the images labelled "sad" in the training set — it is not evidence
that the person is sad.**

People smile when they are uncomfortable, keep a flat face when they are
delighted, and frown while concentrating on something they enjoy. Expressions
vary between cultures, between situations, and between individuals. There is
substantial research arguing that facial movements do not map reliably onto
internal emotional states at all. Nothing here can see how someone feels, and
no amount of accuracy on a benchmark would change that.

So: treat the output as "this face resembles these training images", never as
"this person feels this way".

## How accurate it actually is

Measured on the FER-2013 test set, which the model never saw during training:

| | Score |
| --- | ---: |
| Overall | **66.7%** |
| Best class (happy) | 86.6% |
| Worst class (fear) | **37.0%** |

**Roughly one prediction in three is wrong**, and `fear` is wrong about two
times in three — usually mistaken for `sad`, `angry` or `surprise`. For
context, FER-2013's labels are noisy enough that humans agree with them only
about 65% of the time, so this is a reasonable score rather than a bad one. It
is still nowhere near reliable enough to act on.

Accuracy also depends heavily on how the face is framed. Cropping wider than
the training data costs real accuracy, and the `disgust` class in particular
falls from 71% to 8% recall under a badly framed crop. The live program shows
you exactly what the model is being fed so you can check this yourself.

## Known bias

FER-2013 is a public dataset scraped from web image searches. Its subjects are
not demographically balanced, and I have not audited this model's accuracy
across skin tones, ages, genders, or anything else.

**Expect it to work less well on faces less represented in that dataset.** If
you use it on a group of people, it will very likely be wrong more often for
some of them than for others, and you will have no way of knowing which. That
alone rules out most uses involving other people.

## Do not use this for

Not because of the licence, but because it does not work well enough and the
underlying idea is not sound:

- Hiring, interviews, or any assessment of a candidate
- Monitoring students, employees, drivers, or anyone else
- Policing, security screening, surveillance, or border control
- Insurance, lending, or any decision about money
- Medical, psychological, or psychiatric assessment of any kind
- Lie detection or judging honesty
- Anything that affects a person's rights, safety, opportunities, or money

Emotion recognition is also regulated in some places. In the EU, for example,
using emotion inference in workplaces and schools is restricted. Several
jurisdictions treat face data as biometric data with specific consent rules. If
you are doing anything beyond running this on yourself, check what applies
where you are.

## Consent

Do not run any of this on someone who does not know about it. "It only runs
locally" is not consent, and neither is "it does not save anything".

## What it does with your data

Nothing leaves your computer. Every program here runs entirely on your own
machine, uses your webcam directly, and makes no network requests. No images
are saved, no predictions are logged, and there is no account, telemetry, or
upload of any kind. The only file any program writes is `live_settings.json`,
which stores one number: your preferred crop setting.

The FER-2013 dataset is not included in this repository. If you want to train
the model yourself you download it separately.

## The other programs

The hand and colour programs are simple demos, but two of them touch things
outside the window:

- **Volume control** changes your actual system volume while it runs. It will
  keep doing so as long as your hand is visible.
- **Colour detection with Arduino** writes to a serial port and drives an LED.
  Check your wiring and your port before running it. I am not responsible for
  what happens to your hardware.
- **The virtual keyboards** print typed characters to the console. They do not
  send keystrokes to other applications, so they cannot type into your
  password field by accident.

None of these are secure, sandboxed, or robust against unusual input. They are
demos.

## No warranty

Provided as is, with no warranty of any kind, and no liability for anything
that results from using it. See `LICENSE.md` for the formal terms.
