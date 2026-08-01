# Colour detection

Naming the dominant colour in front of the camera, and pushing that result out
to real hardware.

## The programs

| Program | What it does | Quit |
| --- | --- | --- |
| `dominant color.py` | Names the most common colour in view and prints it over the video. | `q` |
| `dominant color with arduino.py` | The same detection, but also sends the colour to an Arduino over serial so a real LED matches what the camera sees. | `q` |

## How it works

Each frame is converted to HSV, which separates *what* colour something is
(hue) from how bright or washed out it is. That makes colour far easier to
match than in RGB, where a shadow changes all three numbers at once.

Eight colours are defined as hue ranges. Every frame, each range becomes a mask,
the white pixels in each mask are counted, and the colour with the most pixels
wins. If even the winner has fewer than 500 pixels, the program says `Unknown`
rather than guessing at noise.

**Red is a special case.** Hue is a circle, and red sits at the point where it
wraps from 180 back to 0. So red needs two ranges — one at each end — combined
with a bitwise OR. Every other colour is a single range. This is the detail
that catches people out the first time they work in HSV.

## The Arduino version

`dominant color with arduino.py` opens a serial port, then writes the detected
colour as plain text, one line per frame:

```
255,0,0\n
```

The sketch in `arduino/rgb led.ino` reads those lines and drives an RGB LED
with `analogWrite`.

### Wiring

| Arduino pin | LED |
| --- | --- |
| 9 | Red |
| 10 | Green |
| 11 | Blue |
| GND | Common cathode |

Pins 9, 10 and 11 are used because they support PWM, which is what lets the LED
show a mix rather than only full brightness. Put a resistor on each colour leg.

### Before you run it

- Set your port at the top of the Python file. It defaults to `COM3`, which is
  a Windows name; on Linux or macOS it will be something like `/dev/ttyUSB0`.
- **The program will not start without a board connected.** It opens the serial
  port before it opens the camera, so a missing board is an immediate error
  rather than a silent failure.
- Close the Arduino IDE's Serial Monitor first. Only one program can hold a
  serial port at a time.
- `pip install pyserial` — note the package is `pyserial` even though you
  `import serial`.

## Running

```bash
pip install opencv-python numpy
pip install pyserial                # only for the Arduino version

python "color recognition/dominant color.py"
```

## Known limits

The colour ranges are hardcoded and tuned for reasonable indoor lighting. Under
warm lamps or a strong colour cast, things drift — white walls start reading as
yellow, and blues slide toward cyan. The honest fix is to tune the ranges for
your own conditions, in the same spirit as `hsv tuner.py` in the hand tracking
folder.

It also reports one colour for the entire frame, so it is measuring your
background as much as whatever you are holding up. Filling the frame with the
object gives much better results.
