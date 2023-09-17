# Laser-shoting

Inspired by https://github.com/DIMOSUS/Laser-shoting (see https://habr.com/ru/articles/248181/ for more information)

## In action

[![DIY laser pointer shooting](http://img.youtube.com/vi/7N7_aXk8Q74/0.jpg)](http://www.youtube.com/watch?v=7N7_aXk8Q74)

## What you need?

* A simple toy gun made from laser pointer

* Computer with python3 and a web camera

## Modifications to the original version

Main changes:

- No Arduion or other complex hardware is required.
Siple gun with laser pointer and button. The software will detect start and end moment of shooting. You can also specify a delay between shots.

- Probably better performance
The software will check only target and small area around it instead of full frame. This means that less resources are required.

# How to use?

## Target

It's better to print a shooting target on a paper from [src/target.jpg](https://github.com/Stuw/Laser-shoting/blob/master/src/target.jpg). You can also use any dot on a wall as a target.

## Software

At this moment only linux is supported because `aplay` tool is hardcoded to use for sound effects.

### Installation

    python3 -m venv .venv
    . .venv/bin/activate
    pip install --upgrade pip

### Configuration

- Run script in debug mode

        python ./main.py -d

- Adjust a web camera position if needed.

- Move cursor to the center of your target and remember or write down it's possition (check the bottom left corner on Frame window).

- Move cursor to the edge of your target. Check the cursor position and use this position to calculate the target radius in pixels.

- Run script with correct target parameters

        python ./main.py -x 295 -y 233 -r 15 -d

- Start shooting

# License

Code is licensed under MIT license.

License for image and audio files is unknown.
