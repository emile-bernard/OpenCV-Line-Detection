# OpenCV-Line-Detection
This project contains a basic python script to run opencv-python with Hough Line Transform to detect lines in an image. This script shows the product of a basic Canny edge detection and after shows the product of line detection.

## Previews
![Preview1](.documentation/preview/Preview1.JPG?raw=true "Preview1")
![Preview2](.documentation/preview/Preview2.JPG?raw=true "Preview2")
![Preview3](.documentation/preview/Preview3.JPG?raw=true "Preview3")

## How it work's
A line can be represented as y = mx+c or in parametric form, as rho = x cos(theta) + y sin(theta) where rho is the perpendicular distance from origin to the line, and theta is the angle formed by this perpendicular line and horizontal axis measured in counter-clockwise:

![CoordinateSystem](.documentation/ CoordinateSystem.PNG?raw=true "CoordinateSystem")

## How to setup
- Install opencv-python
```
$ pip install opencv-python
```

- Install numpy
```
$ pip install numpy
```

## How to run
- Run
```
$ python ./main.py
```

## Links
- [Python Patterns - Github](https://github.com/faif/python-patterns)
- [Python Design Patterns: For Sleek And Fashionable Code](https://www.toptal.com/python/python-design-patterns)
- [Hough Line Transform](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html)
