import cv2
import numpy as np
import glob

def getGrayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def getEdges(grayscale):
    return cv2.Canny(grayscale, 100, 170, apertureSize = 3)

def showDefaultImage(filepath):
    image = cv2.imread(filepath)
    cv2.imshow("Default Image", image)
    cv2.waitKey(0)

def showGrayscaleAndCannyEdges(filepath):
    image = cv2.imread(filepath)
    grayscale = getGrayscale(image)
    edges = getEdges(grayscale)

    cv2.imshow("Grayscale And Canny Edges", edges)
    cv2.waitKey(0)

def showHoughLines(filepath, rhoAccuracy, theta, lineThreshold):
    image = cv2.imread(filepath)
    grayscale = getGrayscale(image)
    edges = getEdges(grayscale)

    lines = cv2.HoughLines(edges, rhoAccuracy, theta, lineThreshold)

    lineColor = (0, 0, 255)
    lineThickness = 1

    # Convert lines to the format required by cv.lines (end points)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        firstLinePoint = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        secondLinePoint = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(image, firstLinePoint, secondLinePoint, lineColor, lineThickness)

    cv2.imshow('Hough Lines', image)
    cv2.waitKey(0)

def showProbabilisticHoughLines(filepath, rhoAccuracy, theta, lineThreshold):
    image = cv2.imread(filepath)
    grayscale = getGrayscale(image)
    edges = getEdges(grayscale)

    minLineLength = 5
    maxGapBetweenLines = 10
    lines = cv2.HoughLinesP(edges, rhoAccuracy, theta, lineThreshold, minLineLength, maxGapBetweenLines)

    lineColor = (0, 255, 0)
    lineThickness = 2

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), lineColor, lineThickness)

    cv2.imshow('Probabilistic Hough Lines', image)
    cv2.waitKey(0)

for filepath in glob.iglob('assets/images/*.jpg'):
    showDefaultImage(filepath)
    showGrayscaleAndCannyEdges(filepath)

    # Rho accuracy of 1 pixel
    rhoAccuracy = 1
    # Theta accuracy of np.pi / 180 = 1 degree
    theta = np.pi/180
    # Line threshold (number of points on line)
    lineThreshold = 200

    showHoughLines(filepath, rhoAccuracy, theta, lineThreshold)
    showProbabilisticHoughLines(filepath, rhoAccuracy, theta, lineThreshold)

    cv2.destroyAllWindows()
