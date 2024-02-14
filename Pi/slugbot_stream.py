
import cv2 as cv
from CAMERA import VideoCapture

cap = VideoCapture(0)
while True:
    img = cap.read()
    cv.imshow("a", img)
    cv.waitKey(1)
