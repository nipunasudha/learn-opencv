import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while (1):
    # Take each frame
    _, frame = cap.read()

    edges = cv2.Canny(frame, 100, 300)

    cv2.imshow('frame', frame)
    cv2.imshow('edges', edges)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
