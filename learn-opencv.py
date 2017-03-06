import cv2
import numpy as np


def translate(img, x, y, cols=-1, rows=-1):
    Irows, Icols, ch = img.shape
    if ((cols, rows) == (-1, -1)):
        (cols, rows) = (Icols, Irows)
    M = np.float32([[1, 0, x], [0, 1, y]])
    result = cv2.warpAffine(res, M, (cols, rows))
    return result


img = cv2.imread('samples/03.PNG')

res = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

dst = translate(res, -100, -100,100,100)

cv2.imshow("Resized", dst)
cv2.waitKey(0)
