import cv2
import numpy as np


def translate(img, offset, size=(-1, -1)):
    Irows, Icols, ch = img.shape
    if (size == (-1, -1)):
        size = (Icols, Irows)
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    result = cv2.warpAffine(res, M, (size[0],size[1]))
    return result


img = cv2.imread('samples/03.PNG')

res = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

dst = translate(res, (-100, -100))

cv2.imshow("Resized", dst)
cv2.waitKey(0)
