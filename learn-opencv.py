import cv2
import numpy as np


def rotate(img, angle, origin=('x', 'x'), size=(-1, -1), scale=1.0):
    Irows, Icols, ch = img.shape

    if (size == (-1, -1)):
        size = (Icols, Irows)

    if (origin == ('x', 'x')):
        origin = ((size[0] / 2, size[1] / 2))

    M = cv2.getRotationMatrix2D(origin, angle, scale)
    result = cv2.warpAffine(res, M, (size[0], size[1]))
    return result


img = cv2.imread('samples/16.PNG')

res = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

dst = rotate(res,45,scale=0.5)
cv2.imshow("Resized", dst)
cv2.waitKey(0)
