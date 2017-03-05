import cv2
import numpy as np

img = cv2.imread("samples/32.PNG")

img[:, :, 1] = 0  # set green chanel to 0

cv2.imshow("Image", img)
cv2.waitKey(0)
