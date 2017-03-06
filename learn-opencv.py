import cv2
import numpy as np

img = cv2.imread("processing/bluedot.jpg")

a = img[:, :, 1]
b = img[:, :, 0]
c = np.flip(b, 1)

d = cv2.add(a, c)

cv2.imshow("Just Add", d)
cv2.waitKey(0)

e = cv2.addWeighted(a, 0.7, c, 0.3, 0)
#last parameter is like total offset (brightness)
cv2.imshow("Weighted Add", cv2.bitwise_not(e))
cv2.waitKey(0)
