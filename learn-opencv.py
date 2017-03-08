import numpy as np
import cv2

img_ori = cv2.imread('processing/digits copy.jpg')
img_cropped = img_ori[146:201, 48: 304].copy()
img_marked = cv2.rectangle(img_ori, (48, 146), (304, 201), (0, 255, 0), 1)

img_cropped = cv2.medianBlur(img_cropped, 5)
img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
img_cropped = cv2.bitwise_not(img_cropped)
img_cropped = cv2.adaptiveThreshold(img_cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# ------------------------
kernel = np.ones((3, 3), np.uint8)  # building a kernel, a unit of operation
# --------------------------------
# img_cropped = cv2.morphologyEx(img_cropped, cv2.MORPH_OPEN, kernel)
img_cropped = cv2.morphologyEx(img_cropped, cv2.MORPH_CLOSE, kernel)
# ------------------------
cv2.imshow("original", img_marked)
cv2.imshow("processed", img_cropped)
cv2.moveWindow("original", 50, 50)
cv2.moveWindow("processed", 550, 50)
cv2.waitKey(0)

# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# cv2.imshow("original", th3)
# cv2.waitKey(0)
