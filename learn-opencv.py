import numpy as np
import cv2

im = cv2.imread('processing/shapes.png')
# im = cv2.imread('samples/07.PNG')

# making Grayscale, blur
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
imgray = cv2.blur(imgray, (5, 5))
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

#Morphological Transformations
kernel = np.ones((1, 1), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#Finding & drawing contours
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(im, contours, -1, (0, 255,255), 2)

# Straight Bounding Rectangle
x, y, w, h = cv2.boundingRect(contours[1])
image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Rotated Rectangle
rect = cv2.minAreaRect(contours[6])
box = cv2.boxPoints(rect)
box = np.int0(box)
im = cv2.drawContours(im, [box], 0, (0, 0, 255), 2)

#Fitting an Ellipse
ellipse = cv2.fitEllipse(contours[18])
im = cv2.ellipse(im,ellipse,(0,0,255),2)

#------------------------
cv2.imshow("image", image)
cv2.waitKey(0)
