import numpy as np
import cv2

im = cv2.imread('processing/shapes.png')

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
for singleContour in contours:

    #approximation
    epsilon = 0.05*cv2.arcLength(singleContour,True)
    singleContour = cv2.approxPolyDP(singleContour,epsilon,True)

    im = cv2.drawContours(im, [singleContour], 0, (0, 255,255), 2)

#------------------------
cv2.imshow("image", im)
cv2.waitKey(0)
