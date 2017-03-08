# ============================================================================
# importing openCV

import numpy as np
import cv2

# ============================================================================
# opening an image from disk

img = cv2.imread("./samples/20.PNG", 0)

# ============================================================================
# showing a loaded image

cv2.imshow("haiyo", img)

# ============================================================================
# show image using matplotlib

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('samples/03.PNG', 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# ============================================================================
# wait for a keypress event -> if 0 is passed, waits indefinitely

cv2.waitKey(0)
____________________

k = cv2.waitKey(0) & 0xFF
if k == 27:  # wait for ESC key to exit
#######
elif k == ord('s'):  # wait for 's' key to save and exit
#######

# ============================================================================
# destroy all open windows

cv2.destroyAllWindows()

# ============================================================================
# move open windows

cv2.moveWindow("processed", 550, 50)

# ============================================================================
# destroy a specific window

cv2.destroyWindow("windowName")

# ============================================================================
# create empty named window

cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # or cv2.WINDOW_AUTOSIZE

# ============================================================================
# save an image to disk

cv2.imwrite('messigray.png', img)

# ============================================================================
# Convert webcam feed to B&W & display

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 320)  # these are optional
cap.set(4, 240)
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# ============================================================================
# Create a black image
img = np.zeros((512, 512, 3), np.uint8)

# ============================================================================
# DRAWING SHAPES ON AN IMAGE
# Create a black image
img = np.zeros((512, 512, 3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
img = cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
img = cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
# polygon drawing
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))
img = cv2.polylines(img, [pts], True, (0, 255, 255))

cv2.imshow("Drawing", img)
cv2.waitKey(0)

# ============================================================================
# Adding text to Images
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

# ============================================================================
# Color picker demo
import cv2
import numpy as np


def nothing(x):
    pass


# Create a black image, a window
img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

while (1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    s = cv2.getTrackbarPos(switch, 'image')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]

cv2.destroyAllWindows()

# ============================================================================
# Accessing pixel values
px = img[100, 100]
print(px)
# [157 166 200]

# accessing only blue pixel
blue = img[100, 100, 0]
print(blue)
# 157

# ============================================================================
# Modifying pixel values
img[100, 100] = [255, 255, 255]
print(img[100, 100])
# [255 255 255]

# ============================================================================
# A better way of pixel manipulation

# accessing RED value
img.item(10, 10, 2)
59

img.itemset((10, 10, 2), 100)
img.item(10, 10, 2)
100

# ============================================================================
# Accessing image properties
print(img.shape)
print(img.size)
print(img.dtype)

# ============================================================================
# REGION OF IMAGE
ball = img[280:340, 330:390]  # select and name a region of an image
img[273:333, 100:160] = ball  # paste that defined part on another region

# ============================================================================
# CREATE A COPY OF IMAGE
img_cropped = img_ori[146:201, 48: 304].copy()

# ============================================================================
# Splitting & merging chanels
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))
# -----------
b = img[:, :, 0]  # select blue chanel
img[:, :, 2] = 0  # set red to 0

# ============================================================================
# Image adding & blending
resultImg = cv2.add(A, B)
resultImg = cv2.addWeighted(A, 0.7, B, 0.3, 0)  # last parameter is like total offset (brightness)

# ============================================================================
# Bitwise operations
img = cv2.bitwise_not(img)  # invert image
img = cv2.bitwise_and(A, B, mask=mask_inv)

# ============================================================================
# Object Tracking using bitwise() & inRange()
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while (1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

# ============================================================================
# Finding HSV from RGB (checkout shortcut library)
green = np.uint8([[[0, 255, 0]]])
hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
print(hsv_green)
# [[[ 60 255 255]]]

# ============================================================================
# SIMPLE THRESHOLDING
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('processing/bookpage.png', 0)
img = cv2.medianBlur(img, 5)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                            cv2.THRESH_BINARY, 11, 2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# ============================================================================
# ADAPTIVE THRESHOLDING
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('processing/bookpage.png', 0)
img = cv2.medianBlur(img, 5)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                            cv2.THRESH_BINARY, 11, 2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# ============================================================================
# OTSU'S BINARIZATION
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('processing/bookpage.png', 0)

# global thresholding
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu's thresholding
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

for i in range(3):
    plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
plt.show()

# ============================================================================
# Image resizing / scaling
res = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
# OR
res = cv2.resize(img, (newWidth, newHeight), interpolation=cv2.INTER_CUBIC)

# ============================================================================
# Image translation
M = np.float32([[1, 0, 100], [0, 1, 50]])  # build transformation matrix
result = cv2.warpAffine(img, M, (cols, rows))
# OR use shortcut
dst = translate(res, 100, 50)

# ============================================================================
# Image rotation
M = cv2.getRotationMatrix2D(origin, angle, scale)
result = cv2.warpAffine(res, M, (size[0], size[1]))
# OR use shortcut
dst = rotate(res, 45, scale=0.5)

# ============================================================================
# Affine Transformation
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

M = cv2.getAffineTransform(pts1, pts2)

dst = cv2.warpAffine(img, M, (cols, rows))

# ============================================================================
# Perspective Transformation
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (300, 300))

# ============================================================================
# FILTERING
blur = cv2.blur(img, (5, 5))  # (image,boxsize)

blur = cv2.GaussianBlur(img, (5, 5), 0)  # (image,blurRadius, blurAmount)

median = cv2.medianBlur(img, 5)  # (image, amount)

blur = cv2.bilateralFilter(img, 9, 75, 75)

# ============================================================================
# Morphological Transformations
kernel = np.ones((5, 5), np.uint8)  # building a kernel, a unit of operation
# --------------------------------
erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)  # difference btwn erosion-dilation

# ============================================================================
# Image gradients
img = cv2.imread('samples/16.PNG', 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# ============================================================================
# Edge detection
# canny
edges = cv2.Canny(frame, 100, 300)  # frame, min, max

# ============================================================================
# CONTOUR
# Finding & drawing contours
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image = cv2.drawContours(im, allContours, -1, (0, 255, 255), 2)
# OR
image = cv2.drawContours(im, [singleContour], 0, (0, 255, 255), 2)

# Straight Bounding Rectangle
x, y, w, h = cv2.boundingRect(contours[1])
image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Rotated Rectangle
rect = cv2.minAreaRect(contours[6])
box = cv2.boxPoints(rect)
box = np.int0(box)
im = cv2.drawContours(im, [box], 0, (0, 0, 255), 2)

# Fitting an Ellipse
ellipse = cv2.fitEllipse(contours[18])
im = cv2.ellipse(im, ellipse, (0, 0, 255), 2)

# Approximation
epsilon = 0.05 * cv2.arcLength(singleContour, True)
singleContour = cv2.approxPolyDP(singleContour, epsilon, True)

# Properties/functions
Aspect
Ratio
Extent
Solidity
Equivalent
Diameter
Orientation
Mask and Pixel
Points
Mean
Color or Mean
Intensity
Extreme
Points
Convexity
Defects
Point
Polygon
Test!!!
Match
Shapes!!!



# ============================================================================
# Contour Hierarchy
Learn
about
it
later, it is the
hierarchy
of
detected
contour
objects, like
layers

# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#


# ============================================================================
#
