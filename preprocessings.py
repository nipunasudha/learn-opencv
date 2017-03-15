# ============================= ONE (light red chars on dark red background)===============

# img_ori = cv2.imread('processing/digits copy.jpg')
img_ori = cv2.imread('processing/digits-dot.jpg')
img_cropped = img_ori[146:201, 48: 304].copy()
img_marked = cv2.rectangle(img_ori, (48, 146), (304, 201), (0, 255, 0), 1)

img_cropped = cv2.medianBlur(img_cropped, 5)
img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
img_cropped = cv2.bitwise_not(img_cropped)
img_cropped = cv2.adaptiveThreshold(img_cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
cv2.imshow("after threshold", img_cropped)
cv2.moveWindow("after threshold", 50, 650)
# ------------------------
kernel = np.ones((3, 3), np.uint8)  # building a kernel, a unit of operation
# --------------------------------
img_cropped = cv2.morphologyEx(img_cropped, cv2.MORPH_CLOSE, kernel)
img_cropped = cv2.morphologyEx(img_cropped, cv2.MORPH_OPEN, kernel)
# ------------------------

# ============================= TWO (pure red char on black)================================

img_cropped = cv2.medianBlur(img_cropped, 5)
# img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 50, 50])
upper_red = np.array([15, 255, 255])

img_cropped = cv2.inRange(img_cropped, lower_red, upper_red)

img_cropped = cv2.bitwise_not(img_cropped)
# ------------------------
kernel = np.ones((3, 3), np.uint8)  # building a kernel, a unit of operation
# --------------------------------
img_cropped = cv2.morphologyEx(img_cropped, cv2.MORPH_CLOSE, kernel)
img_cropped = cv2.morphologyEx(img_cropped, cv2.MORPH_OPEN, kernel)
# ------------------------

#============================== THREE (too bright digits, off segments visible)

img_cropped = cv2.medianBlur(img_cropped, 5)
lower_red = np.array([0, 0, 200])
upper_red = np.array([50, 255, 255])

ret, img_cropped = cv2.threshold(img_cropped, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("after threshold", img_cropped)
cv2.moveWindow("after threshold", 50, 650)
img_cropped = cv2.bitwise_not(img_cropped)
img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
ret, img_cropped = cv2.threshold(img_cropped, 127, 255, cv2.THRESH_BINARY)
# ------------------------
kernel = np.ones((3, 3), np.uint8)  # building a kernel, a unit of operation
# --------------------------------
img_cropped = cv2.morphologyEx(img_cropped, cv2.MORPH_CLOSE, kernel)
img_cropped = cv2.morphologyEx(img_cropped, cv2.MORPH_OPEN, kernel)