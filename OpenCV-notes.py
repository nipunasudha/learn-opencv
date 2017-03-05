# importing openCV

import numpy as np
import cv2

# opening an image from disk

img = cv2.imread("./samples/20.PNG", 0)

# showing a loaded image

cv2.imshow("haiyo", img)

# show image using matplotlib

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('samples/03.PNG', 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# wait for a keypress event -> if 0 is passed, waits indefinitely

cv2.waitKey(0)
____________________

k = cv2.waitKey(0) & 0xFF
if k == 27:  # wait for ESC key to exit
#######
elif k == ord('s'):  # wait for 's' key to save and exit
#######

# destroy all open windows

cv2.destroyAllWindows()

# destroy a specific window

cv2.destroyWindow("windowName")

# create empty named window

cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # or cv2.WINDOW_AUTOSIZE

# save an image to disk

cv2.imwrite('messigray.png', img)
