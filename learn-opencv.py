import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread("./samples/20.PNG",0)
cv2.imshow("haiyo",img)
cv2.waitKey(0)
cv2.imwrite('samples/messigray.png',img)
cv2.destroyAllWindows()