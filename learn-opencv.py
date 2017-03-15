import numpy as np
import cv2
import sys
import pyocr.builders
import pyocr.tesseract
import PIL
from PIL import Image


# convert OPENCV matrix to PIL image
def CV2PIL(input_img):
    return Image.fromarray(input_img)


def initTool():
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)

    tool = tools[0]
    print("Will use tool '%s'" % (tool.get_name()))
    langs = tool.get_available_languages()
    print("Available languages: %s" % ", ".join(langs))
    lang = langs[1]
    print("Will use lang '%s'" % (lang))
    return tool


tool = initTool()
lang = tool.get_available_languages()[2]

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
# ------------------------
txt = tool.image_to_string(
    CV2PIL(img_cropped),
    lang=lang,
    builder=pyocr.tesseract.DigitBuilder()
)
print(txt)

# ------------------------
cv2.imshow("original", img_marked)
cv2.imshow("processed", img_cropped)
cv2.moveWindow("original", 50, 50)
cv2.moveWindow("processed", 550, 50)
cv2.waitKey(0)
