# ================================================================
import sys
import numpy as np
import PIL
from PIL import Image
import pyocr
import cv2
import pyocr.builders


# -------------------------------------------------------------
# OCR initialization example with pyOCR
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


# convert PIL image to OPENCV matrix
def PIL2CV(input_img):
    open_cv_image = np.array(input_img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


# convert OPENCV matrix to PIL image
def CV2PIL(input_img):
    return Image.fromarray(input_img)


# show openCV image
def showImageCV(input_img, title=""):
    cv2.imshow(title, input_img)
    cv2.waitKey(0)
