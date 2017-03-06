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


# convert BGR to HSV
def BGR2HSV(r, g, b):
    return cv2.cvtColor(np.uint8([[[0, 255, 0]]]), cv2.COLOR_BGR2HSV)


# convert HSV to RGB
def BGR2HSV(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)


# image translation shortcut (improved)
def translate(img, offset, size=(-1, -1)):
    Irows, Icols, ch = img.shape
    if (size == (-1, -1)):
        size = (Icols, Irows)
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    result = cv2.warpAffine(res, M, (size[0], size[1]))
    return result


# Image rotation shortcut
def rotate(img, angle, origin=('x', 'x'), size=(-1, -1), scale=1.0):
    Irows, Icols, ch = img.shape

    if (size == (-1, -1)):
        size = (Icols, Irows)

    if (origin == ('x', 'x')):
        origin = ((size[0] / 2, size[1] / 2))

    M = cv2.getRotationMatrix2D(origin, angle, scale)
    result = cv2.warpAffine(res, M, (size[0], size[1]))
    return result
