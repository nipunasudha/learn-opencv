import cv2
import sys
import operator

# Shared variable declarations
refPts = []
sizeThreshold = 8
image = 1
numSelected = 0
clone = 1
onDrag = 0
font = cv2.FONT_HERSHEY_SIMPLEX


def make_positive(tup):
    x, y = tup
    if x < 0: x = 0
    if y < 0: y = 0
    return (x, y)


def correct_coordinates():
    global sizeThreshold
    for index in range(int(len(refPts) / 2)):
        x1, y1 = make_positive(refPts[(index * 2)])
        x2, y2 = make_positive(refPts[(index * 2) + 1])
        if (abs(x1 - x2) < sizeThreshold or abs(y1 - y2) < sizeThreshold):
            del refPts[(index * 2)]
            del refPts[(index * 2)]
            return 0
        if x1 > x2:
            if y1 > y2:
                refPts[(index * 2)] = (x2, y2)
                refPts[(index * 2) + 1] = (x1, y1)
            else:
                refPts[(index * 2)] = (x2, y1)
                refPts[(index * 2) + 1] = (x1, y2)
        else:
            if y1 > y2:
                refPts[(index * 2)] = (x1, y2)
                refPts[(index * 2) + 1] = (x2, y1)
            else:
                refPts[(index * 2)] = (x1, y1)
                refPts[(index * 2) + 1] = (x2, y2)


def render_instructions(image):
    global font, onDrag
    if (onDrag):
        cv2.rectangle(image, (10, 10), (100, 30), (0, 0, 0), cv2.FILLED, lineType=8)
        cv2.putText(image, 'SELECTING', (15, 25), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(image, (0, image.shape[0] - 20), (image.shape[1], image.shape[0]), (0, 0, 0), cv2.FILLED, lineType=8)
    cv2.putText(image, 'PRESS R TO UNDO, D TO SAVE, ESC TO CANCEL', (0, image.shape[0] - 7), font, 0.5, (255, 255, 255),
                1, cv2.LINE_AA)


def render_overlays():
    global image, font
    local_clone = image.copy()
    render_instructions(local_clone)

    for index in range(int(len(refPts) / 2)):
        cv2.rectangle(local_clone, refPts[index * 2], refPts[(index * 2) + 1], (0, 255, 0), 1, lineType=8)
        cv2.rectangle(local_clone, tuple(map(operator.add, refPts[(index * 2)], (-1, -1))),
                      tuple(map(operator.add, refPts[(index * 2) + 1], (1, 1))), (0, 0, 0), 1, lineType=8)
        cv2.putText(local_clone, 'A ' + str(index + 1), tuple(map(operator.add, refPts[(index * 2)], (0, -3))), font,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("© 2017 Syntac Inc. Palwatte Distillery Monitor", local_clone)


def click_handler(event, x, y, flags, pram):
    global refPts, image, numSelected, onDrag

    if (event == cv2.EVENT_LBUTTONDOWN):
        onDrag = 1
        render_overlays()
        if (len(refPts) == 0):
            refPts = [(x, y)]
        else:
            refPts.append((x, y))
    elif (event == cv2.EVENT_LBUTTONUP):
        onDrag = 0
        refPts.append((x, y))
        correct_coordinates()
        render_overlays()
        numSelected = numSelected + 1


def getSelectionsFromImage(img):
    keepRunning = 1
    global image, refPts, numSelected, clone
    image = img
    refPts = []  # Reinit refPts
    clone = image.copy()
    render_instructions(clone)
    cv2.namedWindow("© 2017 Syntac Inc. Palwatte Distillery Monitor", flags=cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("© 2017 Syntac Inc. Palwatte Distillery Monitor", click_handler)
    cv2.imshow("© 2017 Syntac Inc. Palwatte Distillery Monitor", clone)
    while keepRunning:
        # cv2.imshow("© 2017 Syntac Inc. Palwatte Distillery Monitor", image)
        key = cv2.waitKey(1) & 0xFF
        if (key == ord("d")):
            break
        if (key == ord("r")):
            if ((len(refPts) % 2) == 0):
                refPts = refPts[:len(refPts) - 2]
                numSelected = numSelected - 1
                render_overlays()
        if (key == 27):
            keepRunning = 0
            cv2.destroyAllWindows()
            return []
    if ((len(refPts) % 2) == 0 and len(refPts) != 0 and keepRunning == 1):
        # print(len(refPts) / 2)
        # print(refPts)
        for selection in range(0, int(len(refPts) / 2)):
            roi = image[refPts[0 + (selection * 2)][1]:refPts[1 + (selection * 2)][1],
                  refPts[0 + (2 * selection)][0]:refPts[1 + (2 * selection)][0]]
            cv2.imshow("ROI" + str(selection), roi)

        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return refPts

# img = cv2.imread('samples/37.PNG')
# co = getSelectionsFromImage(img)
# print(co)
