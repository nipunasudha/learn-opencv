import cv2
import sys
import operator

# Shared variable declarations
refPts = []
image = 1
numSelected = 0
clone = 1
onDrag = 0
font = cv2.FONT_HERSHEY_SIMPLEX


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
        render_overlays()
        numSelected = numSelected + 1


def getSelectionsFromImage(img):
    global image, refPts, numSelected, clone
    image = img
    refPts = []  # Reinit refPts
    clone = image.copy()
    render_instructions(clone)
    cv2.namedWindow("© 2017 Syntac Inc. Palwatte Distillery Monitor")
    cv2.setMouseCallback("© 2017 Syntac Inc. Palwatte Distillery Monitor", click_handler)
    cv2.imshow("© 2017 Syntac Inc. Palwatte Distillery Monitor", clone)
    while True:
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
            return None
    if ((len(refPts) % 2) == 0):
        print(len(refPts) / 2)
        print(refPts)
        for selection in range(0, int(len(refPts) / 2)):
            roi = image[refPts[0 + (selection * 2)][1]:refPts[1 + (selection * 2)][1],
                  refPts[0 + (2 * selection)][0]:refPts[1 + (2 * selection)][0]]
            cv2.imshow("ROI" + str(selection), roi)
    else:
        sys.exit("Selection Capture didn't get an even number of bounding points.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return refPts


img = cv2.imread('samples/34.PNG')
co = getSelectionsFromImage(img)
print(co)
