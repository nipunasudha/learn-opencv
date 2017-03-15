import cv2
import sys
import operator

# Shared variable declarations
refPts = []
image = 1
numSelected = 0
clone = 1


def render_boundingboxes():
    global image
    local_clone = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index in range(int(len(refPts) / 2)):
        cv2.rectangle(local_clone, refPts[index * 2], refPts[(index * 2) + 1], (0, 255, 0), 1, lineType=8)
        cv2.rectangle(local_clone, tuple(map(operator.add, refPts[(index * 2)], (-1, -1))),
                      tuple(map(operator.add, refPts[(index * 2) + 1], (1, 1))), (0, 0, 0), 1, lineType=4)
        cv2.putText(local_clone, 'A ' + str(index + 1), tuple(map(operator.add, refPts[(index * 2)], (0, -3))), font,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("Select areas to monitor", local_clone)


def click_handler(event, x, y, flags, pram):
    global refPts, image, numSelected

    if (event == cv2.EVENT_LBUTTONDOWN):
        if (len(refPts) == 0):
            refPts = [(x, y)]
        else:
            refPts.append((x, y))
    elif (event == cv2.EVENT_LBUTTONUP):
        refPts.append((x, y))
        render_boundingboxes()
        numSelected = numSelected + 1


def getSelectionsFromImage(img):
    global image, refPts, numSelected, clone
    image = img
    refPts = []  # Reinit refPts
    clone = image.copy()
    cv2.namedWindow("Select areas to monitor")
    cv2.setMouseCallback("Select areas to monitor", click_handler)
    cv2.imshow("Select areas to monitor", image)
    while True:
        # cv2.imshow("Select areas to monitor", image)
        key = cv2.waitKey(1) & 0xFF
        if (key == ord("d")):
            break
        if (key == ord("r")):
            if ((len(refPts) % 2) == 0):
                refPts = refPts[:len(refPts) - 2]
                numSelected = numSelected - 1
                render_boundingboxes()
        if (key == 27):
            return None
    if ((len(refPts) % 2) == 0):
        print(len(refPts) / 2)
        print(refPts)
        for selection in range(0, int(len(refPts) / 2)):
            roi = clone[refPts[0 + (selection * 2)][1]:refPts[1 + (selection * 2)][1],
                  refPts[0 + (2 * selection)][0]:refPts[1 + (2 * selection)][0]]
            cv2.imshow("ROI" + str(selection), roi)
    else:
        sys.exit("Selection Capture didn't get an even number of bounding points.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return refPts


img = cv2.imread('samples/17.PNG')
co = getSelectionsFromImage(img)
print(co)
