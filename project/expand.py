import numpy as np
import cv2
from main import *

image_path = "assets/cezara.png"
get_image = lambda: cv2.imread(image_path, cv2.IMREAD_COLOR)


def expand(img):

    # expands image by a factor of 4 and smooths is
    img = cv.resize(img, (0, 0), fx=4, fy=4)
    img = apply_gauss(img, 1, 100, 1.5)

    cv2.imwrite("assets/expanded.jpg", img)

    cv2.imshow("Cezara", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


expand(get_image())
