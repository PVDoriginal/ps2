import cv2
import numpy as np
import cv2 as cv
from main import *

image_path = "assets/cezara.png"
get_image = lambda: cv2.imread(image_path, cv2.IMREAD_COLOR)


def smoothen(img):

    # smooths image by a deviation of 0.2
    img = apply_gauss(img, 1, 150, 0.2)

    cv2.imwrite("assets/smoothed.jpg", img)

    cv2.imshow("Cezara", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


smoothen(get_image())


