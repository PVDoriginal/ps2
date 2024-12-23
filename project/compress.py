import cv2
import numpy as np
import cv2 as cv
from main import *

image_path = "assets/sipos.jpg"
get_image = lambda: cv2.imread(image_path, cv2.IMREAD_COLOR)


def compress(img):

    # compresses image by a scale of 5 with slight smoothing
    img = apply_gauss(img, 5, 50, 1)

    cv2.imwrite("assets/compressed.jpg", img)

    cv2.imshow("Sipos", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


compress(get_image())


