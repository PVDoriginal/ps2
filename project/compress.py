import numpy as np
import cv2
from main import *

image_path = "assets/sipos.jpg"
get_image = lambda: cv2.imread(image_path, cv2.IMREAD_COLOR)


def compress(img):

    # compresses image by a scale of 5 with slight smoothing
    img = apply_gauss(img, 1, 50, 1)

    cv2.imwrite("assets/compressed.jpg", img)

    cv2.imshow("Sipos", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    compress(get_image())


