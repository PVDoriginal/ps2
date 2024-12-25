import numpy as np
import cv2
from main import *

image_path = "assets/cezara.png"
get_image = lambda: cv2.imread(image_path, cv2.IMREAD_COLOR)


def expand(img):
    img = cv2.resize(img, (0, 0), fx=5, fy=5)
    img = apply_gauss(img, 1, 50, 2.5)

    cv2.imwrite("assets/expanded.jpg", img)

    cv2.imshow("Cezara", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    expand(get_image())
