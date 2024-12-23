import numpy as np
import cv2
from main import *

image_path = "assets/cezara.png"
get_image = lambda: cv2.imread(image_path, cv2.IMREAD_COLOR)


def smoothen(img):
    img = apply_gauss(img, 1, 50, 1)
    cv2.imwrite("assets/smoothed.jpg", img)

    cv2.imshow("Cezara", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    smoothen(get_image())


