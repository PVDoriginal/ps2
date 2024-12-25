import numpy as np
import cv2
from gauss import *

get_image = lambda name: cv2.imread(f"assets/{name}", cv2.IMREAD_COLOR)


def smoothen(name, ext):
    img = get_image(f"{name}.{ext}")
    img = apply_gauss(img, 1, 50, 3)

    cv2.imwrite(f"assets/{name}_smoothened.png", img)
    cv2.imshow(name, img)


def compress(name, ext):
    img = get_image(f"{name}.{ext}")
    img = apply_gauss(img, 10, 100, 10)

    cv2.imwrite(f"assets/{name}compressed.png", img)
    cv2.imshow(name, img)


def expand(name, ext):
    img = get_image(f"{name}.{ext}")
    img = cv2.resize(img, (0, 0), fx=5, fy=5)
    img = apply_gauss(img, 1, 50, 2.5)

    cv2.imwrite(f"assets/{name}_expanded.png", img)
    cv2.imshow(name, img)


if __name__ == '__main__':

    # smoothen("cezara", "png")
    # expand("cezara", "png")
    # compress("sipos", "jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

