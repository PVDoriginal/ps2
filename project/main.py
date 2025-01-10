import numpy as np
import cv2
import cv2.data

from gauss import *

from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000, deltaE_ciede94, deltaE_cie76, deltaE_cmc


def get_image(name):
    return cv2.imread(f"assets/{name}", cv2.IMREAD_COLOR)


def smoothen(name, ext):
    img = get_image(f"{name}.{ext}")
    img = apply_gauss(img, 1, 20, 25)

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


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def hide_faces(img, compression=1, norm_count=10, deviation=10):
    mask = np.full((img.shape[0], img.shape[1], 1), False)
    faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.2, 6)
    for (x, y, w, h) in faces:
        print("face detected!")
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        mask[y:y + h, x:x + w] = True
        pass

    return apply_gauss(img, compression, norm_count, deviation, mask)


def hide_faces_image(name, ext):
    img = get_image(f"{name}.{ext}")
    img = hide_faces(img)

    cv2.imwrite(f"assets/{name}_masked.png", img)
    cv2.imshow(name, img)


if __name__ == '__main__':

    # smoothen("moisil", "png")
    # expand("cezara", "png")
    # compress("sipos", "jpg")
    # hide_faces_image("cezara3", "jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

