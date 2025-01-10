import numpy as np
import cv2
import cv2.data

from gauss import *

from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000, deltaE_ciede94, deltaE_cie76, deltaE_cmc


def get_image(name):
    return cv2.imread(f"assets/{name}", cv2.IMREAD_COLOR)


def smoothen(name, ext):
    img = get_image(f"{name}.{ext}")
    img = apply_gauss(img, 1, 50, 6)

    cv2.imwrite(f"assets/{name}_smoothened.png", img)
    cv2.imshow(name, img)


def compress(name, ext):
    img = get_image(f"{name}.{ext}")
    img = apply_gauss(img, 10, 0, 0)

    cv2.imwrite(f"assets/{name}_compressed.png", img)
    cv2.imshow(name, img)


def expand(name, ext):
    img = get_image(f"{name}.{ext}")
    img = cv2.resize(img, (0, 0), fx=10, fy=10)
    img = apply_gauss(img, 1, 50, 5)

    cv2.imwrite(f"assets/{name}_expanded.png", img)
    cv2.imshow(name, img)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


def hide_cascade(img, target, compression=1, norm_count=50, deviation=20):
    mask = np.full((img.shape[0], img.shape[1], 1), False)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:

        if 'face' in target:
            mask[y:y + h, x:x + w] = True
            break

        face_roi = gray[y:y+h, x:x+w]
        mask_roi = mask[y:y+h, x:x+w]

        if 'eye' in target:
            eyes = eye_cascade.detectMultiScale(face_roi, 1.2, 10)
            for (ex, ey, ew, eh) in eyes:
                mask_roi[ey:ey + eh, ex:ex + ew] = True

        if 'mouth' in target:
            mouth = mouth_cascade.detectMultiScale(face_roi, 1.3, 10)
            for (ex, ey, ew, eh) in mouth:
                mask_roi[ey:ey + eh, ex:ex + ew] = True

    return apply_gauss(img, compression, norm_count, deviation, mask)


def hide_faces(name, ext):
    img = get_image(f"{name}.{ext}")
    img = hide_cascade(img, 'face')

    cv2.imwrite(f"assets/{name}_masked.png", img)
    cv2.imshow(name, img)


def hide_eyes(name, ext):
    img = get_image(f"{name}.{ext}")
    img = hide_cascade(img, 'eyes')

    cv2.imwrite(f"assets/{name}_masked_eyes.png", img)
    cv2.imshow(name, img)


def hide_eyes_and_mouth(name, ext):
    img = get_image(f"{name}.{ext}")
    img = hide_cascade(img, 'eyesmouth')

    cv2.imwrite(f"assets/{name}_masked_eyes_mouth.png", img)
    cv2.imshow(name, img)


if __name__ == '__main__':

    # smoothen("Sorina-Nicoleta-Predut", "png")
    # compress("sipos", "jpg")
    # expand("cezara", "jpeg")
    # hide_faces("bucataru", "png")
    # hide_eyes("irofti", "jpg")
    hide_eyes_and_mouth("moisil", "jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

