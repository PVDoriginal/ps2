import numpy as np
import cv2 as cv


def generate_norms_2d(n, deviation):
    theta = np.random.random(n) * np.pi * 2
    r = -2 * np.log(np.random.random(n))

    x = np.multiply(np.sqrt(r), np.cos(theta)) * deviation
    y = np.sqrt(r) * np.sin(theta) * deviation

    return np.dstack((x, y))[0]


def apply_gauss(img, compression=1, norm_count=100, deviation=1):
    norms = generate_norms_2d(norm_count, deviation)

    # get list of rows based on compression
    rows = np.linspace(0, img.shape[0]-1, int(img.shape[0]/compression), dtype=int)

    # get list of columns for each row based on compression
    cols = list()
    for _ in rows:
        cols.append(np.linspace(0, img.shape[1]-1, int(img.shape[1]/compression), dtype=int))

    new_img = np.empty(shape=(len(rows), len(cols[0]), img.shape[2]), dtype=np.uint8)
    for i in range(new_img.shape[0]):
        print(str(round((i / new_img.shape[0]) * 100, 2)) + "%")

        for j in range(new_img.shape[1]):

            points = [img[rows[i], cols[i][j]]]

            for norm in norms:
                ii = rows[i] + int(norm[0])
                jj = cols[i][j] + int(norm[1])

                if 0 < ii < img.shape[0] and 0 < jj < img.shape[1]:
                    points.append(img[ii, jj])

            new_img[i, j] = np.mean(np.array(points), axis=0)

    return new_img

