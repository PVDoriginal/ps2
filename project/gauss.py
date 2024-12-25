import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

thread_count = 10


def generate_norms(n, deviation):
    theta = np.random.random(n) * np.pi * 2
    r = -2 * np.log(np.random.random(n))

    x = np.multiply(np.sqrt(r), np.cos(theta)) * deviation
    # y = np.sqrt(r) * np.sin(theta) * deviation

    return x


def apply_gauss(img, compression=1, norm_count=100, deviation=1):
    norms = generate_norms(norm_count, deviation)

    # get new list of rows based on compression
    rows = np.linspace(0, img.shape[0]-1, int(img.shape[0]/compression), dtype=int)

    # get new list of columns based on compression
    cols = np.linspace(0, img.shape[1]-1, int(img.shape[1]/compression), dtype=int)

    # create a shared memory for threads to write the results in
    a = np.empty(shape=(len(rows), len(cols), img.shape[2]), dtype=np.uint8)
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes, name="img")
    new_img = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)

    # split line indexes into sets
    lines = np.arange(0, len(rows))
    lines = np.array_split(lines, thread_count)

    threads = list()

    # create a thread for each set of lines
    for line_set in lines:
        thread = mp.Process(target=apply_gauss_line_thread, args=(line_set, len(cols), img, rows, cols, norms,))
        threads.append(thread)
        thread.start()

    # wait for all threads to finish
    for thread in threads:
        thread.join()

    # clone image locally (in order to free shared memory)
    new_img2 = np.array(new_img)

    shm.close()
    shm.unlink()

    return new_img2


def apply_gauss_line_thread(lines, width, img, rows, cols, norms):

    # open shared memory
    existing_shm = shared_memory.SharedMemory(name='img')
    new_img = np.ndarray(shape=(len(rows), len(cols), img.shape[2]), dtype=np.uint8, buffer=existing_shm.buf)

    # compute values for all given lines
    for i in lines:
        for j in range(width):
            points = [img[rows[i], cols[j]]]

            for norm in norms:
                ii = rows[i] + int(norm)
                jj = cols[j] + int(norm)

                if 0 <= ii < img.shape[0] and 0 <= jj < img.shape[1]:
                    points.append(img[ii, jj])

            # set element as mean of surrounding elements
            new_img[i, j] = np.mean(np.array(points), axis=0)

        # show progress
        if lines[0] == 0:
            print("progress: " + str(round((i / lines[-1]) * 100, 2)) + "%")

    existing_shm.close()

