import numpy as np
import multiprocessing as mp


thread_count = 10


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

    # create a list of threads and a return dict
    threads = list()
    manager = mp.Manager()
    return_dict = manager.dict()

    # divide lines in buckets
    lines = np.arange(0, len(rows))
    lines = np.array_split(lines, thread_count)

    # assign each thread to a bucket
    for line_set in lines:
        thread = mp.Process(target=apply_gauss_line_thread, args=(line_set, len(cols[0]), img.shape[2], img, rows, cols, norms, return_dict,))
        threads.append(thread)
        thread.start()

    # wait for all threads to finish
    for thread in threads:
        thread.join()

    # return dict as image
    return np.array(return_dict.values())


def apply_gauss_line_thread(lines, width, col_count, img, rows, cols, norms, return_dict):

    # compute values for all given lines
    for i in lines:
        line = np.empty(shape=[width, col_count], dtype=np.uint8)

        for j in range(width):
            points = [img[rows[i], cols[i][j]]]

            for norm in norms:
                ii = rows[i] + int(norm[0])
                jj = cols[i][j] + int(norm[1])

                if 0 <= ii < img.shape[0] and 0 <= jj < img.shape[1]:
                    points.append(img[ii, jj])

            line[j] = np.mean(np.array(points), axis=0)
            #line[j] = img[rows[i]][cols[i][j]]
        return_dict[i] = line

        if lines[0] == 0:
            print("progress: " + str(round((i / lines[-1]) * 100, 2)) + "%")

