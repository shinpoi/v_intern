# encode: utf-8
# Python3.5

import numpy as np
import logging
from ctypes import *

import setting

move_point = cdll.LoadLibrary('./move_point.so').move_point
logging.basicConfig(level=setting.LOG_LEVEL,
                    format='[%(levelname)s]   \t %(asctime)s \t%(message)s\t',
                    datefmt='%Y/%m/%d (%A) - %H:%M:%S',
                    )

data_path = setting.DATA_PATH
if not data_path.endswith('/'):
    data_path += '/'


###########################
# once function

# read original data -->  trans data to array like data[user][item] = rating -->  save data as .npy files.
def data2npy(src_name='u.data', save_name='data.npy'):
    with open(data_path+src_name) as f:
        data = []
        while 1:
            line = f.readline()
            if not line:
                break
            data.append([int(i) for i in line[:-1].split('\t')])

        user = set([line[0] for line in data])
        item = set([line[1] for line in data])
        logging.debug("number of user: %d" % len(user))
        logging.debug("number of item: %d" % len(item))

        npy_data = np.zeros((len(user), len(item)), dtype=np.int32)
        for line in data:
            npy_data[line[0]-1][line[1]-1] = line[2]

        np.save(save_name, npy_data)
        logging.info("np.array is saved as %s. shape: (%d, %d)" % (save_name, npy_data.shape[0], npy_data.shape[1]))

        return npy_data

# end once function
###########################


def load_data(name='data.npy'):
    logging.info("load datafile: %s" % name)
    try:
        data = np.load(name)
    except FileNotFoundError:
        logging.warning("not found datafile. try to load original data file..." % name)
        data = data2npy(save_name=name)
    return data


# need two 1-dim np.arrays x and y.
def sim_pearson(x, y):
    if len(x) != len(y):
        logging.error("len(x) != len(y)")
        raise ValueError

    # item be rated by both people.
    item = []
    for i in range(len(x)):
        if x[i] != 0 and y[i] != 0:
            item.append(i)

    n = len(item)
    if n == 0: return 0

    sum_x = sum([x[i] for i in item])
    sum_y = sum([y[i] for i in item])
    sum_xy = sum([x[i]*y[i] for i in item])

    sum_xsq = sum([x[i]**2 for i in item])
    sum_ysq = sum([y[i]**2 for i in item])

    num = n*sum_xy - sum_x*sum_y
    # a = (n*sum_xsq - sum_x**2)
    # b = (n*sum_ysq - sum_y**2)
    # print(a, b)
    # den = np.sqrt(a*b)
    den = np.sqrt((n*sum_xsq - sum_x**2) * (n*sum_ysq - sum_y**2))

    if den == 0: return 0

    return num/den


# for test
def sq_distance(x, y):
    return ((x-y)**2).mean()


def k_means(data, k, distance=sim_pearson, bias=1):
    user_len = data.shape[0]
    item_len = data.shape[1]

    # init points
    points = np.zeros((k, item_len), dtype=np.float32)
    for i in range(k):
        # point value: 0~5
        points[i] = [np.random.randint(6) for j in range(item_len)]

    last_matches = [[] for i in range(k)]
    logging.debug("points init complete.")

    loop = 0
    while 1:
        loop += 1
        # matches
        logging.info("matching... loop: %d" % loop)
        matches = [[] for i in range(k)]
        for i in range(user_len):
            min_distance = 9999
            min_point = k+1
            for j in range(k):
                d = abs(distance(data[i], points[j])-bias)
                if d < min_distance:
                    min_distance = d
                    min_point = j
            matches[min_point].append(i)

        if matches == last_matches:
            logging.info("cluster complete! loop: %d" % loop)
            break
        else:
            logging.debug("clusting... loop: %d" % loop)
            last_matches = matches

        # move point(center of group)
        # for every point
        logging.info("move points... loop: %d" % loop)

        # move point by Ctypes
        # init data
        matches_np = np.zeros((k, user_len), dtype=np.int32)
        matches_len = np.zeros(k, dtype=np.int32)
        for i in range(k):
            length = len(matches[i])
            matches_len[i] = length
            matches_np[i, :length] = matches[i]

        logging.debug("move points by c-functions... loop: %d" % loop)
        # void move_point(int data[], int user_len, int item_len, int k, int matches[], int matches_len[], float points[])
        move_point(data.ctypes.data_as(POINTER(c_int)),
                   user_len, item_len, k,
                   matches_np.ctypes.data_as(POINTER(c_int)),
                   matches_len.ctypes.data_as(POINTER(c_int)),
                   points.ctypes.data_as(POINTER(c_float)))
        """
        # move point by Python
        for i in range(k):
            # for every item
            for j in range(item_len):
                num = 0.0
                sum_ = []
                # average rating of group's users
                for user in matches[i]:
                    sum_.append(data[user][j])
                    if data[user][j] != 0:
                        num += 1
                points[i][j] = sum(sum_)/num
        """

    # logging.debug("k points: %s" % str(points))
    return points
    pass


###########################
# run

if __name__ == "__main__":
    # data2npy()
    """
    print(sim_pearson([], []))
    print(sim_pearson([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))
    print(sim_pearson([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]))
    print(sim_pearson([1, 0, 0, 0, 1], [0, 1, 1, 1, 10]))
    print(sim_pearson([1, 2, 5, 4, 3], [6, 1, 5, 7, 9]))
    print(sim_pearson([1, 1, 2, 1, 1], [1, 1, 2, 1, 1]))
    print(sim_pearson([1, 2, 1, 0, 0], [1, 2, 0, 0, 0]))
    """

    data_set = load_data()
    # ps = k_means(data_set, 5, distance=sq_distance, bias=0)
    ps = k_means(data_set, 5)
    print("k-means end!")
    for core in ps:
        print(str(core[:100]))
