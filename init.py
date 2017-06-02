# encode: utf-8
# Python3.5

import numpy as np
import logging

import setting

logging.basicConfig(level=setting.LOG_LEVEL,
                    format='[%(levelname)s]   \t %(asctime)s \t%(message)s\t',
                    datefmt='%Y/%m/%d (%A) - %H:%M:%S',
                    )

data_path = setting.DATA_PATH
if not data_path.endswith('/'):
    data_path += '/'


###########################
# once function

# read original data -->  trans data to array like user[item] = rating -->  save data as .npy files.
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

# end once function
###########################


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
    den = np.sqrt((n*sum_xsq - sum_x**2) * (n*sum_ysq - sum_y**2))

    if den == 0: return 0

    return float(num)/den

###########################
# run

if __name__ == "__main__":
    # data2npy()
    print(sim_pearson([], []))
    print(sim_pearson([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))
    print(sim_pearson([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]))
    print(sim_pearson([1, 0, 0, 0, 1], [0, 1, 1, 1, 10]))
    print(sim_pearson([1, 2, 5, 4, 3], [6, 1, 5, 7, 9]))
    print(sim_pearson([1, 1, 2, 1, 1], [1, 1, 2, 1, 1]))
    print(sim_pearson([1, 2, 1, 0, 0], [1, 2, 0, 0, 0]))
