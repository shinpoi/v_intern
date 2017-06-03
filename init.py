# encode: utf-8
# Python3.5

import numpy as np
import logging
from ctypes import *
from PIL import Image, ImageDraw

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
        logging.warning("not found datafile. try to load original data file...")
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

    return 1.0 - num/den


# for test
def sq_distance(x, y):
    return ((x-y)**2).mean()


# sub-function in k-means. get data and k-means's center, return list[group][users].
def matche_user(data, points, distance=sim_pearson):
    k = points.shape[0]
    user_len = data.shape[0]
    matches = [[] for i in range(k)]
    for i in range(user_len):
        min_distance = 9999
        min_point = k + 1
        for j in range(k):
            d = distance(data[i], points[j])
            if d < min_distance:
                min_distance = d
                min_point = j
        matches[min_point].append(i)
    return matches


# k-means algorithm. get data and k. return np.array of k center.
def k_means(data, k, distance=sim_pearson):
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
        matches = matche_user(data, points, distance=distance)

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

        # move point(euclid distance)
        points = np.zeros((k, item_len), dtype=np.float32)
        for i in range(len(matches)):
            for j in range(len(matches[i])):
                points[i] += data[matches[i][j]]
            points[i] /= len(matches[i])
        if loop > 20:
            break
        """

    # logging.debug("k points: %s" % str(points))
    np.save('k_means_%d.npy' % k, points)
    return points


# create a 2-dim graph of users. return np.array of each users location.
def mds2d(data, distance=sim_pearson, move_rate=0.00005, max_loop=100, save_name="graph.npy"):
    logging.info("function scaledown is runing...")

    user_len = data.shape[0]
    item_len = data.shape[1]

    try:
        real_dis = np.load("real_dis.npy")
        logging.info("load file 'real_dis.npy'...")
    except FileNotFoundError:
        # real distance = pearson distance, shape = (user_len, user_len)
        real_dis = np.zeros((user_len, user_len), dtype=np.float32)
        for i in range(user_len):
            for j in range(user_len):
                real_dis[i][j] = distance(data[i], data[j])
            print("init real distance: %d" % i)

        np.save("real_dis.npy", real_dis)

    # 2-dim location of users, shape = (user_len, 1)
    location = [(np.random.random(), np.random.random()) for i in range(user_len)]
    location = np.array(location, dtype=np.float32)*2

    # euclid distance
    euclid_dis = np.zeros((user_len, user_len), dtype=np.float32)

    logging.debug("scaledown: init complete...")
    last_error = 0
    counter = -1
    for m in range(max_loop):
        # euclid distance
        for i in range(user_len):
            for j in range(i, user_len):
                euclid_dis[i][j] = np.sqrt(sum((location[i]-location[j])**2))

        move = np.zeros((user_len, 2), dtype=np.float32)
        error_sum = 0

        # calculate move distance
        for i in range(user_len):
            for j in range(user_len):
                if i == j: continue
                if real_dis[i][j] == 0:
                    error = euclid_dis[i][j]*1000
                else:
                    error = (euclid_dis[i][j] - real_dis[i][j]) / real_dis[i][j]

                if euclid_dis[i][j] != 0:
                    move[i][0] += ((location[i][0] - location[j][0]) / euclid_dis[i][j]) * error
                    move[i][1] += ((location[i][1] - location[j][1]) / euclid_dis[i][j]) * error

                error_sum += abs(error)

        # move rate control
        counter += 1
        if counter > 20:
            move_rate += 0.00001
            logging.info("loop: %d, turn up move_rate, now move_rate is %f" % (m, move_rate))
            counter = 0
        logging.debug("loop: %d, sum of error: %f" % (m, error_sum))
        if last_error and last_error < error_sum:
            move_rate -= 0.00001
            counter = 0
            logging.info("loop: %d, last_error < error_sum !! turn down move_rate, now move_rate is %f" % (m, move_rate))
            if move_rate < 0.00001:
                logging.warning("loop will break!!" % m)
                np.save(save_name, location)
                logging.info("save graph as '%s'" % save_name)
                break
        last_error = error_sum

        # move
        for i in range(user_len):
            # print("user[%d] move %s" % (i, str(move_rate*move[i])))
            location[i] -= move_rate*move[i]

    np.save(save_name, location)
    return location


# trans location to image.
def draw2d(location, matches=(), save_name='draw2d.png'):
    if not matches:
        matches = [[i for i in range(location.shape[0])]]
        k = 0
        color = [(0, 0, 0)]
    else:
        k = len(matches)
        color = []
        for i in range(k):
            color.append((np.random.randint(256), np.random.randint(256), np.random.randint(256)))
        logging.info("color is: %s" % str(color))

    img = Image.new('RGB', (5000, 5000), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(len(location)):
        x = (location[i][0] + 0.5) * 2000
        y = (location[i][1] + 0.5) * 2000
        group = 0
        for j in range(k):
            if i in matches[j]:
                group = j
        draw.text((x, y), str(i+1), color[group])
    img.save(save_name, 'PNG')

###########################
# run

if __name__ == "__main__":
    # data2npy()
    data_set = load_data()
    local = np.load('graph.npy')
    # local = mds2d(data_set, distance=sq_distance, move_rate=0.001)
    points = np.load('k_means_5.npy')
    # points = k_means(data_set, k=5)
    """
    print(sim_pearson([], []))
    print(sim_pearson([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))
    print(sim_pearson([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]))
    print(sim_pearson([1, 0, 0, 0, 1], [0, 1, 1, 1, 10]))
    print(sim_pearson([1, 2, 5, 4, 3], [6, 1, 5, 7, 9]))
    print(sim_pearson([1, 1, 2, 1, 1], [1, 1, 2, 1, 1]))
    print(sim_pearson([1, 2, 1, 0, 0], [1, 2, 0, 0, 0]))
    """
    # ps = k_means(data_set, 5, distance=sq_distance)
    # ps = k_means(data_set, 20)
    # print("k-means end!")
    print("start match users")
    m = matche_user(data_set, points, distance=sq_distance)
    for line in m:
        print(line)
    draw2d(local, m)
