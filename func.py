# -*- coding: utf-8 -*-
# Python3.5

import numpy as np
import logging
from ctypes import *
from PIL import Image, ImageDraw

import setting

move_point = cdll.LoadLibrary('./move_point.so').move_point
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

"""
# return a dict of category's info
# e.g: {'0': 'unknown', '1': 'Action', ... '18': 'Western'}
def load_category_name(src_name='u.genre'):
    category = {}
    temp = []
    with open(data_path + src_name) as f:
        while 1:
            l = f.readline()
            if not l[:-1]:
                break
            temp.append(l[:-1].split('|'))
        print(temp)
        for c in temp:
            category[c[1]] = c[0]
    return category
"""


# return (item_category, name_dict)
# item_category.shape: (item_len, category), np.array of item include which category
# name_dict: key-value: {item id: item name}
def load_item_info(src_name='u.item'):
    category_temp = []
    name_temp = []
    with open(data_path + src_name) as f:
        while 1:
            l = f.readline()
            if not l:
                break
            category_temp.append(l[:-1].split('|')[5:])
            name_temp.append(l.split('|')[1])
        item_category = np.array(category_temp, dtype=np.uint8)
        name_dict = {}
        for i in range(len(name_temp)):
            name_dict[i] = name_temp[i]
        logging.debug("category_info.shape: %s" % str(item_category.shape))
        logging.debug("length of name_dict: %d" % len(name_dict))
    return item_category, name_dict


# return a np.array(shape = (user_len, category_len)) of how user like of each category
# item_category.shape = (item_len, category_len), np.array of each item included in which category
def user_favourite_array(data, item_category):
    user_len = data.shape[0]
    item_len = data.shape[1]
    category_len = item_category.shape[1]
    user_favourite = np.zeros((user_len, category_len), dtype=np.float32)

    for i in range(user_len):
        sum_category = np.zeros(category_len, dtype=np.float32)
        for j in range(item_len):
            if data[i][j] == 0:
                continue
            sum_category += data[i][j] * item_category[j]
        user_favourite[i] = sum_category / sum(sum_category)

    return user_favourite


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
    return np.sqrt(sum((x-y)**2))


# sub-function in k-means. get data and k-means's center, return list[group][users].
def matche_user(data, centers, distance=sim_pearson):
    k = centers.shape[0]
    user_len = data.shape[0]
    matches = [[] for i in range(k)]
    for i in range(user_len):
        min_distance = 9999
        min_center = k + 1
        for j in range(k):
            d = distance(data[i], centers[j])
            if d < min_distance:
                min_distance = d
                min_center = j
        matches[min_center].append(i)
    return matches


# k-means algorithm. get data and k. return np.array of k center.
def k_means(data, k, distance=sim_pearson, init_method=np.random.randint, init_para=6, save=True):
    user_len = data.shape[0]
    item_len = data.shape[1]

    # init centers
    centers = np.zeros((k, item_len), dtype=np.float32)
    for i in range(k):
        # center value: 0~5
        centers[i] = [init_method(init_para) for j in range(item_len)]

    last_matches = [[] for i in range(k)]
    logging.debug("centers init complete.")

    loop = 0
    while 1:
        loop += 1
        # matches
        logging.info("matching... loop: %d" % loop)
        matches = matche_user(data, centers, distance=distance)

        if matches == last_matches:
            logging.info("cluster complete! loop: %d" % loop)
            break
        else:
            logging.debug("clusting... loop: %d" % loop)
            last_matches = matches

        # move center(center of group)
        # for every center
        logging.info("move centers... loop: %d" % loop)

        if distance == sim_pearson:
            # move center by Ctypes (distance == Pearson Correlation Coefficient)
            # init data
            matches_np = np.zeros((k, user_len), dtype=np.int32)
            matches_len = np.zeros(k, dtype=np.int32)
            for i in range(k):
                length = len(matches[i])
                matches_len[i] = length
                matches_np[i, :length] = matches[i]
            logging.debug("move centers by c-functions... loop: %d" % loop)
            # void move_point(int data[], int user_len, int item_len, int k,
            #                 int matches[], int matches_len[], float centers[])
            move_point(data.ctypes.data_as(POINTER(c_int)),
                       user_len, item_len, k,
                       matches_np.ctypes.data_as(POINTER(c_int)),
                       matches_len.ctypes.data_as(POINTER(c_int)),
                       centers.ctypes.data_as(POINTER(c_float)))

        else:
            # move center (distance == Euclid distance)
            centers = np.zeros((k, item_len), dtype=np.float32)
            for i in range(len(matches)):
                for j in range(len(matches[i])):
                    centers[i] += data[matches[i][j]]
                centers[i] /= len(matches[i])
            if loop > 20:
                break

        """
        # move center by Python (distance == Pearson Correlation Coefficient)
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
                centers[i][j] = sum(sum_)/num
        """
    if save:
        np.save('k_means_k%d.npy' % k, centers)
        logging.info("saved k-means centers: %s" % 'k_means_k%d.npy' % k)
    return centers


"""
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
"""


# trans location to image.
def draw2d(location, matches=(), save_name='draw2d.png', local_bias=0):
    if not matches:
        matches = [[i for i in range(location.shape[0])]]
        k = 0
        color = [(0, 0, 0)]
    else:
        k = len(matches)
        color = []
        for i in range(k):
            color.append((np.random.randint(256), np.random.randint(256), np.random.randint(256)))
        logging.debug("color is: %s" % str(color))

    img = Image.new('RGB', (1000, 1000), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i in range(len(location)):
        x = (location[i][0]+local_bias) * 1000
        y = (location[i][1]+local_bias) * 1000
        group = 0
        for j in range(k):
            if i in matches[j]:
                group = j
        draw.text((x, y), str(i+1), color[group])
    img.save(save_name, 'PNG')


# parameter: data: np.array, user-item data; user_id: 0~ int, group_list: list of k-means users group which user_id in.
# return a set() include movie which target user don't rate and at least two group members rated.
def rec_movie_set(data, user_id, group_list):
    item_len = data.shape[1]
    item_set = set()
    group_rate = np.zeros(item_len, dtype=np.float32)
    for user in group_list:
        group_rate += data[user]
    for i in range(len(data[user_id])):
        if data[user_id][i] != 0 and group_rate[i] > 10:
            item_set.add(i)
    return item_set


# recommend: weighted rating by user
def recommend(data, user_id, matches, user_favourite=(), item_category=(), rec_num=10):
    # in which center?
    c = len(matches) + 1
    for i in range(len(matches)):
        if user_id in matches[i]:
            c = i

    # user-rating
    rec_set = rec_movie_set(data, user_id, matches[c])
    rec_list = []
    for item in rec_set:
        rating = 0
        num = 0
        for user in matches[c]:
            if user == user_id: continue
            rating += data[user][item]
            if data[user][item] != 0: num += 1
        rec_list.append([float(rating)/num, item])
        logging.debug("add movie: %d by rating: %d/%d = %f in user: %d" % (item, rating, num, rating/num, user_id+1))

    # add weight of category
    if user_favourite != () and item_category != ():
        for x in rec_list:
            # x == [rating, item]
            weight = sum(user_favourite[user_id]*item_category[x[1]])
            logging.debug("trans movie: %d by rating: %f -> %f in user: %d" % (x[1], x[0], weight*x[0], user_id+1))
            x[0] *= weight

    # sort by rating
    rec_list.sort()

    # return list[[rating, item_id], [rating, item_id] ...]
    return rec_list[-rec_num:][::-1]


# test if functions can running actually.
# create a random 2-d data_set (value: 0~1, float) --> k-means --> draw 2d graph.
def test_2d_data(data_len=600, k=10, save_name="k_means_test.png"):
    logging.info("Start test_2d_data()...")
    data_set = [[np.random.random(), np.random.random()] for i in range(data_len)]
    data_set = np.array(data_set, dtype=np.float32)
    centers = k_means(data_set, k=k, distance=sq_distance,
                      init_method=np.random.random, init_para=None, save=False)
    m = matche_user(data_set, centers, distance=sq_distance)
    for line in m:
        logging.debug("group member: %s" % str(line))
    draw2d(data_set, m, save_name=save_name)


def test_real_data(data_name='data.npy', k=10, rec_users=(1, 345, 579, 900), use_cache=True):
    logging.info("Start test_real_data()...")
    logging.info("load data...")
    data_set = load_data(data_name)

    logging.info("load information of movies...")
    item_category, name = load_item_info()

    logging.info("calculate favourite(by category) of users...")
    user_favourite = user_favourite_array(data_set, item_category)

    try:
        if not use_cache:
            raise FileNotFoundError
        centers = np.load('k_means_k%d.npy' % k)
        logging.info("load k-means centers...")
    except FileNotFoundError:
        logging.info("k-means centers file not found. create new k-means centers file...")
        centers = k_means(data_set, k=k)

    logging.info("create k-means groups list... wait a minute")
    group_list = matche_user(data_set, centers)

    # print recommend message
    for user in rec_users:
        rec_set = recommend(data_set, user_id=user-1, matches=group_list, user_favourite=user_favourite,
                            item_category=item_category, rec_num=20)
        logging.info("")
        logging.info("For user: %d, Recommend movie:" % user)
        for rating, item in rec_set:
            logging.info("%s  ,  rating: %f" % (name[item], rating))

###########################
# run

if __name__ == "__main__":
    # test_2d_data()
    test_real_data()
