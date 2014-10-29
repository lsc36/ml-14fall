#!/usr/bin/env python3

import random
import sys
from multiprocessing import Pool


def sgn(x):
    if x > 0: return 1.0
    return -1.0


def sgn_noise(x):
    if random.randrange(5) == 0:  # 20%
        return -sgn(x)
    return sgn(x)


def stump(dataset):
    dataset.sort()
    err_min = len(dataset)
    list_st = []
    for s in [1, -1]:
        for t in range(len(dataset) + 1):
            err = 0
            for i in range(t):
                if dataset[i][1] != -s: err = err + 1
            for i in range(t, len(dataset)):
                if dataset[i][1] != s: err = err + 1
            if t == 0: theta = dataset[0][0] - 0.000001
            else: theta = dataset[t - 1][0]
            if err_min > err:
                err_min = err
                list_st = [(s, theta)]
            elif err_min == err:
                list_st.append((s, theta))
    return (err_min, random.choice(list_st))


def prob1718_thread(arg):
    dataset = [(x, sgn_noise(x)) for x in
        [random.uniform(-1.0, 1.0) for i in range(20)]
    ]
    return stump(dataset)


def prob1718():
    n_threads = 8
    pool = Pool(n_threads)
    res_list = pool.map(prob1718_thread, range(5000))
    print("avg. E_in = %f"
        % (sum([err / 20 for (err, (s, t)) in res_list]) / 5000))
    print("avg. E_out = %f"
        % (sum([0.5 + 0.3 * s * (abs(t) - 1)
            for (err, (s, t)) in res_list]) / 5000))


def read_data(filename):
    f = open(filename)
    data = [[float(x) for x in line.split()] for line in f.readlines()]
    f.close()
    return data


def prob1920():
    d_train = read_data("data/hw2_train.dat")
    dim = len(d_train[0]) - 1
    list_est = []
    for i in range(dim):
        list_est.append((stump([(d[i], d[dim]) for d in d_train]), i))
    list_est.sort()
    list_est_min = [est for est in list_est if est[0] == list_est[0][0]]
    (err, (s, t)), i = random.choice(list_est_min)
    print("E_in = %f, s = %d, t = %f, i = %d" % (err / len(d_train), s, t, i))

    d_test = read_data("data/hw2_test.dat")
    err = 0
    for d in d_test:
        if s * sgn(d[i] - t) != d[dim]: err = err + 1
    print("est. E_out = %f" % (err / len(d_test)))


def main():
    fmap = {
        '17': prob1718,
        '18': prob1718,
        '19': prob1920,
        '20': prob1920,
    }
    try:
        fmap[sys.argv[1]]()
    except (KeyError, IndexError) as e:
        print("Usage: %s <prob_num>" % sys.argv[0])


if __name__ == "__main__": main()
