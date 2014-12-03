#!/usr/bin/env python3

import sys
import numpy as np
from multiprocessing import Pool


n_threads = 8


def read_data(filename):
    f = open(filename)
    data = [[float(x) for x in line.split()] for line in f.readlines()]
    f.close()
    X = np.array([[1.0] + x[:-1] for x in data])
    y = np.array([x[-1] for x in data])
    return (X, y)


d_train = read_data('hw4_train.dat')
d_test = read_data('hw4_test.dat')
d_train_s = (d_train[0][:120], d_train[1][:120])
d_val = (d_train[0][-80:], d_train[1][-80:])


def sgn(x):
    if x > 0: return 1.0
    return -1.0


def reglinreg(dataset, l):
    X, y = dataset
    dim = len(X[0])
    w_reg = np.linalg.inv(
        X.transpose().dot(X) + l*np.identity(dim)
        ).dot(X.transpose()).dot(y)
    return w_reg


def err(dataset, w):
    X, y = dataset
    n = len(X)
    cnt = sum([sgn(X[i].dot(w)) != y[i] for i in range(n)])
    return cnt / n


def prob13():
    w_reg = reglinreg(d_train, 10)
    print("E_in = %f, E_out = %f" % (err(d_train, w_reg), err(d_test, w_reg)))


def prob14_thread(l):
    w_reg = reglinreg(d_train, l)
    return (l, err(d_train, w_reg), err(d_test, w_reg))


def prob1415(p):
    pool = Pool(n_threads)
    results = pool.map(prob14_thread, [10**x for x in range(-10, 3)])
    if p == 14:  # sort by E_in
        results.sort(key=lambda x: (x[1], -x[0]))
    else:  # sort by E_out
        results.sort(key=lambda x: (x[2], -x[0]))
    for entry in results:
        print("lambda = %E, E_in = %f, E_out = %f" % entry)


def prob16_thread(l):
    w_reg = reglinreg(d_train_s, l)
    return (l, err(d_train_s, w_reg), err(d_val, w_reg), err(d_test, w_reg))


def prob1617(p):
    pool = Pool(n_threads)
    results = pool.map(prob16_thread, [10**x for x in range(-10, 3)])
    if p == 16:  # sort by E_train
        results.sort(key=lambda x: (x[1], -x[0]))
    else:  # sort by E_val
        results.sort(key=lambda x: (x[2], -x[0]))
    for entry in results:
        print("lambda = %E, E_train = %f, E_val = %f, E_out = %f" % entry)


def prob18():
    # lambda = 1.0
    w_reg = reglinreg(d_train, 1.0)
    print("E_in = %f, E_out = %f" % (err(d_train, w_reg), err(d_test, w_reg)))


def prob19_thread(l):
    errsum = 0.0
    for i in range(5):
        w_reg = reglinreg((
            np.vstack([d_train[0][:i*40], d_train[0][(i+1)*40:]]),
            np.hstack([d_train[1][:i*40], d_train[1][(i+1)*40:]])
            ), l)
        errsum += err((d_train[0][i*40:(i+1)*40], d_train[1][i*40:(i+1)*40]),
            w_reg)
    return (l, errsum / 5)


def prob19():
    pool = Pool(n_threads)
    results = pool.map(prob19_thread, [10**x for x in range(-10, 3)])
    results.sort(key=lambda x: (x[1], -x[0]))
    for entry in results:
        print("lambda = %E, E_cv = %f" % entry)


def prob20():
    # lambda = 1e-8
    w_reg = reglinreg(d_train, 1e-8)
    print("E_in = %f, E_out = %f" % (err(d_train, w_reg), err(d_test, w_reg)))


def main():
    fmap = {
        '13': prob13,
        '14': lambda: prob1415(14),
        '15': lambda: prob1415(15),
        '16': lambda: prob1617(16),
        '17': lambda: prob1617(17),
        '18': prob18,
        '19': prob19,
        '20': prob20,
    }
    try:
        fmap[sys.argv[1]]()
    except (KeyError, IndexError) as e:
        print("Usage: %s <prob_num>" % sys.argv[0])


if __name__ == '__main__':
    main()
