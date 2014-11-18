#!/usr/bin/env python3

import random
from multiprocessing import Pool
import sys
import numpy as np


n_test = 1000
n_threads = 8


def sgn(x):
    if x > 0: return 1.0
    return -1.0


def gen_dataset(n):
    l = [(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(n)]
    l = [((x[0], x[1]), sgn(x[0]**2 + x[1]**2 - 0.6)) for x in l]
    for i in random.sample(range(n), n//10):
        l[i] = (l[i][0], -l[i][1])
    return l


def prob13_thread(arg):
    n = 1000
    dataset = gen_dataset(n)
    X = np.array([[1, entry[0][0], entry[0][1]] for entry in dataset])
    y = np.array([entry[1] for entry in dataset])
    w_lin = np.linalg.pinv(X).dot(y)
    E_in = sum([sgn(w_lin.dot(X[i])) != y[i] for i in range(n)]) / n
    return E_in


def prob13():
    pool = Pool(n_threads)
    avg_E_in = sum(pool.map(prob13_thread, range(n_test))) / n_test
    print("avg. E_in = %f" % avg_E_in)


def prob14():
    n = 1000
    dataset = gen_dataset(n)
    X = np.array([[1, x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2]
        for x, y in dataset])
    y = np.array([entry[1] for entry in dataset])
    w_lin = np.linalg.pinv(X).dot(y)
    print("w_lin = %s" % w_lin)
    return w_lin


def prob15_thread(w_lin):
    n = 1000
    dataset = gen_dataset(n)
    X = np.array([[1, x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2]
        for x, y in dataset])
    y = np.array([entry[1] for entry in dataset])
    E_out = sum([sgn(w_lin.dot(X[i])) != y[i] for i in range(n)]) / n
    return E_out


def prob15():
    w_lin = prob14()
    pool = Pool(n_threads)
    avg_E_out = sum(pool.map(prob15_thread, [w_lin] * n_test)) / n_test
    print("avg. E_out = %f" % avg_E_out)


def main():
    fmap = {
        '13': prob13,
        '14': prob14,
        '15': prob15,
    }
    try:
        fmap[sys.argv[1]]()
    except (KeyError, IndexError) as e:
        print("Usage: %s <prob_num>" % sys.argv[0])


if __name__ == '__main__':
    main()
