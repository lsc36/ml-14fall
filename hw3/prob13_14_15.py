#!/usr/bin/env python3

import random
from multiprocessing import Pool
import numpy as np


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


def main():
    n_test = 1000
    n_threads = 8
    pool = Pool(n_threads)
    avg_E_in = sum(pool.map(prob13_thread, range(n_test))) / n_test
    print("prob13. avg. E_in = %f" % avg_E_in)


if __name__ == '__main__':
    main()
