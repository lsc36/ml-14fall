#!/usr/bin/env python3

import sys
from multiprocessing import Pool
import numpy as np


def read_data(filename):
    f = open(filename)
    data = [[float(x) for x in line.split()] for line in f.readlines()]
    f.close()
    X = np.array([x[:-1] for x in data])
    y = np.array([x[-1] for x in data])
    return (X, y)


def sgn(x):
    if x > 0: return 1.0
    return -1.0


def theta(s):
    return 1.0 / (1.0 + np.exp(-s))


def logreg(X, y, eta, T, thres=1e-6):
    n = len(X)
    dim = len(X[0])
    w = np.array([0] * dim)
    for i in range(T):
        grad = sum([theta(-y[i] * w.dot(X[i])) * (-y[i] * X[i])
            for i in range(n)]) / n
        if abs(np.linalg.norm(grad)) < thres: break
        w = w - eta * grad
        if i % 100 == 0:
            print("grad = %s" % grad)
            print("|grad| = %f" % np.linalg.norm(grad))
            print("w_%d = %s" % (i, w))
    return w


def prob1819(eta):
    X, y = read_data('hw3_train.dat')
    w = logreg(X, y, eta, 2000)
    X, y = read_data('hw3_test.dat')
    n = len(X)
    E_out = sum([sgn(w.dot(X[i])) != y[i] for i in range(n)]) / n
    print("E_out = %f" % E_out)


def main():
    fmap = {
        '18': lambda: prob1819(0.001),
        '19': lambda: prob1819(0.01),
        #'20': prob20,
    }
    try:
        fmap[sys.argv[1]]()
    except (KeyError, IndexError) as e:
        print("Usage: %s <prob_num>" % sys.argv[0])


if __name__ == "__main__":
    main()
