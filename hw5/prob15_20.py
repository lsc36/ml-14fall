#!/usr/bin/env python

import sys
from svmutil import *
import numpy as np


def read_data(filename):
    f = open(filename, 'r')
    entry = [[float(col) for col in line.split()] for line in f]
    f.close()
    return entry


d_train = read_data('features.train')
d_test = read_data('features.test')


def prob15():
    X = [row[1:] for row in d_train]
    y = [1 if row[0] == 0.0 else 0 for row in d_train]
    model = svm_train(y, X, '-s 0 -h 0 -c 0.01 -t 0')
    w = sum([coef * np.array([sv[1], sv[2]])
        for sv, coef in zip(model.get_SV(), model.get_sv_coef())])
    print("|w| = %f" % np.linalg.norm(w))


def main():
    fmap = {
        '15': prob15,
    }
    try:
        fmap[sys.argv[1]]()
    except (KeyError, IndexError) as e:
        print("Usage: %s <prob_num>" % sys.argv[0])


if __name__ == '__main__':
    main()
