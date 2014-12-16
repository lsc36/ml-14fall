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


def get_margin(model, kernel):
    ww = sum([coef[0] ** 2 * kernel([sv[1], sv[2]], [sv[1], sv[2]])
        for sv, coef in zip(model.get_SV(), model.get_sv_coef())])
    return 1 / np.sqrt(ww)


def rbf_kernel(x1, x2, gamma=100):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)


def prob15():
    X = [row[1:] for row in d_train]
    y = [1 if row[0] == 0.0 else 0 for row in d_train]
    model = svm_train(y, X, '-s 0 -h 0 -c 0.01 -t 0')
    w = sum([coef * np.array([sv[1], sv[2]])
        for sv, coef in zip(model.get_SV(), model.get_sv_coef())])
    print("|w| = %f" % np.linalg.norm(w))


def prob1617():
    X = [row[1:] for row in d_train]
    for k in [0.0, 2.0, 4.0, 6.0, 8.0]:
        y = [1 if row[0] == k else 0 for row in d_train]
        model = svm_train(y, X, '-s 0 -h 0 -c 0.01 -t 1 -d 2 -g 1 -r 1')
        p_labels, p_acc, p_vals = svm_predict(y, X, model)
        print("%d vs. not %d: E_in = %f%%" % (k, k, 100.0 - p_acc[0]))
        print("sum of alpha = %f" % sum([abs(a[0]) for a in model.get_sv_coef()]))


def prob1819():
    X_train = [row[1:] for row in d_train]
    y_train = [1 if row[0] == 0.0 else 0 for row in d_train]
    X_test = [row[1:] for row in d_test]
    y_test = [1 if row[0] == 0.0 else 0 for row in d_test]
    for c in ['0.001', '0.01', '0.1', '1', '10']:
        model = svm_train(y_train, X_train, '-s 0 -h 0 -c %s -t 2 -g 100' % c)
        p_labels, p_acc, p_vals = svm_predict(y_test, X_test, model)
        print("C = %s: margin = %f, E_out = %f%%"
            % (c, get_margin(model, rbf_kernel), 100.0 - p_acc[0]))


def main():
    fmap = {
        '15': prob15,
        '16': prob1617,
        '17': prob1617,
        '18': prob1819,
        '19': prob1819,
    }
    try:
        fmap[sys.argv[1]]()
    except (KeyError, IndexError) as e:
        print("Usage: %s <prob_num>" % sys.argv[0])


if __name__ == '__main__':
    main()
