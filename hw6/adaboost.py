#!/usr/bin/env python3

import random
from math import sqrt, log


def read_data(filename):
    with open(filename, 'r') as f:
        l = [line.split() for line in f.readlines()]
    X, y = zip(*[((float(ll[0]), float(ll[1])), int(ll[2])) for ll in l])  # ((x1, x2), y)
    return X, y


def stump_train(X, y, u):
    err_min = float('+inf')
    list_st = []
    for dim in range(len(X[0])):
        key = list(range(len(X)))
        key.sort(key=lambda i: X[i][dim])
        for s in [1, -1]:
            err_left = 0.0
            err_right = sum([u[i] for i in range(len(X)) if y[i] != s])
            if err_min > err_left + err_right:
                err_min = err_left + err_right
                list_st = [(dim, s, X[key[0]][dim] - 0.000001)]
            elif err_min == err_left + err_right:
                list_st.append([(dim, s, X[key[0]][dim] - 0.000001)])
            for i in range(len(X)):
                if y[key[i]] == s:
                    err_left = err_left + u[key[i]]
                else:
                    err_right = err_right - u[key[i]]
                if err_min > err_left + err_right:
                    err_min = err_left + err_right
                    list_st = [(dim, s, X[key[i]][dim])]
                elif err_min == err_left + err_right:
                    list_st.append([(dim, s, X[key[i]][dim])])
    return err_min, random.choice(list_st)


def stump_predict(model, x):
    dim, s, theta = model
    if x[dim] > theta: return s
    return -s


def sgn(x):
    if x > 0: return 1
    return -1


def adaboost_predict(models, x):
    return sgn(sum([alpha * stump_predict(model, x)
        for model, alpha in models]))


def main():
    X_train, y_train = read_data('hw6_train.dat')
    X_test, y_test = read_data('hw6_test.dat')
    u = [1 / len(X_train)] * len(X_train)
    models = []
    for T in range(300):
        err, model = stump_train(X_train, y_train, u)
        err_list = list(map(lambda k: stump_predict(model, k[0]) != k[1], zip(X_train, y_train)))
        eps = sum([uu for uu, err in zip(u, err_list) if err]) / sum(u)
        ratio = sqrt((1 - eps) / eps)
        for i in range(len(u)):
            if err_list[i]:
                u[i] = u[i] * ratio
            else:
                u[i] = u[i] / ratio
        models.append((model, log(ratio)))
        print("Round %3d: E_in(g) = %f, E_in(G) = %f, E_out(G) = %f, U = %f, alpha = %f" % (
            T,
            sum(err_list) / len(X_train),
            sum([adaboost_predict(models, x) != yy for x, yy in zip(X_train, y_train)]) / len(X_train),
            sum([adaboost_predict(models, x) != yy for x, yy in zip(X_test, y_test)]) / len(X_test),
            sum(u),
            log(ratio)
            ))


if __name__ == '__main__': main()
