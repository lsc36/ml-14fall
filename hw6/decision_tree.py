#!/usr/bin/env python3

import random
from collections import Counter


def impurity(y):
    cnt = Counter(y)
    return 1 - (cnt[-1] / len(y)) ** 2 - (cnt[1] / len(y)) ** 2


def uniq(l):
    return len(set(l)) == 1


def stump_train_gini(X, y):
    err_min = float('+inf')
    list_st = []
    key = list(range(len(X)))
    for dim in range(len(X[0])):
        key.sort(key=lambda i: X[i][dim])
        y_ord = [y[key[i]] for i in range(len(X))]
        for t in range(1, len(y)):
            err = t * impurity(y_ord[:t]) + (len(y) - t) * impurity(y_ord[t:])
            if err_min > err:
                err_min = err
                list_st = [(dim, X[key[t - 1]][dim])]
            elif err_min == err:
                list_st.append((dim, X[key[t - 1]][dim]))
    return err_min, random.choice(list_st)


def stump_predict(model, x):
    dim, theta = model
    if x[dim] > theta: return 1
    return -1


class DecisionTree(object):

    def __init__(self, X, y):
        if uniq(y):
            self.branch = 0
            self.label = y[0]
        elif uniq(X):
            self.branch = 0
            self.label = Counter(y).most_common()[0][0]
        else:
            err, self.model = stump_train_gini(X, y)
            self.lchild = DecisionTree(
                *zip(*[(x, yy) for x, yy in zip(X, y) if stump_predict(self.model, x) == -1])
                )
            self.rchild = DecisionTree(
                *zip(*[(x, yy) for x, yy in zip(X, y) if stump_predict(self.model, x) == 1])
                )
            self.branch = self.lchild.branch + self.rchild.branch + 1

    def predict(self, x):
        if self.branch > 0:
            if stump_predict(self.model, x) == -1:
                return self.lchild.predict(x)
            else:
                return self.rchild.predict(x)
        else:
            return self.label


def read_data(filename):
    with open(filename, 'r') as f:
        l = [line.split() for line in f.readlines()]
    X, y = zip(*[((float(ll[0]), float(ll[1])), int(ll[2])) for ll in l])  # ((x1, x2), y)
    return X, y


def main():
    X_train, y_train = read_data('hw6_train.dat')
    X_test, y_test = read_data('hw6_test.dat')
    tree = DecisionTree(X_train, y_train)
    print("%d branches" % tree.branch)
    print("E_in = %f, E_out = %f" % (
        sum([1 for x, yy in zip(X_train, y_train) if tree.predict(x) != yy]) / len(X_train),
        sum([1 for x, yy in zip(X_test, y_test) if tree.predict(x) != yy]) / len(X_test),
        ))


if __name__ == '__main__': main()
