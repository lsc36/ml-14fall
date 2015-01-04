#!/usr/bin/env python3

import random
from collections import Counter
from multiprocessing import Pool


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

    def err(self, X, y):
        return sum([1 for x, yy in zip(X, y) if self.predict(x) != yy]) / len(X)


def read_data(filename):
    with open(filename, 'r') as f:
        l = [line.split() for line in f.readlines()]
    X, y = zip(*[((float(ll[0]), float(ll[1])), int(ll[2])) for ll in l])  # ((x1, x2), y)
    return X, y


X_train, y_train = read_data('hw6_train.dat')
X_test, y_test = read_data('hw6_test.dat')


def sgn(x):
    if x > 0: return 1
    return -1


def forest_predict(forest, x):
    return sgn(sum([tree.predict(x) for tree in forest]))


def forest_err(forest, X, y):
    return sum([1 for x, yy in zip(X, y) if forest_predict(forest, x) != yy]) / len(X)


def bag(X, y, T):
    Xy = list(zip(X, y))
    for t in range(T):
        yield zip(*[random.choice(Xy) for i in range(len(Xy))])


def rf_thread(tid):
    print(tid)
    forest = list(map(lambda arg: DecisionTree(*arg), bag(X_train, y_train, 300)))
    E_out = forest_err(forest, X_test, y_test)
    for t in range(300):
        if forest[t].err(X_test, y_test) < E_out:
            print("===== Counterexample Found =====")
            break
    return E_out


def main():
    tree = DecisionTree(X_train, y_train)
    print("%d branches" % tree.branch)
    print("E_in = %f, E_out = %f" % (tree.err(X_train, y_train), tree.err(X_test, y_test)))
    print("E_out(G_RF) = %f" % (sum(Pool(8).map(rf_thread, range(100))) / 100))


if __name__ == '__main__': main()
