#!/usr/bin/env python3

import random
import sys
from multiprocessing import Process, Manager


def sgn(x):
    if x > 0: return 1.0
    return -1.0


def dot(x, y):
    return sum([a * b for (a, b) in zip(x, y)])


def error(vec, entry):
    dim = len(entry) - 1
    return sgn(dot(vec, entry[:dim])) != entry[dim]


def train(data, eta=1.0):
    dimensions = len(data[0]) - 1
    vec = [0.0] * dimensions
    updates = 0
    while True:
        mistake = False
        for entry in data:
            if error(vec, entry):
                mistake = True
                vec = [w + eta * entry[dimensions] * x for (w, x) in zip(vec, entry[:dimensions])]
                updates = updates + 1
        if mistake: continue
        break
    return (vec, updates)


def count_error(vec, data):
    count = 0
    for entry in data:
        if error(vec, entry): count = count + 1
    return count


def train_pocket(data, max_updates):
    dim = len(data[0]) - 1
    vec = [0.0] * dim
    updates = 0
    w_pocket = vec
    err_count = count_error(w_pocket, data)
    while updates < max_updates and err_count > 0:
        entry = random.choice(data)
        if error(vec, entry):
            vec = [w + entry[dim] * x for (w, x) in zip(vec, entry[:dim])]
            new_err_count = count_error(vec, data)
            updates = updates + 1
            if new_err_count < err_count:
                w_pocket = vec
                err_count = new_err_count
    return (w_pocket, updates, err_count)


def read_data(filename):
    f = open(filename)
    data = [[float(x) for x in ['1.0'] + line.split()] for line in f.readlines()]
    f.close()
    return data


def prob15():
    data = read_data('data/hw1_15_train.dat')
    print("w = %s\nupdates = %d" % train(data))


def prob1617(eta):
    data = read_data('data/hw1_15_train.dat')
    total_updates = 0
    for i in range(2000):
        data_shuf = data[:]
        random.shuffle(data_shuf)
        (vec, updates) = train(data_shuf, eta)
        total_updates = total_updates + updates
    print("avg. updates = %f" % (float(total_updates) / 2000))


def prob18_thread(data, test_data, err_count_list):
    w_pocket = train_pocket(data, 50)[0]
    err_count_list.append(count_error(w_pocket, test_data))


def prob18():
    data = read_data('data/hw1_18_train.dat')
    test_data = read_data('data/hw1_18_test.dat')
    num_threads = 20
    err_count_list = Manager().list()
    for i in range(2000 // num_threads):
        print("%d/%d" % (i, 2000 // num_threads))
        threads = [Process(target=prob18_thread, args=(data, test_data,
            err_count_list)) for i in range(num_threads)]
        for t in threads: t.start()
        for t in threads: t.join()
    print("error rate = %f" % (float(sum(err_count_list)) / 2000 /
        len(test_data)))


def main():
    fmap = {
        '15': prob15,
        '16': lambda: prob1617(1.0),
        '17': lambda: prob1617(0.5),
        '18': prob18,
    }
    try:
        fmap[sys.argv[1]]()
    except KeyError:
        print("Usage: %s <prob_num>" % sys.argv[0])


if __name__ == "__main__":
    main()
