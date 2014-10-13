#!/usr/bin/env python3

def sgn(x):
    if x > 0: return 1.0
    return -1.0


def dot(x, y):
    return sum([a * b for (a, b) in zip(x, y)])


def train(data):
    dimensions = len(data[0]) - 1
    vec = [0.0] * dimensions
    updates = 0
    while True:
        mistake = False
        for entry in data:
            if sgn(dot(vec, entry[:dimensions])) != entry[dimensions]:
                mistake = True
                vec = [w + entry[dimensions] * x for (w, x) in zip(vec, entry[:dimensions])]
                updates = updates + 1
        if mistake: continue
        break
    return (vec, updates)


def prob15():
    f = open('data/hw1_15_train.dat')
    data = [[float(x) for x in ['1.0'] + line.split()] for line in f.readlines()]
    f.close()
    print("w = %s\nupdates = %d" % train(data))


def main():
    fmap = {
        '15': prob15,
    }
    import sys
    try:
        fmap[sys.argv[1]]()
    except:
        print("Usage: %s <prob_num>" % sys.argv[0])


if __name__ == "__main__":
    main()
