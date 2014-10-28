#!/usr/bin/env python3

import random
import sys
from multiprocessing import Pool


def sgn(x):
    if x > 0: return 1.0
    return -1.0


def sgn_noise(x):
    if random.randrange(5) == 0:  # 20%
        return -sgn(x)
    return sgn(x)


def prob1718_thread(arg):
    dataset = [(x, sgn_noise(x)) for x in
        [random.uniform(-1.0, 1.0) for i in range(20)]
    ]
    dataset.sort()
    err_min = 20
    list_st = []
    for s in [1, -1]:
        for t in range(21):
            err = 0
            for i in range(t):
                if dataset[i][1] != -s: err = err + 1
            for i in range(t, 20):
                if dataset[i][1] != s: err = err + 1
            if t == 0: theta = -1.0
            else: theta = dataset[t - 1][0]
            if err_min > err:
                err_min = err
                list_st = [(s, theta)]
            elif err_min == err:
                list_st.append((s, theta))
    return (err_min / 20, random.choice(list_st))


def prob1718():
    n_threads = 8
    pool = Pool(n_threads)
    res_list = pool.map(prob1718_thread, range(5000))
    print("avg. E_in = %f"
        % (sum([e_in for (e_in, (s, t)) in res_list]) / 5000))
    print("avg. E_out = %f"
        % (sum([0.5 + 0.3 * s * (abs(t) - 1)
            for (e_in, (s, t)) in res_list]) / 5000))


def main():
    fmap = {
        '17': prob1718,
        '18': prob1718,
    }
    try:
        fmap[sys.argv[1]]()
    except (KeyError, IndexError) as e:
        print("Usage: %s <prob_num>" % sys.argv[0])


if __name__ == "__main__": main()
