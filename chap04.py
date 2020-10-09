import sys
from matplotlib.pyplot import plot, xlabel
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import cumsum
import seaborn as sns
from collections import defaultdict

import nsfg
import mystats


def myCdf():
    preg = nsfg.ReadFemPreg()
    live = preg[preg.outcome == 1]
    firsts = live[live.birthord == 1]["totalwgt_lb"].dropna()
    others = live[live.birthord != 1]["totalwgt_lb"].dropna()
    print(firsts.describe())

    hist_data = []
    for data in [firsts, others]:
        hist = {}
        for x in data:
            hist[x] = hist.get(x, 0) + 1
        hist_data.append(hist)

    for data in hist_data:
        total = sum(data.values())
        for key, value in data.items():
            data[key] = value / total

    cum_data = []
    for data in hist_data:
        xs, freq = zip(*sorted(data.items()))
        ps = cumsum(freq, dtype=np.float)
        cum_data.append([xs, ps])

    labels = ["firsts", "others"]
    for i in range(len(cum_data)):
        xs, ps = cum_data[i]
        plt.plot(xs, ps, alpha=0.5, label=labels[i])

    plt.xlim(0, 12)
    plt.ylim(0, 1)
    plt.title("birth weight")
    getattr(plt, "xlabel")("weight(pounds)")
    plt.ylabel("cdf")
    plt.legend()
    plt.show()


def main(script):
    myCdf()


if __name__ == "__main__":
    main(*sys.argv)
