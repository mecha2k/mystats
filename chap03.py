import sys
from matplotlib.pyplot import xlabel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import nsfg
import mystats


def myNSFG():
    df = nsfg.ReadFemPreg()

    df["agepreg"] /= 100
    na_vals = [97, 98, 99]
    df["birthwgt_lb"].replace(na_vals, np.nan, inplace=True)
    df["birthwgt_oz"].replace(na_vals, np.nan, inplace=True)
    df["totalwgt_lb"] = df["birthwgt_lb"] + df["birthwgt_oz"] / 16.0

    d = defaultdict(list)
    for index, caseid in df.caseid.iteritems():
        d[caseid].append(index)


def myPmf():
    preg = nsfg.ReadFemPreg()
    live = preg[preg.outcome == 1]
    firsts = live[live.birthord == 1].prglngth
    others = live[live.birthord != 1].prglngth

    first1 = live.groupby("birthord")
    for name, group in first1:
        print(name, group.count())
    print(first1.size())

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

    labels = ["firsts", "others"]
    for i in range(len(hist_data)):
        xs, ys = zip(*sorted(hist_data[i].items()))
        plt.bar(xs, ys, alpha=0.5, label=labels[i])
        # print(sorted(hist_data[i].items()))

    diffs = []
    weeks = range(35, 46)
    for week in weeks:
        diff = hist_data[0][week] - hist_data[1][week]
        diffs.append(diff)

    plt.bar(weeks, diffs, label="diffs")
    plt.xlim(min(weeks), max(weeks))
    plt.ylim(-0.1, 0.3)
    getattr(plt, "xlabel")("weeks")
    plt.ylabel("probability")
    plt.legend()
    plt.show()


def main(script):
    myPmf()


if __name__ == "__main__":
    main(*sys.argv)
