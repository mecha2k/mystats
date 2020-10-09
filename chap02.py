from __future__ import print_function, division

import sys
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


def distributions():
    preg = nsfg.ReadFemPreg()
    print(preg.describe())
    print(preg["outcome"].value_counts())

    live = preg[preg.outcome == 1]
    birthwgt = live["birthwgt_lb"].dropna().astype(int)
    print(birthwgt.value_counts())

    print(live.describe())
    print(preg.loc[preg.outcome == 1].info())

    hist = {}
    for x in birthwgt:
        hist[x] = hist.get(x, 0) + 1

    # plt.hist(birthwgt, bins=15, rwidth=0.9, label="birthwgt")
    # plt.title("birth weight")
    # plt.xlabel("weight(pounds)")
    # plt.ylabel("count")
    # plt.legend()
    # plt.show()

    # plt.hist(np.floor(live["agepreg"]), bins=34, label="agepreg", rwidth=0.9)
    # plt.show()

    # sns.distplot(np.floor(live["prglngth"]), label="prglenth")
    # plt.show()

    firsts = live[live.birthord == 1]
    others = live[live.birthord != 1]

    plt.hist(
        firsts["prglngth"].astype(int),
        bins=30,
        range=(20, 50),
        label="firsts",
        alpha=0.5,
        rwidth=0.9,
    )
    plt.hist(
        others["prglngth"].astype(int),
        bins=30,
        range=(20, 50),
        label="others",
        alpha=0.5,
        rwidth=0.9,
    )
    plt.title("Histogram")
    plt.xlabel("weeks")
    plt.ylabel("frequency")
    plt.legend()
    plt.show()


def main(script):
    distributions()


if __name__ == "__main__":
    main(*sys.argv)
