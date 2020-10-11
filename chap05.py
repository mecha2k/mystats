import warnings

import sys
import numpy as np
from numpy.core.fromnumeric import cumsum
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nsfg
import mystats

warnings.simplefilter(action="ignore", category=FutureWarning)


def ReadBabyBoom(filename="data/babyboom.dat"):
    var_info = [
        ("time", 1, 8, int),
        ("sex", 9, 16, int),
        ("weight_g", 17, 24, int),
        ("minutes", 25, 32, int),
    ]
    columns = ["name", "start", "end", "type"]
    variables = pd.DataFrame(var_info, columns=columns)
    variables.end += 1
    dct = mystats.FixedWidthVariables(variables, index_base=1)
    df = dct.ReadFixedWidth(filename, skiprows=59)
    return df


def babyboom():
    df = ReadBabyBoom()
    diffs = df.minutes.diff()

    hist = {}
    for x in diffs:
        hist[x] = hist.get(x, 0) + 1

    total = sum(hist.values())
    for key, value in hist.items():
        hist[key] = value / total

    xs, freq = zip(*sorted(hist.items()))
    ps = cumsum(freq, dtype=np.float)

    plt.plot(xs, ps, alpha=0.5, label="baby")
    plt.show()


def main(script):
    babyboom()


if __name__ == "__main__":
    main(*sys.argv)
