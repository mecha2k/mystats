import warnings

import sys
import math
import scipy
import bisect
import random
import numpy as np
from numpy.core.fromnumeric import cumsum
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nsfg
import first
import brfss
import mystats
import myplots

warnings.simplefilter(action="ignore", category=FutureWarning)


def PDF(xs, mu=0, sigma=1):
    return scipy.stats.norm.pdf(xs, mu, sigma)


def _underride(d, **options):
    if d is None:
        d = {}
    for key, val in options.items():
        d.setdefault(key, val)
    return d


def plotPDF(mu, sigma, **options):
    n = options.pop("n", 101)
    low, high = options.pop("low", None), options.pop("high", None)
    if low is not None and high is not None:
        xs = np.linspace(low, high, n)
    else:
        xs = options.pop("xs", None)
        if xs is None:
            low, high = mu - 6 * sigma, mu + 6 * sigma
            xs = np.linspace(low, high, n)
    ps = PDF(xs, mu, sigma)

    label = options.pop("label", "")
    options = _underride(options, label=label)
    label = getattr(options, "label", "_nolegend_")
    options = _underride(options, linewidth=3, alpha=0.7, label=label)
    plt.plot(xs, ps, **options)

    sample = [random.gauss(mu, sigma) for _ in range(n)]
    kde = scipy.stats.gaussian_kde(sample)
    ps = kde.evaluate(xs)

    label = "sample KDE"
    options.pop("label")
    options = _underride(options, linewidth=3, alpha=0.7, label=label)
    plt.plot(xs, ps, **options)


def MakePdfExample(n=500):
    # mean and var of women's heights in cm, from the BRFSS
    mean, var = 163, 52.8
    std = math.sqrt(var)

    print(PDF(mean + std, mean, std))
    plotPDF(mean, std, n=n, label="normal")

    myplots.Save(root="pdf_example", xlabel="Height (cm)", ylabel="Density")


def calcMedian(data):
    d = {}
    for x in data:
        d[x] = d.get(x, 0) + 1
    total = sum(d.values())
    for key, val in d.items():
        d[key] = val / total

    xs, freq = zip(*sorted(d.items()))
    xs = np.asarray(xs)
    ps = np.cumsum(freq, dtype=np.float)
    ps = ps / ps[-1]

    index = bisect.bisect_left(ps, 0.5)
    return xs[index]


def calcSkewness(xs):
    return StandardizedMoment(xs, 3)


def RawMoment(xs, k):
    return sum(x ** k for x in xs) / len(xs)


def CentralMoment(xs, k):
    mean = RawMoment(xs, 1)
    return sum((x - mean) ** k for x in xs) / len(xs)


def StandardizedMoment(xs, k):
    var = CentralMoment(xs, 2)
    std = math.sqrt(var)
    return CentralMoment(xs, k) / std ** k


def Summarize(data):
    mean = data.mean()
    std = data.std()
    median = calcMedian(data)
    print("mean", mean)
    print("std", std)
    print("median", median)
    print("skewness", calcSkewness(data))
    print("pearson skewness", mystats.PearsonMedianSkewness(data))

    return mean, median


def ComputeSkewnesses():
    def VertLine(x, y):
        myplots.Plot([x, x], [0, y], color="0.6", linewidth=1)

    live, firsts, others = first.MakeFrames()
    data = live.totalwgt_lb.dropna()
    print("Birth weight")
    mean, median = Summarize(data)

    y = 0.35
    VertLine(mean, y)
    myplots.Text(mean - 0.15, 0.1 * y, "mean", horizontalalignment="right")
    VertLine(median, y)
    myplots.Text(median + 0.1, 0.1 * y, "median", horizontalalignment="left")

    pdf = mystats.EstimatedPdf(data)
    myplots.Pdf(pdf, label="birth weight")
    myplots.Save(root="density_totalwgt_kde", xlabel="lbs", ylabel="PDF")

    df = brfss.ReadBrfss(nrows=None)
    data = df.wtkg2.dropna()
    print("Adult weight")
    mean, median = Summarize(data)

    y = 0.02499
    VertLine(mean, y)
    myplots.Text(mean + 1, 0.1 * y, "mean", horizontalalignment="left")
    VertLine(median, y)
    myplots.Text(median - 1.5, 0.1 * y, "median", horizontalalignment="right")

    pdf = mystats.EstimatedPdf(data)
    myplots.Pdf(pdf, label="adult weight")
    myplots.Save(root="density_wtkg2_kde", xlabel="kg", ylabel="PDF", xlim=[0, 200])


def main(script):
    random.seed(100)
    np.random.seed(100)

    MakePdfExample()
    ComputeSkewnesses()


if __name__ == "__main__":
    main(*sys.argv)
