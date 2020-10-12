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


def GetHeightWeight(df, hjitter=0.0, wjitter=0.0):
    heights = df.htm3
    if hjitter:
        heights = mystats.Jitter(heights, hjitter)
    weights = df.wtkg2
    if wjitter:
        weights = mystats.Jitter(weights, wjitter)
    return heights, weights


def ScatterPlot(heights, weights, alpha=1.0):
    myplots.Scatter(heights, weights, alpha=alpha)
    myplots.Config(
        xlabel="height (cm)", ylabel="weight (kg)", axis=[140, 210, 20, 200], legend=False
    )


def HexBin(heights, weights, bins=None):
    myplots.HexBin(heights, weights, bins=bins)
    myplots.Config(
        xlabel="height (cm)", ylabel="weight (kg)", axis=[140, 210, 20, 200], legend=False
    )


def MakeFigures(df):
    sample = mystats.SampleRows(df, 5000)

    # simple scatter plot
    myplots.PrePlot(cols=2)
    heights, weights = GetHeightWeight(sample)
    ScatterPlot(heights, weights)

    # scatter plot with jitter
    myplots.SubPlot(2)
    heights, weights = GetHeightWeight(sample, hjitter=1.3, wjitter=0.5)
    ScatterPlot(heights, weights)

    myplots.Save(root="scatter1")

    # with jitter and transparency
    myplots.PrePlot(cols=2)
    ScatterPlot(heights, weights, alpha=0.1)

    # hexbin plot
    myplots.SubPlot(2)
    heights, weights = GetHeightWeight(df, hjitter=1.3, wjitter=0.5)
    HexBin(heights, weights)
    myplots.Save(root="scatter2")


def BinnedPercentiles(df):
    cdf = mystats.Cdf(df.htm3)
    print("Fraction between 140 and 200 cm", cdf[200] - cdf[140])

    bins = np.arange(135, 210, 5)
    indices = np.digitize(df.htm3, bins)
    groups = df.groupby(indices)

    heights = [group.htm3.mean() for i, group in groups][1:-1]
    cdfs = [mystats.Cdf(group.wtkg2) for i, group in groups][1:-1]

    myplots.PrePlot(3)
    for percent in [75, 50, 25]:
        weights = [cdf.Percentile(percent) for cdf in cdfs]
        label = "%dth" % percent
        myplots.Plot(heights, weights, label=label)

    myplots.Save(root="scatter3", xlabel="height (cm)", ylabel="weight (kg)")


def Correlations(df):
    print("pandas cov", df.htm3.cov(df.wtkg2))
    print("NumPy cov", np.cov(df.htm3, df.wtkg2))
    print("mystats Cov", mystats.Cov(df.htm3, df.wtkg2))
    print("pandas corr", df.htm3.corr(df.wtkg2))
    print("NumPy corrcoef", np.corrcoef(df.htm3, df.wtkg2))
    print("mystats Corr", mystats.Corr(df.htm3, df.wtkg2))
    print("pandas corr spearman", df.htm3.corr(df.wtkg2, method="spearman"))
    print("mystats SpearmanCorr", mystats.SpearmanCorr(df.htm3, df.wtkg2))
    print("mystats SpearmanCorr log wtkg3", mystats.SpearmanCorr(df.htm3, np.log(df.wtkg2)))
    print("mystats Corr log wtkg3", mystats.Corr(df.htm3, np.log(df.wtkg2)))


def main(script):
    random.seed(100)
    np.random.seed(100)

    df = brfss.ReadBrfss(nrows=None)
    df = df.dropna(subset=["htm3", "wtkg2"])

    Correlations(df)
    MakeFigures(df)
    BinnedPercentiles(df)


if __name__ == "__main__":
    main(*sys.argv)
