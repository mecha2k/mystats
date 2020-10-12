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


def MeanError(estimates, actual):
    errors = [estimate - actual for estimate in estimates]
    return np.mean(errors)


def RMSE(estimates, actual):
    e2 = [(estimate - actual) ** 2 for estimate in estimates]
    mse = np.mean(e2)
    return math.sqrt(mse)


def Estimate1(n=7, m=1000):
    mu = 0
    sigma = 1

    means = []
    medians = []
    for _ in range(m):
        xs = [random.gauss(mu, sigma) for _ in range(n)]
        xbar = np.mean(xs)
        median = np.median(xs)
        means.append(xbar)
        medians.append(median)

    print("Experiment 1")
    print("rmse xbar", RMSE(means, mu))
    print("rmse median", RMSE(medians, mu))


def Estimate2(n=7, m=1000):
    mu = 0
    sigma = 1

    estimates1 = []
    estimates2 = []
    for _ in range(m):
        xs = [random.gauss(mu, sigma) for _ in range(n)]
        biased = np.var(xs)
        unbiased = np.var(xs, ddof=1)
        estimates1.append(biased)
        estimates2.append(unbiased)

    print("Experiment 2")
    print("mean error biased", MeanError(estimates1, sigma ** 2))
    print("mean error unbiased", MeanError(estimates2, sigma ** 2))


def Estimate3(n=7, m=1000):
    lam = 2
    means = []
    medians = []
    for _ in range(m):
        xs = np.random.exponential(1 / lam, n)
        L = 1 / np.mean(xs)
        Lm = math.log(2) / np.median(xs)
        means.append(L)
        medians.append(Lm)

    print("Experiment 3")
    print("rmse L", RMSE(means, lam))
    print("rmse Lm", RMSE(medians, lam))
    print("mean error L", MeanError(means, lam))
    print("mean error Lm", MeanError(medians, lam))


def SimulateSample(mu=90, sigma=7.5, n=9, m=1000):
    def VertLine(x, y=1):
        myplots.Plot([x, x], [0, y], color="0.8", linewidth=3)

    means = []
    for _ in range(m):
        xs = np.random.normal(mu, sigma, n)
        xbar = np.mean(xs)
        means.append(xbar)

    stderr = RMSE(means, mu)
    print("standard error", stderr)
    cdf = mystats.Cdf(means)
    ci = cdf.Percentile(5), cdf.Percentile(95)
    print("confidence interval", ci)
    VertLine(ci[0])
    VertLine(ci[1])

    # plot the CDF
    myplots.Cdf(cdf)
    myplots.Save(
        root="estimation1", xlabel="sample mean", ylabel="CDF", title="Sampling distribution"
    )


def main(script):
    random.seed(100)
    np.random.seed(100)

    Estimate1()
    Estimate2()
    Estimate3(m=1000)
    SimulateSample()


if __name__ == "__main__":
    main(*sys.argv)


# def main():
#     thinkstats2.RandomSeed(17)

#     Estimate1()
#     Estimate2()
#     Estimate3(m=1000)
#     SimulateSample()


# if __name__ == '__main__':
#     main()
