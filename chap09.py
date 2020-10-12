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

import nsfg2
import first
import brfss
import mystats
import myplots

warnings.simplefilter(action="ignore", category=FutureWarning)


class CoinTest(mystats.HypothesisTest):
    def TestStatistic(self, data):
        heads, tails = data
        test_stat = abs(heads - tails)
        return test_stat

    def RunModel(self):
        heads, tails = self.data
        n = heads + tails
        sample = [random.choice("HT") for _ in range(n)]
        hist = mystats.Hist(sample)
        data = hist["H"], hist["T"]
        return data


class DiffMeansPermute(mystats.HypothesisTest):
    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[: self.n], self.pool[self.n :]
        return data


class DiffMeansOneSided(DiffMeansPermute):
    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = group1.mean() - group2.mean()
        return test_stat


class DiffStdPermute(DiffMeansPermute):
    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = group1.std() - group2.std()
        return test_stat


class CorrelationPermute(mystats.HypothesisTest):
    def TestStatistic(self, data):
        xs, ys = data
        test_stat = abs(mystats.Corr(xs, ys))
        return test_stat

    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys


class DiceTest(mystats.HypothesisTest):
    def TestStatistic(self, data):
        observed = data
        n = sum(observed)
        expected = np.ones(6) * n / 6
        test_stat = sum(abs(observed - expected))
        return test_stat

    def RunModel(self):
        n = sum(self.data)
        values = [1, 2, 3, 4, 5, 6]
        rolls = np.random.choice(values, n, replace=True)
        hist = mystats.Hist(rolls)
        freqs = hist.Freqs(values)
        return freqs


class DiceChiTest(DiceTest):
    def TestStatistic(self, data):
        observed = data
        n = sum(observed)
        expected = np.ones(6) * n / 6
        test_stat = sum((observed - expected) ** 2 / expected)
        return test_stat


class PregLengthTest(mystats.HypothesisTest):
    def TestStatistic(self, data):
        firsts, others = data
        stat = self.ChiSquared(firsts) + self.ChiSquared(others)
        return stat

    def ChiSquared(self, lengths):
        hist = mystats.Hist(lengths)
        observed = np.array(hist.Freqs(self.values))
        expected = self.expected_probs * len(lengths)
        stat = sum((observed - expected) ** 2 / expected)
        return stat

    def MakeModel(self):
        firsts, others = self.data
        self.n = len(firsts)
        self.pool = np.hstack((firsts, others))

        pmf = mystats.Pmf(self.pool)
        self.values = range(35, 44)
        self.expected_probs = np.array(pmf.Probs(self.values))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[: self.n], self.pool[self.n :]
        return data


def RunDiceTest():
    data = [8, 9, 19, 5, 8, 11]
    dt = DiceTest(data)
    print("dice test", dt.PValue(iters=10000))
    dt = DiceChiTest(data)
    print("dice chi test", dt.PValue(iters=10000))


def FalseNegRate(data, num_runs=1000):
    group1, group2 = data
    count = 0

    for i in range(num_runs):
        sample1 = mystats.Resample(group1)
        sample2 = mystats.Resample(group2)
        ht = DiffMeansPermute((sample1, sample2))
        p_value = ht.PValue(iters=101)
        if p_value > 0.05:
            count += 1

    return count / num_runs


def PrintTest(p_value, ht):
    print("p-value =", p_value)
    print("actual =", ht.actual)
    print("ts max =", ht.MaxTestStat())


def RunTests(data, iters=1000):
    # test the difference in means
    ht = DiffMeansPermute(data)
    p_value = ht.PValue(iters=iters)
    print("\nmeans permute two-sided")
    PrintTest(p_value, ht)

    ht.PlotCdf()
    myplots.Save(
        root="hypothesis1",
        title="Permutation test",
        xlabel="difference in means (weeks)",
        ylabel="CDF",
        legend=False,
    )

    # test the difference in means one-sided
    ht = DiffMeansOneSided(data)
    p_value = ht.PValue(iters=iters)
    print("\nmeans permute one-sided")
    PrintTest(p_value, ht)

    # test the difference in std
    ht = DiffStdPermute(data)
    p_value = ht.PValue(iters=iters)
    print("\nstd permute one-sided")
    PrintTest(p_value, ht)


def ReplicateTests():
    live, firsts, others = nsfg2.MakeFrames()

    # compare pregnancy lengths
    print("\nprglngth2")
    data = firsts.prglngth.values, others.prglngth.values
    ht = DiffMeansPermute(data)
    p_value = ht.PValue(iters=1000)
    print("means permute two-sided")
    PrintTest(p_value, ht)

    print("\nbirth weight 2")
    data = (firsts.totalwgt_lb.dropna().values, others.totalwgt_lb.dropna().values)
    ht = DiffMeansPermute(data)
    p_value = ht.PValue(iters=1000)
    print("means permute two-sided")
    PrintTest(p_value, ht)

    # test correlation
    live2 = live.dropna(subset=["agepreg", "totalwgt_lb"])
    data = live2.agepreg.values, live2.totalwgt_lb.values
    ht = CorrelationPermute(data)
    p_value = ht.PValue()
    print("\nage weight correlation 2")
    PrintTest(p_value, ht)

    # compare pregnancy lengths (chi-squared)
    data = firsts.prglngth.values, others.prglngth.values
    ht = PregLengthTest(data)
    p_value = ht.PValue()
    print("\npregnancy length chi-squared 2")
    PrintTest(p_value, ht)


def main(script):
    random.seed(100)
    np.random.seed(100)

    # run the coin test
    ct = CoinTest((140, 110))
    pvalue = ct.PValue()
    print("coin test p-value", pvalue)

    # compare pregnancy lengths
    print("\nprglngth")
    live, firsts, others = first.MakeFrames()
    data = firsts.prglngth.values, others.prglngth.values
    RunTests(data)

    # compare birth weights
    print("\nbirth weight")
    data = (firsts.totalwgt_lb.dropna().values, others.totalwgt_lb.dropna().values)
    ht = DiffMeansPermute(data)
    p_value = ht.PValue(iters=1000)
    print("means permute two-sided")
    PrintTest(p_value, ht)

    # test correlation
    live2 = live.dropna(subset=["agepreg", "totalwgt_lb"])
    data = live2.agepreg.values, live2.totalwgt_lb.values
    ht = CorrelationPermute(data)
    p_value = ht.PValue()
    print("\nage weight correlation")
    print("n=", len(live2))
    PrintTest(p_value, ht)

    # run the dice test
    RunDiceTest()

    # compare pregnancy lengths (chi-squared)
    data = firsts.prglngth.values, others.prglngth.values
    ht = PregLengthTest(data)
    p_value = ht.PValue()
    print("\npregnancy length chi-squared")
    PrintTest(p_value, ht)

    # compute the false negative rate for difference in pregnancy length
    data = firsts.prglngth.values, others.prglngth.values
    neg_rate = FalseNegRate(data)
    print("false neg rate", neg_rate)

    # run the tests with new nsfg data
    ReplicateTests()


if __name__ == "__main__":
    main(*sys.argv)
