from __future__ import print_function, division

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import nsfg
import mystats


def ReadFemResp(dct_file="data/2002FemResp.dct", dat_file="data/2002FemResp.dat.gz", nrows=None):
    dct = mystats.ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression="gzip", nrows=nrows)
    CleanFemResp(df)
    return df


def CleanFemResp(df):
    pass


def ValidatePregnum(resp):
    # read the pregnancy frame
    preg = nsfg.ReadFemPreg()
    print(preg)
    print(preg.info())
    print(preg.describe())
    # make the map from caseid to list of pregnancy indices
    preg_map = nsfg.MakePregMap(preg)
    print(preg_map)

    # iterate through the respondent pregnum series
    for index, pregnum in resp.pregnum.items():
        caseid = resp.caseid[index]
        indices = preg_map[caseid]

        if len(indices) != pregnum:
            print(caseid, len(indices), pregnum)
            return False

    return True


def validatedata(script):
    resp = ReadFemResp()
    assert len(resp) == 7643
    assert resp.pregnum.value_counts()[1] == 1267
    assert ValidatePregnum(resp)
    print("%s: All tests passed." % script)


def myNSFG():
    df = nsfg.ReadFemPreg()
    # print(df.columns)
    print(df.info())

    df["agepreg"] /= 100
    na_vals = [97, 98, 99]
    df["birthwgt_lb"].replace(na_vals, np.nan, inplace=True)
    df["birthwgt_oz"].replace(na_vals, np.nan, inplace=True)
    df["totalwgt_lb"] = df["birthwgt_lb"] + df["birthwgt_oz"] / 16.0

    # print(pregordr.describe())
    # print(df["birthwgt_lb"].describe())
    # print(df["birthwgt_lb"].isnull())
    # print(df["birthwgt_lb"].value_counts(sort=False))
    # print(df["outcome"].value_counts().sort_index())
    # print(df["birthwgt_lb"].value_counts().sort_index())

    d = defaultdict(list)
    for index, caseid in df.caseid.iteritems():
        d[caseid].append(index)
    caseid = 10229
    indices = d[caseid]
    print(df.outcome[indices].values)

    # sns.distplot(df["agepreg"])
    # sns.distplot(df["birthwgt_lb"])
    # plt.hist(df['agepreg'], bins=100)
    # plt.show()


def main(script):
    myNSFG()
    # validatedata(script)


if __name__ == "__main__":
    main(*sys.argv)
