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


def main(script):
    random.seed(100)
    np.random.seed(100)

    df = brfss.ReadBrfss(nrows=None)
    df = df.dropna(subset=["htm3", "wtkg2"])


if __name__ == "__main__":
    main(*sys.argv)
