#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd

def split_days(file_index):
    """
    For a given lightcurve file index, return the indices which 
    represent the first indices of a night.

    Parameters
    ----------
    file_index : int
        The index of the lightcurve file.
    """

    # -- get the time stamps files
    dpath = os.path.join("..", "registration", "output")
    fname = os.path.join(dpath, "register_{0:04}.csv".format(file_index))

    # -- read the timestamps and calculate the deltas
    data = pd.read_csv(fname, parse_dates=["timestamp"]).timestamp
    dts  = [(data.iloc[i] - data.iloc[i-1]).seconds for i in 
            range(1, len(data))]

    return np.where(np.array(dts) > 10000)[0] + 1
