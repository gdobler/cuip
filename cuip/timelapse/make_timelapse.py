#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from datetime import datetime
from scipy.misc import imsave
from cuip.cuip.utils.misc import query_db

def read_raw(fname):
    return np.memmap(fname, np.uint8, mode="r") \
        .reshape(2160, 4096, 3)[..., ::-1]

# -- set start and end times
sts = "2013.11.02"
ens = "2013.11.03"
st  = datetime.strptime(sts, "%Y.%m.%d")
en  = datetime.strptime(ens, "%Y.%m.%d")

# -- query the database                                                     
print("getting filelist from database")
db = os.getenv("CUIP_DBNAME")
fl  = query_db(db, datetime(2013, 11, 2, 19, 0), datetime(2013, 11, 3, 5, 0), 
               columns=["fname", "fpath", "timestamp"])
nfl = len(fl)

# -- take every minute (6th) image
sub = fl[::6]

# -- make jpgs
base = "img_{0:05}.jpg"
for ii in range(len(sub)):
    if (ii + 1) % 10 == 0:
        print("\r{0:5} of {1}".format(ii + 1, len(sub))),
        sys.stdout.flush()
    imsave(base.format(ii), 
           read_raw(os.path.join(sub.iloc[ii].fpath, 
                                 sub.iloc[ii].fname))[::4, ::4])
print("")
