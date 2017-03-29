#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from datetime import datetime
from scipy.misc import imsave
from cuip.cuip.utils.misc import query_db

def read_raw(fname):
    """
    Return mem mapped image.
    """
    return np.memmap(fname, np.uint8, mode="r") \
        .reshape(2160, 4096, 3)[..., ::-1]

# -- set start and end times
syn   = "\n  syntax is \n    python make_timelapse.py y1.m1.d1 y2.m2.d2\n"

try:
    date1 = sys.argv[1]
    date2 = sys.argv[2]
except:
    print(syn)
    sys.exit()

if "." not in date1 or "." not in date2:
    print(syn)
    sys.exit()

y1, m1, d1 = [int(i) for i in date1.split(".")]
y2, m2, d2 = [int(i) for i in date2.split(".")]

# -- query the database                                                     
print("getting filelist from database")
db = os.getenv("CUIP_DBNAME")
fl  = query_db(db, datetime(y1, m1, d1, 19, 0), datetime(y2, m2, d2, 5, 0), 
               columns=["fname", "fpath", "timestamp"])
nfl = len(fl)

# -- take every minute (6th) image
sub = fl[::6]
nsub = len(sub)

# -- make jpgs
base = "img_{0:05}.jpg"
for ii in range(nsub):
    if (ii + 1) % 10 == 0:
        print("\r{0:5} of {1}".format(ii + 1, nsub)),
        sys.stdout.flush()
    imsave(base.format(ii), 
           read_raw(os.path.join(sub.iloc[ii].fpath, 
                                 sub.iloc[ii].fname))[::4, ::4])
print("")

# -- make movie
cmd = "ffmpeg -r 30 -i img_%05d.jpg -qscale 0 night_{0}_{1}.mp4"
os.system(cmd.format(date1, date2))
os.system("rm *.jpg")
