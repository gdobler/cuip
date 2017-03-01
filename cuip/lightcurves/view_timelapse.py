#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scipy.ndimage.measurements as spm
from cuip.cuip.registration.uo_tools import read_raw


lind = 3245 # see get_subset.py


# -- get the source labels
nrow = 2160
ncol = 4096
buff = 20
wins = os.path.join(os.getenv("CUIP_SUPPLEMENTARY"), "window_labels.out")
srcs = np.zeros((nrow, ncol), dtype=bool)
srcs[buff:-buff, buff:-buff] = np.fromfile(wins, int) \
    .reshape(nrow - 2 * buff, ncol - 2 * buff) \
    .astype(bool)
labs = spm.label(srcs)
nlab = labs[1]


# -- open registration dictionary
reg = pd.read_csv("../registration/output/register_0000.csv", 
                  parse_dates=["timestamp"])


# -- utilities
deg2rad  = np.pi / 180.
nro2     = nrow // 2
nco2     = ncol // 2
cgr, rgr = np.meshgrid(range(ncol), range(nrow))
rlabs    = rgr[srcs] - nro2
clabs    = cgr[srcs] - nco2
llabs    = labs[0][srcs]
rot      = np.zeros_like(labs[0])


# -- initialize the plotting window
plt.close("all")
fig, ax = plt.subplots(figsize=(5, 5), num=1)
fig.subplots_adjust(0, 0, 1, 1)
ax.axis("off")
im = ax.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
fig.canvas.draw()
plt.show()

# -- 
# -- For each time
# -- 

ii = 0

# -- apply registration
rec  = reg.iloc[ii]
ct   = np.cos(-rec.dtheta * deg2rad)
st   = np.sin(-rec.dtheta * deg2rad)
img  = read_raw(rec.fpath, rec.fname)
rsrc = (rlabs * ct - clabs * st - rec.drow + nro2).round().astype(int)
csrc = (rlabs * st + clabs * ct - rec.dcol + nco2).round().astype(int)
gind = (rsrc >= 0) & (rsrc < nrow) & (csrc >= 0) & (csrc < ncol)
rsrc = rsrc[gind]
csrc = csrc[gind]
lsrc = llabs[gind]

rot[rsrc, csrc] = lsrc

# -- get the centroid of the source in question
coms = np.array(spm.center_of_mass(rot > 0, rot, np.unique(lsrc))) \
    .T.round().astype(int)


# -- plot a timelapse centered on that source of radius ~50
r0 = coms[0, lind - 1]
c0 = coms[1, lind - 1]
slen = 50

im.set_data(img[r0-slen:r0+slen, c0-slen:c0+slen])
fig.canvas.draw()
plt.pause(1e-3)
