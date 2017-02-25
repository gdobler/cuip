#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import scipy.ndimage.measurements as ndm
from cuip.cuip.registration.uo_tools import read_raw

ind = 0

# -- read in the registration results
reg  = pd.read_csv(os.path.join("..", "registration", "output",
                                "register_{0:04}.csv".format(ind)),
                   parse_dates=["timestamp"])
nobs = len(reg)


# -- read in the source labels
nrow = 2160
ncol = 4096
buff = 20
wins = os.path.join(os.getenv("CUIP_SUPPLEMENTARY"), "window_labels.out")
srcs = np.zeros((nrow, ncol), dtype=bool)
srcs[buff:-buff, buff:-buff] = np.fromfile(wins, int) \
    .reshape(nrow - 2 * buff, ncol - 2 * buff) \
    .astype(bool)
labs = ndm.label(srcs)
nlab = labs[1]


# -- set the ouput file
oname = os.path.join("output", "light_curves_{0:04}.npy".format(ind))


# -- initialize lightcurve array
lcs  = np.zeros((nobs, nlab, 3), dtype=float) - 9999


# -- utilities
deg2rad = np.pi / 180.
nro2    = nrow // 2
nco2    = ncol // 2


# -- initialize the rows and columns grids (the nrow//2 and ncol//2
#    performs rotation about the center of the image)
cgr, rgr = np.meshgrid(range(ncol), range(nrow))
rlabs    = rgr[srcs] - nro2
clabs    = cgr[srcs] - nco2
llabs    = labs[0][srcs]
rot      = np.zeros_like(labs[0])


# -- read in image
for ii in range(nobs):

    print("  extracting brightensses for obs {0} of {1}".format(ii, nobs))

    if reg.iloc[ii].drow == -9999:
        continue

    rot[...] = 0.
    rec      = reg.iloc[ii]
    ct       = np.cos(-rec.dtheta * deg2rad)
    st       = np.sin(-rec.dtheta * deg2rad)
    img      = read_raw(rec.fpath, rec.fname)

    # -- rotate the source labels (the nrow//2 and ncol//2 performs
    #    rotation about the center of the image)
    rsrc = (rlabs * ct - clabs * st - rec.drow + nro2).round().astype(int)
    csrc = (rlabs * st + clabs * ct - rec.dcol + nco2).round().astype(int)
    gind = (rsrc >= 0) & (rsrc < nrow) & (csrc >= 0) & (csrc < ncol)
    rsrc = rsrc[gind]
    csrc = csrc[gind]
    lsrc = llabs[gind]

    rot[rsrc, csrc] = lsrc

    # -- get brightnesses
    lun = np.unique(lsrc)
    lum = np.array([ndm.mean(img[..., i], rot, lun) for i in [0, 1, 2]]).T

    # -- set indices of extracted sources to their values
    lcs[ii, lun - 1] = lum

    # -- periodically write to file
    if (ii + 1) % 100 == 0:
        np.save(oname, lcs[:ii])


# -- # -- # -- # -- # -- 
#tlab = rot[1430, 1875]
tlab = rot[1560, 1770]
foo  = rot == tlab
trrng = (rgr[foo].min(), rgr[foo].max())
tcrng = (cgr[foo].min(), cgr[foo].max())

pred = img[trrng[0]:trrng[1]+1, tcrng[0]:tcrng[1]+1].mean(0).mean(0)


