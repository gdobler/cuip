#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scipy.ndimage.measurements as ndm
from cuip.cuip.registration.uo_tools import read_raw

# -- read in the registration results
reg = pd.read_csv(os.path.join("..", "registration", "output", 
                               "register_0010_0.csv"), 
                  parse_dates=["timestamp"])
rec = reg.iloc[0]

# -- read in an image with a registration offset
img = read_raw(rec.fpath, rec.fname)

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

# -- rotate the source labels (the nrow//2 and ncol//2 performs
#    rotation about the center of the image)
cgr, rgr = np.meshgrid(range(4096), range(2160))
rlabs    = rgr[srcs] - nrow // 2
clabs    = cgr[srcs] - ncol // 2
llabs    = labs[0][srcs]

rsrc = (rlabs * np.cos(-rec.dtheta * np.pi / 180.) - 
        clabs * np.sin(-rec.dtheta * np.pi / 180.) - rec.drow
        + nrow // 2) \
        .round().astype(int)
csrc = (rlabs * np.sin(-rec.dtheta * np.pi / 180.) + 
        clabs * np.cos(-rec.dtheta * np.pi / 180.) - rec.dcol
        + ncol // 2) \
        .round().astype(int)

brind = (rsrc >= 0) & (rsrc < nrow)
bcind = (csrc >= 0) & (csrc < ncol)

rsrc = rsrc[brind & bcind]
csrc = csrc[brind & bcind]
lsrc = llabs[brind & bcind]

result = np.zeros_like(labs[0])
result[rsrc, csrc] = lsrc

# -- get brightnesses
lun = np.unique(lsrc)
lum = np.array([ndm.mean(img[..., i], result, lun) for i in [0, 1, 2]]).T

# -- check alignment
# imshow((5.0*img + result[..., newaxis]*np.array([128.,0,0])).clip(0,255) \
#            .astype(uint8))
