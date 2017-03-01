#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.ndimage.measurements as spm

def assign_bbls():
    """
    Return the BBL labels for the sources.
    """

    # -- read in the source labels and get the center of mass
    nrow = 2160
    ncol = 4096
    buff = 20
    wins = os.path.join(os.getenv("CUIP_SUPPLEMENTARY"), "window_labels.out")
    srcs = np.zeros((nrow, ncol), dtype=bool)
    srcs[buff:-buff, buff:-buff] = np.fromfile(wins, int) \
        .reshape(nrow - 2 * buff, ncol - 2 * buff) \
        .astype(bool)
    labs = spm.label(srcs)
    coms = np.array(spm.center_of_mass(labs[0] > 0, labs[0],
                                       np.arange(1, labs[1]+1))) \
                                       .T.round().astype(int)

    # -- read in the BBLs and assign
    bname = os.path.join("..", "data", "12_3_14_bblgrid_clean.npy")
    bmap  = np.zeros((nrow, ncol))
    bmap[buff:-buff, buff:-buff] = np.load(bname)

    return bmap[coms[0], coms[1]].astype(int)
