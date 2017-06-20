#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from split_days import *

# -- read in the on/off files
ons  = [np.load("output/good_ons_{0:04}.npy".format(i)) for i in range(20)]
offs = [np.load("output/good_offs_{0:04}.npy".format(i)) for i in range(20)]


# -- split nights
nights_ons  = []
nights_offs = []

for ii in range(20):
    print("splitting days for {0:2} of 20...".format(ii))
    dind_lo      = list(split_days(ii))
    dind_hi      = dind_lo[1:] + [ons[ii].shape[0]]
    nights_ons  += [ons[ii][i:j] for i, j in zip(dind_lo, dind_hi)]
    nights_offs += [offs[ii][i:j] for i, j in zip(dind_lo, dind_hi)]


# -- stack together the number over nights
st    = 1000
nons  = [i[st:].sum() for i in nights_ons]
noffs = [i[st:].sum() for i in nights_offs]


# -- plot ons/offs
fig, ax = plt.subplots(figsize=(10., 6.0))
pnts_on,  = ax.plot(noffs, "-o")
pnts_off, = ax.plot(nons, "-o")
ax.set_xlabel("night index")
ax.set_ylabel("number of transitions")
ax.legend((pnts_on, pnts_off), ("off transitions", "on transitions"))
fig.canvas.draw()
