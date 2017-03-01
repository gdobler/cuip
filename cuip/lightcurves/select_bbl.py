#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from cuip.cuip.registration.register import get_reference_image

def select_bbl():
    """
    Return BBL at selected position.
    """

    # -- set suplemental data directory
    supl = os.getenv("CUIP_SUPPLEMENTARY")

    # -- get reference image
    ref = get_reference_image()

    # -- get the BBL map
    buff  = 20
    bname = os.path.join(supl, "12_3_14_bblgrid_clean.npy")
    bmap  = np.zeros((ref.shape[0], ref.shape[1]))
    bmap[buff:-buff, buff:-buff] = np.load(bname)

    # -- initialize the plot
    xs = 8.0
    ys = xs * float(ref.shape[0]) / float(ref.shape[1])
    fig, ax = plt.subplots(figsize=(xs, ys))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    im = ax.imshow(ref)
    plt.show()

    # -- select
    cind, rind = [int(round(i)) for i in ginput(1, -1)[0]]

    return int(bmap[rind, cind])
