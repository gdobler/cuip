#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gf

def read_raw(in1, in2=None, nrow=2160, ncol=4096, nwav=3):
    """
    Read in a raw file.
    """

    # -- check for input path
    if in2 is not None:
        fname = os.path.join(in1, in2)
    else:
        fname = in1

    try:
        return np.fromfile(fname, np.uint8).reshape(nrow, ncol, nwav)[:,:,::-1]
    except:
        print("FILE READ ERROR!!!")
        return -1



def high_pass_filter(img, sigma):
    """
    Create a high pass filtered version of an image.
    """

    # -- check if three color
    if img.shape[-1] == 3:
        sigma = (sigma, sigma, 0)

    if img.dtype == np.uint8:
        fimg = img.astype(float)
        return fimg - gf(fimg, sigma)
    else:
        return img - gf(img, sigma)
